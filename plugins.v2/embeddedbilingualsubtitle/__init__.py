import json
import re
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from threading import Event as ThreadEvent
from typing import Any, Dict, List, Optional, Tuple

import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from app.core.config import settings
from app.core.event import eventmanager, Event
from app.log import logger
from app.plugins import _PluginBase
from app.schemas import TransferInfo
from app.schemas.types import EventType, NotificationType
from app.utils.http import RequestUtils
from app.utils.system import SystemUtils


TEXT_SUBTITLE_CODECS = {
    "ass",
    "mov_text",
    "srt",
    "ssa",
    "subrip",
    "text",
    "ttml",
    "webvtt",
}

ENGLISH_HINTS = {"en", "eng", "english", "英文", "英语"}
CHINESE_HINTS = {
    "zh",
    "zh-cn",
    "zh-hans",
    "zh-hant",
    "zho",
    "chi",
    "chs",
    "cht",
    "chinese",
    "中文",
    "简中",
    "简体",
    "繁中",
    "繁体",
}

ffmpeg_lock = threading.Lock()
DEFAULT_SILICONFLOW_URL = "https://api.siliconflow.cn/v1"
DEFAULT_TEST_TEXT = "We need to leave before sunrise, or we will miss the last train."


@dataclass
class SubtitleStream:
    index: int
    codec_name: str
    language: str
    title: str
    is_default: bool
    is_forced: bool

    @property
    def is_text(self) -> bool:
        return (self.codec_name or "").lower() in TEXT_SUBTITLE_CODECS


@dataclass
class SubtitleCue:
    index: int
    start_ms: int
    end_ms: int
    text: str


@dataclass
class ProcessResult:
    file_path: str
    status: str
    mode: str
    reason: str = ""
    output_path: str = ""
    english_stream: str = ""
    chinese_stream: str = ""


def _normalize_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\\N", "\n")
    text = re.sub(r"\{\\.*?\}", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def _parse_srt_timestamp(value: str) -> int:
    match = re.match(r"(\d+):(\d+):(\d+)[,\.](\d+)", value.strip())
    if not match:
        raise ValueError(f"无效时间戳: {value}")
    hour, minute, second, millisecond = [int(part) for part in match.groups()]
    return ((hour * 60 + minute) * 60 + second) * 1000 + millisecond


def _format_srt_timestamp(total_ms: int) -> str:
    total_ms = max(total_ms, 0)
    ms = total_ms % 1000
    total_seconds = total_ms // 1000
    second = total_seconds % 60
    total_minutes = total_seconds // 60
    minute = total_minutes % 60
    hour = total_minutes // 60
    return f"{hour:02d}:{minute:02d}:{second:02d},{ms:03d}"


def _parse_srt_file(file_path: Path) -> List[SubtitleCue]:
    content = file_path.read_text(encoding="utf-8", errors="ignore")
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n{2,}", content.strip())
    cues: List[SubtitleCue] = []
    for block in blocks:
        lines = [line for line in block.split("\n") if line.strip()]
        if len(lines) < 2:
            continue
        pointer = 0
        if re.match(r"^\d+$", lines[0].strip()):
            pointer = 1
        if pointer >= len(lines):
            continue
        if "-->" not in lines[pointer]:
            continue
        start_raw, end_raw = [part.strip() for part in lines[pointer].split("-->", 1)]
        try:
            start_ms = _parse_srt_timestamp(start_raw)
            end_ms = _parse_srt_timestamp(end_raw.split()[0])
        except ValueError:
            continue
        text = _normalize_text("\n".join(lines[pointer + 1:]))
        if not text:
            continue
        cues.append(
            SubtitleCue(
                index=len(cues) + 1,
                start_ms=start_ms,
                end_ms=end_ms,
                text=text,
            )
        )
    return cues


def _write_srt_file(file_path: Path, cues: List[SubtitleCue]) -> None:
    lines: List[str] = []
    for idx, cue in enumerate(cues, start=1):
        lines.append(str(idx))
        lines.append(f"{_format_srt_timestamp(cue.start_ms)} --> {_format_srt_timestamp(cue.end_ms)}")
        lines.append(cue.text.strip())
        lines.append("")
    file_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _extract_json_text(text: str) -> str:
    text = (text or "").strip()
    fenced_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced_match:
        return fenced_match.group(1).strip()
    array_match = re.search(r"(\[[\s\S]*\])", text)
    if array_match:
        return array_match.group(1).strip()
    return text


def _strip_markdown_fences(text: str) -> str:
    text = (text or "").strip()
    fenced_match = re.search(r"```(?:json|text|markdown)?\s*(.*?)```", text, re.S)
    if fenced_match:
        return fenced_match.group(1).strip()
    return text


def _timing_overlap(left: SubtitleCue, right: SubtitleCue) -> int:
    return max(0, min(left.end_ms, right.end_ms) - max(left.start_ms, right.start_ms))


def _timing_score(left: SubtitleCue, right: SubtitleCue) -> float:
    overlap = _timing_overlap(left, right)
    if overlap > 0:
        return overlap - abs(left.start_ms - right.start_ms) * 0.15
    distance = abs(left.start_ms - right.start_ms)
    return 400 - distance if distance <= 1200 else -1


def _build_bilingual_cues(
    english_cues: List[SubtitleCue],
    chinese_cues: List[SubtitleCue],
) -> Tuple[List[SubtitleCue], float]:
    if not english_cues:
        return [], 0
    if not chinese_cues:
        bilingual = [
            SubtitleCue(
                index=cue.index,
                start_ms=cue.start_ms,
                end_ms=cue.end_ms,
                text=cue.text,
            )
            for cue in english_cues
        ]
        return bilingual, 0

    results: List[SubtitleCue] = []
    matched = 0
    right_cursor = 0

    for cue in english_cues:
        best_index = -1
        best_score = -1.0
        window_end = min(len(chinese_cues), right_cursor + 8)
        for idx in range(right_cursor, window_end):
            score = _timing_score(cue, chinese_cues[idx])
            if score > best_score:
                best_score = score
                best_index = idx

        chinese_text = ""
        if best_index >= 0 and best_score >= 0:
            chinese_text = chinese_cues[best_index].text
            right_cursor = max(best_index, right_cursor)
            matched += 1

        merged_text = cue.text if not chinese_text else f"{cue.text}\n{chinese_text}"
        results.append(
            SubtitleCue(
                index=cue.index,
                start_ms=cue.start_ms,
                end_ms=cue.end_ms,
                text=merged_text,
            )
        )

    return results, matched / max(len(english_cues), 1)


class EmbeddedBilingualSubtitle(_PluginBase):
    plugin_name = "内嵌双语字幕合成"
    plugin_desc = "抽取媒体文件内嵌字幕，合成为上英下中的外置双语字幕；缺少中文字幕时可翻译英文字幕。"
    plugin_icon = "bilingual_subtitle.svg"
    plugin_version = "1.1.1"
    plugin_author = "Codex"
    author_url = "https://github.com/openai"
    plugin_config_prefix = "embeddedbilingualsubtitle_"
    plugin_order = 35
    auth_level = 1

    _scheduler = None
    _enabled = False
    _notify = True
    _monitor_transfer = True
    _onlyonce = False
    _overwrite = False
    _keep_temp = False
    _cron = ""
    _scan_paths = ""
    _exclude_paths = ""
    _custom_files = ""
    _output_suffix = "zh.default"
    _ffmpeg_path = "ffmpeg"
    _ffprobe_path = "ffprobe"
    _translate_url = DEFAULT_SILICONFLOW_URL
    _translate_api_key = ""
    _translate_model = ""
    _translate_batch_size = 20
    _translate_timeout = 180
    _test_onlyonce = False
    _test_text = DEFAULT_TEST_TEXT
    _event = ThreadEvent()

    def init_plugin(self, config: dict = None):
        if config:
            self._enabled = bool(config.get("enabled"))
            self._notify = bool(config.get("notify", True))
            self._monitor_transfer = bool(config.get("monitor_transfer", True))
            self._onlyonce = bool(config.get("onlyonce"))
            self._overwrite = bool(config.get("overwrite"))
            self._keep_temp = bool(config.get("keep_temp"))
            self._cron = (config.get("cron") or "").strip()
            self._scan_paths = config.get("scan_paths") or ""
            self._exclude_paths = config.get("exclude_paths") or ""
            self._custom_files = config.get("custom_files") or ""
            self._output_suffix = (config.get("output_suffix") or "zh.default").strip().strip(".")
            self._ffmpeg_path = (config.get("ffmpeg_path") or "ffmpeg").strip()
            self._ffprobe_path = (config.get("ffprobe_path") or "ffprobe").strip()
            self._translate_url = (config.get("translate_url") or DEFAULT_SILICONFLOW_URL).strip()
            self._translate_api_key = (config.get("translate_api_key") or "").strip()
            self._translate_model = (config.get("translate_model") or "").strip()
            self._translate_batch_size = max(1, min(int(config.get("translate_batch_size") or 20), 50))
            self._translate_timeout = max(30, min(int(config.get("translate_timeout") or 180), 600))
            self._test_onlyonce = bool(config.get("test_onlyonce"))
            self._test_text = (config.get("test_text") or DEFAULT_TEST_TEXT).strip()

        self.stop_service()

        if self._enabled or self._onlyonce or self._test_onlyonce:
            self._scheduler = BackgroundScheduler(timezone=settings.TZ)
            if self._cron:
                try:
                    self._scheduler.add_job(
                        func=self.__scheduled_scan,
                        trigger=CronTrigger.from_crontab(self._cron),
                        name="内嵌双语字幕合成",
                    )
                    logger.info(f"内嵌双语字幕合成服务启动，周期：{self._cron}")
                except Exception as err:
                    logger.error(f"内嵌双语字幕合成定时任务配置错误：{str(err)}")
                    self.systemmessage.put(
                        f"内嵌双语字幕合成定时任务配置错误：{str(err)}",
                        title="内嵌双语字幕合成",
                    )
            if self._onlyonce:
                self._scheduler.add_job(
                    func=self.__run_once_scan,
                    trigger="date",
                    run_date=datetime.now(tz=pytz.timezone(settings.TZ)) + timedelta(seconds=3),
                    name="内嵌双语字幕合成立即运行一次",
                )
                self._onlyonce = False
                self.__update_config()
            if self._test_onlyonce:
                self._scheduler.add_job(
                    func=self.__run_translate_test_once,
                    trigger="date",
                    run_date=datetime.now(tz=pytz.timezone(settings.TZ)) + timedelta(seconds=3),
                    name="内嵌双语字幕翻译接口测试",
                )
                self._test_onlyonce = False
                self.__update_config()
            if self._scheduler.get_jobs():
                self._scheduler.print_jobs()
                self._scheduler.start()

    def get_state(self) -> bool:
        return self._enabled

    @staticmethod
    def get_command() -> List[Dict[str, Any]]:
        return [
            {
                "cmd": "/embsub_scan",
                "event": EventType.PluginAction,
                "desc": "立即扫描并生成双语字幕",
                "category": "媒体库",
                "data": {"action": "embsub_scan"},
            },
            {
                "cmd": "/embsub_test_translate",
                "event": EventType.PluginAction,
                "desc": "测试硅基流动翻译接口",
                "category": "工具",
                "data": {"action": "embsub_test_translate"},
            }
        ]

    def get_api(self) -> List[Dict[str, Any]]:
        return [
            {
                "path": "/test_translate",
                "endpoint": self._api_test_translate,
                "methods": ["POST"],
                "summary": "测试硅基流动翻译接口",
                "description": "使用当前保存的模型和测试文本执行一次翻译测试",
            }
        ]

    def get_form(self) -> Tuple[List[dict], Dict[str, Any]]:
        return [
            {
                "component": "VForm",
                "content": [
                    {
                        "component": "VRow",
                        "content": [
                            {
                                "component": "VCol",
                                "props": {"cols": 12, "md": 3},
                                "content": [
                                    {
                                        "component": "VSwitch",
                                        "props": {"model": "enabled", "label": "启用插件"},
                                    }
                                ],
                            },
                            {
                                "component": "VCol",
                                "props": {"cols": 12, "md": 3},
                                "content": [
                                    {
                                        "component": "VSwitch",
                                        "props": {"model": "notify", "label": "发送通知"},
                                    }
                                ],
                            },
                            {
                                "component": "VCol",
                                "props": {"cols": 12, "md": 3},
                                "content": [
                                    {
                                        "component": "VSwitch",
                                        "props": {"model": "monitor_transfer", "label": "监听入库事件"},
                                    }
                                ],
                            },
                            {
                                "component": "VCol",
                                "props": {"cols": 12, "md": 3},
                                "content": [
                                    {
                                        "component": "VSwitch",
                                        "props": {"model": "onlyonce", "label": "立即运行一次"},
                                    }
                                ],
                            },
                            {
                                "component": "VCol",
                                "props": {"cols": 12, "md": 3},
                                "content": [
                                    {
                                        "component": "VSwitch",
                                        "props": {"model": "test_onlyonce", "label": "测试翻译一次"},
                                    }
                                ],
                            },
                        ],
                    },
                    {
                        "component": "VRow",
                        "content": [
                            {
                                "component": "VCol",
                                "props": {"cols": 12, "md": 4},
                                "content": [
                                    {
                                        "component": "VCronField",
                                        "props": {
                                            "model": "cron",
                                            "label": "定时扫描周期",
                                            "placeholder": "5位 cron 表达式，留空关闭",
                                        },
                                    }
                                ],
                            },
                            {
                                "component": "VCol",
                                "props": {"cols": 12, "md": 4},
                                "content": [
                                    {
                                        "component": "VSwitch",
                                        "props": {"model": "overwrite", "label": "覆盖已有输出字幕"},
                                    }
                                ],
                            },
                            {
                                "component": "VCol",
                                "props": {"cols": 12, "md": 4},
                                "content": [
                                    {
                                        "component": "VSwitch",
                                        "props": {"model": "keep_temp", "label": "保留临时字幕文件"},
                                    }
                                ],
                            },
                        ],
                    },
                    {
                        "component": "VRow",
                        "content": [
                            {
                                "component": "VCol",
                                "props": {"cols": 12},
                                "content": [
                                    {
                                        "component": "VTextarea",
                                        "props": {
                                            "model": "scan_paths",
                                            "label": "扫描路径",
                                            "rows": 4,
                                            "placeholder": "每行一个目录，用于定时扫描存量媒体",
                                        },
                                    }
                                ],
                            }
                        ],
                    },
                    {
                        "component": "VRow",
                        "content": [
                            {
                                "component": "VCol",
                                "props": {"cols": 12},
                                "content": [
                                    {
                                        "component": "VTextarea",
                                        "props": {
                                            "model": "exclude_paths",
                                            "label": "排除路径",
                                            "rows": 2,
                                            "placeholder": "每行一个目录",
                                        },
                                    }
                                ],
                            }
                        ],
                    },
                    {
                        "component": "VRow",
                        "content": [
                            {
                                "component": "VCol",
                                "props": {"cols": 12},
                                "content": [
                                    {
                                        "component": "VTextarea",
                                        "props": {
                                            "model": "custom_files",
                                            "label": "立即运行文件列表",
                                            "rows": 3,
                                            "placeholder": "立即运行一次时优先处理这些文件，每行一个完整路径",
                                        },
                                    }
                                ],
                            }
                        ],
                    },
                    {
                        "component": "VRow",
                        "content": [
                            {
                                "component": "VCol",
                                "props": {"cols": 12, "md": 4},
                                "content": [
                                    {
                                        "component": "VTextField",
                                        "props": {
                                            "model": "output_suffix",
                                            "label": "字幕后缀",
                                            "placeholder": "默认 zh.default",
                                        },
                                    }
                                ],
                            },
                            {
                                "component": "VCol",
                                "props": {"cols": 12, "md": 4},
                                "content": [
                                    {
                                        "component": "VTextField",
                                        "props": {
                                            "model": "ffmpeg_path",
                                            "label": "ffmpeg 路径",
                                            "placeholder": "默认 ffmpeg",
                                        },
                                    }
                                ],
                            },
                            {
                                "component": "VCol",
                                "props": {"cols": 12, "md": 4},
                                "content": [
                                    {
                                        "component": "VTextField",
                                        "props": {
                                            "model": "ffprobe_path",
                                            "label": "ffprobe 路径",
                                            "placeholder": "默认 ffprobe",
                                        },
                                    }
                                ],
                            },
                        ],
                    },
                    {
                        "component": "VRow",
                        "content": [
                            {
                                "component": "VCol",
                                "props": {"cols": 12, "md": 4},
                                "content": [
                                    {
                                        "component": "VTextField",
                                        "props": {
                                            "model": "translate_url",
                                            "label": "硅基流动 / OpenAI兼容地址",
                                            "placeholder": "默认 https://api.siliconflow.cn/v1",
                                        },
                                    }
                                ],
                            },
                            {
                                "component": "VCol",
                                "props": {"cols": 12, "md": 4},
                                "content": [
                                    {
                                        "component": "VTextField",
                                        "props": {
                                            "model": "translate_model",
                                            "label": "硅基流动模型",
                                            "placeholder": "填写完整模型名，例如 Qwen/Qwen2.5-72B-Instruct",
                                        },
                                    }
                                ],
                            },
                            {
                                "component": "VCol",
                                "props": {"cols": 12, "md": 4},
                                "content": [
                                    {
                                        "component": "VTextField",
                                        "props": {
                                            "model": "translate_api_key",
                                            "label": "硅基流动 API Key",
                                            "placeholder": "硅基流动必填，其他兼容服务按需填写",
                                            "type": "password",
                                        },
                                    }
                                ],
                            },
                        ],
                    },
                    {
                        "component": "VRow",
                        "content": [
                            {
                                "component": "VCol",
                                "props": {"cols": 12},
                                "content": [
                                    {
                                        "component": "VTextarea",
                                        "props": {
                                            "model": "test_text",
                                            "label": "测试翻译文本",
                                            "rows": 2,
                                            "placeholder": DEFAULT_TEST_TEXT,
                                        },
                                    }
                                ],
                            }
                        ],
                    },
                    {
                        "component": "VRow",
                        "content": [
                            {
                                "component": "VCol",
                                "props": {"cols": 12, "md": 6},
                                "content": [
                                    {
                                        "component": "VTextField",
                                        "props": {
                                            "model": "translate_batch_size",
                                            "label": "翻译批次大小",
                                            "placeholder": "默认 20",
                                        },
                                    }
                                ],
                            },
                            {
                                "component": "VCol",
                                "props": {"cols": 12, "md": 6},
                                "content": [
                                    {
                                        "component": "VTextField",
                                        "props": {
                                            "model": "translate_timeout",
                                            "label": "翻译超时秒数",
                                            "placeholder": "默认 180",
                                        },
                                    }
                                ],
                            },
                        ],
                    },
                    {
                        "component": "VRow",
                        "content": [
                            {
                                "component": "VCol",
                                "props": {"cols": 12},
                                "content": [
                                    {
                                        "component": "VAlert",
                                        "props": {
                                            "type": "info",
                                            "variant": "tonal",
                                            "text": "默认按硅基流动 OpenAI 兼容接口工作。模型名可自定义；保存配置后可通过“测试翻译一次”、命令 /embsub_test_translate 或插件页面按钮验证翻译接口。",
                                        },
                                    }
                                ],
                            }
                        ],
                    },
                ],
            }
        ], {
            "enabled": False,
            "notify": True,
            "monitor_transfer": True,
            "onlyonce": False,
            "test_onlyonce": False,
            "overwrite": False,
            "keep_temp": False,
            "cron": "",
            "scan_paths": "",
            "exclude_paths": "",
            "custom_files": "",
            "output_suffix": "zh.default",
            "ffmpeg_path": "ffmpeg",
            "ffprobe_path": "ffprobe",
            "translate_url": DEFAULT_SILICONFLOW_URL,
            "translate_api_key": "",
            "translate_model": "",
            "translate_batch_size": 20,
            "translate_timeout": 180,
            "test_text": DEFAULT_TEST_TEXT,
        }

    def get_page(self) -> List[dict]:
        latest_test = self.get_data("last_translate_test") or {}
        success = latest_test.get("success")
        success_text = "未测试" if success is None else ("成功" if success else "失败")
        success_color = "success" if success else ("error" if success is False else "info")
        return [
            {
                "component": "div",
                "props": {"class": "d-flex align-center"},
                "content": [
                    {
                        "component": "h2",
                        "props": {"class": "page-title m-0"},
                        "text": "硅基流动翻译测试",
                    },
                    {"component": "VSpacer"},
                    {
                        "component": "VBtn",
                        "props": {
                            "prepend-icon": "mdi-translate",
                            "variant": "tonal",
                        },
                        "text": "立即测试翻译",
                        "events": {
                            "click": {
                                "api": f"plugin/{self.__class__.__name__}/test_translate?apikey={settings.API_TOKEN}",
                                "method": "post",
                            }
                        },
                    },
                ],
            },
            {
                "component": "VRow",
                "content": [
                    {
                        "component": "VCol",
                        "props": {"cols": 12},
                        "content": [
                            {
                                "component": "VAlert",
                                "props": {
                                    "type": success_color,
                                    "variant": "tonal",
                                    "text": (
                                        f"最近测试状态：{success_text}\n"
                                        f"时间：{latest_test.get('time') or '-'}\n"
                                        f"模型：{latest_test.get('model') or '-'}\n"
                                        f"接口：{latest_test.get('endpoint') or '-'}\n"
                                        f"原文：{latest_test.get('input') or '-'}\n"
                                        f"译文：{latest_test.get('output') or '-'}\n"
                                        f"说明：{latest_test.get('reason') or '-'}\n"
                                        "点击按钮后会调用当前保存的配置并覆盖这里的结果。"
                                    ),
                                },
                            }
                        ],
                    }
                ],
            },
        ]

    def _api_test_translate(self) -> dict:
        result = self.__run_translate_test(source="api")
        return {
            "success": result.get("success", False),
            "message": result.get("reason") or result.get("output") or "",
            "data": result,
        }

    def stop_service(self):
        try:
            if self._scheduler:
                self._scheduler.remove_all_jobs()
                if self._scheduler.running:
                    self._event.set()
                    self._scheduler.shutdown()
                    self._event.clear()
                self._scheduler = None
        except Exception as err:
            logger.error(f"停止内嵌双语字幕合成服务失败：{str(err)}")

    @eventmanager.register(EventType.PluginAction)
    def handle_command(self, event: Event = None):
        if not event:
            return
        event_data = event.event_data or {}
        action = event_data.get("action")
        if action == "embsub_test_translate":
            result = self.__run_translate_test(source="command")
            title = "硅基流动翻译测试成功" if result.get("success") else "硅基流动翻译测试失败"
            text = (
                f"模型：{result.get('model') or '-'}\n"
                f"接口：{result.get('endpoint') or '-'}\n"
                f"原文：{result.get('input') or '-'}\n"
                f"译文：{result.get('output') or '-'}\n"
                f"说明：{result.get('reason') or '-'}"
            )
            self.post_message(
                channel=event_data.get("channel"),
                title=title,
                userid=event_data.get("user"),
                text=text,
            )
            return
        if action != "embsub_scan":
            return
        self.post_message(
            channel=event_data.get("channel"),
            title="开始扫描内嵌字幕并生成双语字幕",
            userid=event_data.get("user"),
        )
        results = self.__collect_and_process_paths(source="command")
        success = len([item for item in results if item.status == "success"])
        failed = len([item for item in results if item.status == "failed"])
        skipped = len([item for item in results if item.status == "skipped"])
        self.post_message(
            channel=event_data.get("channel"),
            title="内嵌双语字幕扫描完成",
            userid=event_data.get("user"),
            text=f"成功 {success} 个，失败 {failed} 个，跳过 {skipped} 个。",
        )

    @eventmanager.register(EventType.TransferComplete)
    def handle_transfer_complete(self, event: Event):
        if not self._enabled or not self._monitor_transfer:
            return
        transferinfo: TransferInfo = (event.event_data or {}).get("transferinfo")
        if not transferinfo:
            return
        if transferinfo.target_diritem and transferinfo.target_diritem.storage != "local":
            logger.warn(f"内嵌双语字幕合成不支持非本地存储：{transferinfo.target_diritem.storage}")
            return
        for file_item in transferinfo.file_list_new or []:
            self.__process_single_path(Path(file_item), source="transfer")

    def __run_once_scan(self):
        self.__collect_and_process_paths(source="once")

    def __run_translate_test_once(self):
        self.__run_translate_test(source="config")

    def __scheduled_scan(self):
        self.__collect_and_process_paths(source="cron")

    def __collect_and_process_paths(self, source: str) -> List[ProcessResult]:
        results: List[ProcessResult] = []
        custom_files = [line.strip() for line in self._custom_files.splitlines() if line.strip()]
        if custom_files:
            logger.info(f"内嵌双语字幕合成开始处理自定义文件，共 {len(custom_files)} 个")
            for file_name in custom_files:
                results.append(self.__process_single_path(Path(file_name), source=source))
            self.__notify_batch_summary(results, source=source)
            return results

        scan_paths = [line.strip() for line in self._scan_paths.splitlines() if line.strip()]
        exclude_paths = [Path(line.strip()) for line in self._exclude_paths.splitlines() if line.strip()]
        if not scan_paths:
            logger.warn("内嵌双语字幕合成未配置扫描路径")
            return results

        for path_text in scan_paths:
            if self._event.is_set():
                return results
            root = Path(path_text)
            if not root.exists():
                reason = f"扫描路径不存在：{path_text}"
                logger.warn(reason)
                results.append(ProcessResult(file_path=path_text, status="failed", mode="scan", reason=reason))
                continue
            logger.info(f"开始扫描媒体目录：{root}")
            for file_path in SystemUtils.list_files(root, extensions=settings.RMT_MEDIAEXT):
                if self._event.is_set():
                    return results
                if self.__is_excluded(file_path, exclude_paths):
                    continue
                results.append(self.__process_single_path(file_path, source=source))
        self.__notify_batch_summary(results, source=source)
        return results

    def __is_excluded(self, file_path: Path, exclude_paths: List[Path]) -> bool:
        for exclude_path in exclude_paths:
            try:
                if file_path.is_relative_to(exclude_path):
                    logger.debug(f"{file_path} 位于排除路径中，跳过")
                    return True
            except Exception:
                continue
        return False

    def __process_single_path(self, file_path: Path, source: str) -> ProcessResult:
        logger.info(f"开始处理内嵌字幕：{file_path}")
        try:
            with ffmpeg_lock:
                result = self.__do_process_single_path(file_path=file_path)
        except Exception as err:
            result = ProcessResult(
                file_path=str(file_path),
                status="failed",
                mode="runtime",
                reason=str(err),
            )
        self.__record_history(result)
        self.__notify_single_result(result=result, source=source)
        if result.status == "success":
            logger.info(f"双语字幕生成成功：{result.file_path} -> {result.output_path}")
        elif result.status == "failed":
            logger.error(f"双语字幕生成失败：{result.file_path}，原因：{result.reason}")
        else:
            logger.info(f"双语字幕跳过：{result.file_path}，原因：{result.reason}")
        return result

    def __do_process_single_path(self, file_path: Path) -> ProcessResult:
        if not file_path.exists():
            return ProcessResult(file_path=str(file_path), status="failed", mode="check", reason="文件不存在")
        if file_path.suffix.lower() not in settings.RMT_MEDIAEXT:
            return ProcessResult(file_path=str(file_path), status="skipped", mode="check", reason="不是支持的视频文件")

        output_path = self.__build_output_path(file_path)
        if output_path.exists() and not self._overwrite:
            return ProcessResult(
                file_path=str(file_path),
                status="skipped",
                mode="check",
                reason=f"输出字幕已存在：{output_path.name}",
                output_path=str(output_path),
            )

        temp_dir = Path(settings.TEMP_PATH) / "embeddedbilingualsubtitle"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_prefix = f"{file_path.stem}_{int(datetime.now().timestamp())}_{threading.get_ident()}"
        english_srt = temp_dir / f"{temp_prefix}.eng.srt"
        chinese_srt = temp_dir / f"{temp_prefix}.chi.srt"

        try:
            streams = self.__probe_subtitle_streams(file_path)
            if not streams:
                return ProcessResult(file_path=str(file_path), status="failed", mode="probe", reason="未找到任何内嵌字幕流")

            english_stream = self.__select_stream(streams, target="english")
            if not english_stream:
                return ProcessResult(file_path=str(file_path), status="failed", mode="probe", reason="未找到英文字幕流")
            if not english_stream.is_text:
                return ProcessResult(
                    file_path=str(file_path),
                    status="failed",
                    mode="probe",
                    reason=f"英文字幕流为图片字幕，当前不支持 OCR，codec={english_stream.codec_name}",
                    english_stream=str(english_stream.index),
                )

            chinese_stream = self.__select_stream(streams, target="chinese")
            chinese_text_stream = chinese_stream if chinese_stream and chinese_stream.is_text else None

            self.__extract_stream_to_srt(file_path=file_path, stream=english_stream, output_path=english_srt)
            english_cues = _parse_srt_file(english_srt)
            if not english_cues:
                return ProcessResult(
                    file_path=str(file_path),
                    status="failed",
                    mode="extract",
                    reason="英文字幕流抽取后为空",
                    english_stream=str(english_stream.index),
                )

            if chinese_text_stream:
                self.__extract_stream_to_srt(file_path=file_path, stream=chinese_text_stream, output_path=chinese_srt)
                chinese_cues = _parse_srt_file(chinese_srt)
                if not chinese_cues:
                    return ProcessResult(
                        file_path=str(file_path),
                        status="failed",
                        mode="extract",
                        reason="中文字幕流抽取后为空",
                        english_stream=str(english_stream.index),
                        chinese_stream=str(chinese_text_stream.index),
                    )
                bilingual_cues, coverage = _build_bilingual_cues(english_cues, chinese_cues)
                if coverage < 0.35:
                    return ProcessResult(
                        file_path=str(file_path),
                        status="failed",
                        mode="merge",
                        reason=f"中英字幕时间轴差异过大，匹配覆盖率仅 {coverage:.0%}",
                        english_stream=str(english_stream.index),
                        chinese_stream=str(chinese_text_stream.index),
                    )
                _write_srt_file(output_path, bilingual_cues)
                return ProcessResult(
                    file_path=str(file_path),
                    status="success",
                    mode="merge",
                    reason=f"匹配覆盖率 {coverage:.0%}",
                    output_path=str(output_path),
                    english_stream=str(english_stream.index),
                    chinese_stream=str(chinese_text_stream.index),
                )

            if chinese_stream and not chinese_stream.is_text:
                logger.warn(
                    f"{file_path} 存在中文字幕流 {chinese_stream.index}，但为图片字幕 {chinese_stream.codec_name}，将尝试翻译英文字幕生成双语字幕"
                )

            config_error = self.__translation_config_error()
            if config_error:
                return ProcessResult(
                    file_path=str(file_path),
                    status="failed",
                    mode="translate",
                    reason=f"缺少可用中文字幕，且翻译配置不完整：{config_error}",
                    english_stream=str(english_stream.index),
                    chinese_stream=str(chinese_stream.index) if chinese_stream else "",
                )

            chinese_lines = self.__translate_cues(english_cues)
            bilingual_cues = [
                SubtitleCue(
                    index=cue.index,
                    start_ms=cue.start_ms,
                    end_ms=cue.end_ms,
                    text=f"{cue.text}\n{translated}".strip(),
                )
                for cue, translated in zip(english_cues, chinese_lines)
            ]
            _write_srt_file(output_path, bilingual_cues)
            return ProcessResult(
                file_path=str(file_path),
                status="success",
                mode="translate",
                reason="已使用英文字幕翻译生成中英双语字幕",
                output_path=str(output_path),
                english_stream=str(english_stream.index),
                chinese_stream=str(chinese_stream.index) if chinese_stream else "",
            )
        except Exception as err:
            return ProcessResult(
                file_path=str(file_path),
                status="failed",
                mode="runtime",
                reason=str(err),
                english_stream=str(english_stream.index) if english_stream else "",
                chinese_stream=str(chinese_stream.index) if chinese_stream else "",
            )
        finally:
            if not self._keep_temp:
                for temp_file in [english_srt, chinese_srt]:
                    try:
                        if temp_file.exists():
                            temp_file.unlink()
                    except Exception:
                        pass

    def __probe_subtitle_streams(self, file_path: Path) -> List[SubtitleStream]:
        try:
            command = [
                self._ffprobe_path or "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_streams",
                "-select_streams",
                "s",
                str(file_path),
            ]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
        except FileNotFoundError:
            raise RuntimeError(f"ffprobe 不可用：{self._ffprobe_path}")
        except Exception as err:
            raise RuntimeError(f"执行 ffprobe 失败：{str(err)}")

        if completed.returncode != 0:
            error_message = completed.stderr.strip() or completed.stdout.strip() or "未知错误"
            raise RuntimeError(f"ffprobe 执行失败：{error_message}")

        try:
            data = json.loads(completed.stdout or "{}")
        except json.JSONDecodeError as err:
            raise RuntimeError(f"ffprobe 输出解析失败：{str(err)}")

        streams: List[SubtitleStream] = []
        for item in data.get("streams") or []:
            tags = item.get("tags") or {}
            disposition = item.get("disposition") or {}
            streams.append(
                SubtitleStream(
                    index=int(item.get("index")),
                    codec_name=(item.get("codec_name") or "").lower(),
                    language=(tags.get("language") or "").strip().lower(),
                    title=(tags.get("title") or "").strip(),
                    is_default=bool(disposition.get("default")),
                    is_forced=bool(disposition.get("forced")),
                )
            )
        return streams

    def __select_stream(self, streams: List[SubtitleStream], target: str) -> Optional[SubtitleStream]:
        candidates: List[Tuple[int, SubtitleStream]] = []
        for stream in streams:
            score = self.__score_stream(stream, target=target)
            if score > 0:
                candidates.append((score, stream))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def __score_stream(self, stream: SubtitleStream, target: str) -> int:
        language = (stream.language or "").lower()
        title = (stream.title or "").lower()
        hints = ENGLISH_HINTS if target == "english" else CHINESE_HINTS
        score = 0

        if language in hints:
            score += 120
        elif any(token in language for token in hints):
            score += 80

        if any(token in title for token in hints):
            score += 50

        if stream.is_text:
            score += 30
        if stream.is_default:
            score += 20
        if stream.is_forced:
            score -= 15

        return score

    def __extract_stream_to_srt(self, file_path: Path, stream: SubtitleStream, output_path: Path) -> None:
        try:
            command = [
                self._ffmpeg_path or "ffmpeg",
                "-y",
                "-i",
                str(file_path),
                "-map",
                f"0:{stream.index}",
                "-vn",
                "-an",
                "-c:s",
                "srt",
                str(output_path),
            ]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
        except FileNotFoundError:
            raise RuntimeError(f"ffmpeg 不可用：{self._ffmpeg_path}")
        except Exception as err:
            raise RuntimeError(f"执行 ffmpeg 失败：{str(err)}")

        if completed.returncode != 0 or not output_path.exists():
            message = completed.stderr.strip() or completed.stdout.strip() or "未知错误"
            raise RuntimeError(f"字幕流 {stream.index} 抽取失败：{message.splitlines()[-1]}")

    def __translation_ready(self) -> bool:
        return self.__translation_config_error() is None

    def __translation_config_error(self) -> Optional[str]:
        if not self._translate_url:
            return "未配置翻译接口地址"
        if not self._translate_model:
            return "未配置翻译模型"
        if "siliconflow.cn" in self._translate_url.lower() and not self._translate_api_key:
            return "硅基流动模式下必须填写 API Key"
        return None

    def __run_translate_test(self, source: str) -> dict:
        test_text = (self._test_text or DEFAULT_TEST_TEXT).strip() or DEFAULT_TEST_TEXT
        endpoint = self.__build_translate_endpoint(self._translate_url)
        result = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": source,
            "input": test_text,
            "output": "",
            "success": False,
            "reason": "",
            "model": self._translate_model,
            "endpoint": endpoint,
        }
        try:
            config_error = self.__translation_config_error()
            if config_error:
                raise RuntimeError(config_error)
            translated = self.__translate_batch(
                [SubtitleCue(index=1, start_ms=0, end_ms=1000, text=test_text)]
            )
            result["output"] = translated[0]
            result["success"] = True
            result["reason"] = "测试翻译成功"
            logger.info(f"硅基流动翻译测试成功，模型：{self._translate_model}")
        except Exception as err:
            result["reason"] = str(err)
            logger.error(f"硅基流动翻译测试失败：{str(err)}")

        self.save_data("last_translate_test", result)
        if self._notify:
            self.post_message(
                mtype=NotificationType.Plugin,
                title="硅基流动翻译测试成功" if result["success"] else "硅基流动翻译测试失败",
                text=(
                    f"来源：{source}\n"
                    f"模型：{result.get('model') or '-'}\n"
                    f"接口：{result.get('endpoint') or '-'}\n"
                    f"原文：{result.get('input') or '-'}\n"
                    f"译文：{result.get('output') or '-'}\n"
                    f"说明：{result.get('reason') or '-'}"
                ),
            )
        return result

    def __translate_cues(self, english_cues: List[SubtitleCue]) -> List[str]:
        translations: List[str] = []
        for start in range(0, len(english_cues), self._translate_batch_size):
            batch = english_cues[start:start + self._translate_batch_size]
            translations.extend(self.__translate_batch(batch))
        if len(translations) != len(english_cues):
            raise RuntimeError("翻译结果数量与英文字幕条数不一致")
        return translations

    def __translate_batch(self, cues: List[SubtitleCue]) -> List[str]:
        endpoint = self.__build_translate_endpoint(self._translate_url)
        prompt_lines = [
            f"{cue.index}\t{cue.text.replace(chr(10), ' / ')}"
            for cue in cues
        ]
        payload = {
            "model": self._translate_model,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是专业影视字幕译者。"
                        "请将输入的英文字幕逐条翻译成简体中文。"
                        "必须保持顺序与数量一一对应，不要省略，不要增加解释。"
                        "优先输出每行一条，严格使用“索引<TAB>译文”格式，例如“1\t你好”。"
                        "如果模型支持结构化输出，也可以返回 JSON 数组，格式为"
                        "[{\"index\":1,\"translation\":\"...\"}]。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "请把下面内容翻译成简体中文，保持索引不变，只返回翻译结果：\n"
                        + "\n".join(prompt_lines)
                    ),
                },
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self._translate_api_key:
            headers["Authorization"] = f"Bearer {self._translate_api_key}"

        response = RequestUtils(
            headers=headers,
            timeout=self._translate_timeout,
            proxies=settings.PROXY,
        ).post(endpoint, json=payload)
        if not response:
            raise RuntimeError("翻译接口无响应")
        if response.status_code >= 300:
            raise RuntimeError(f"翻译接口调用失败：HTTP {response.status_code}")

        try:
            response_json = response.json()
        except Exception as err:
            raise RuntimeError(f"翻译接口返回的 JSON 无法解析：{str(err)}")

        message = (((response_json.get("choices") or [{}])[0]).get("message") or {}).get("content") or ""
        if isinstance(message, list):
            message = "".join(part.get("text", "") for part in message if isinstance(part, dict))
        if not message:
            raise RuntimeError("翻译接口未返回有效内容")
        return self.__parse_translate_output(message=message, cues=cues)

    @staticmethod
    def __build_translate_endpoint(base_url: str) -> str:
        normalized = (base_url or "").strip().rstrip("/")
        if normalized.endswith("/chat/completions"):
            return normalized
        if normalized.endswith("/v1"):
            return f"{normalized}/chat/completions"
        return f"{normalized}/v1/chat/completions"

    def __parse_translate_output(self, message: str, cues: List[SubtitleCue]) -> List[str]:
        json_error = None
        try:
            return self.__parse_translate_json(message=message, cues=cues)
        except Exception as err:
            json_error = str(err)

        plain_text = _strip_markdown_fences(message)
        plain_text = plain_text.replace("\r\n", "\n").replace("\r", "\n").strip()
        lines = [line.strip() for line in plain_text.split("\n") if line.strip()]

        indexed_map: Dict[int, str] = {}
        for line in lines:
            indexed = self.__parse_indexed_translation_line(line)
            if indexed:
                index, translation = indexed
                indexed_map[index] = translation

        if indexed_map:
            translations = []
            for cue in cues:
                translation = indexed_map.get(cue.index)
                if not translation:
                    raise RuntimeError(f"翻译结果缺少索引 {cue.index}")
                translations.append(translation)
            return translations

        cleaned_lines = [self.__clean_translation_line(line) for line in lines]
        cleaned_lines = [line for line in cleaned_lines if line]

        if len(cues) == 1:
            single_text = "\n".join(cleaned_lines).strip() or self.__clean_translation_line(plain_text)
            if single_text:
                return [single_text]

        if len(cleaned_lines) == len(cues):
            return cleaned_lines

        if len(cleaned_lines) > len(cues):
            trimmed = cleaned_lines[:len(cues)]
            if all(trimmed):
                return trimmed

        raise RuntimeError(
            f"翻译结果无法匹配字幕条数，期望 {len(cues)} 条，收到 {len(cleaned_lines)} 条；JSON解析错误：{json_error or '-'}"
        )

    def __parse_translate_json(self, message: str, cues: List[SubtitleCue]) -> List[str]:
        items = json.loads(_extract_json_text(message))
        if isinstance(items, dict):
            for key in ["translations", "items", "data", "result"]:
                value = items.get(key)
                if isinstance(value, list):
                    items = value
                    break
            else:
                if len(cues) == 1 and items.get("translation"):
                    return [_normalize_text(str(items.get("translation")))]
        if not isinstance(items, list):
            raise RuntimeError("翻译结果格式错误，返回内容不是数组")

        if items and all(isinstance(item, str) for item in items):
            if len(items) != len(cues):
                raise RuntimeError("翻译字符串数组数量与字幕条数不一致")
            return [_normalize_text(item) for item in items]

        translated_map: Dict[int, str] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                index = int(item.get("index"))
            except Exception:
                continue
            translation = _normalize_text(str(item.get("translation") or item.get("text") or ""))
            if translation:
                translated_map[index] = translation

        translations: List[str] = []
        for cue in cues:
            translation = translated_map.get(cue.index)
            if not translation:
                raise RuntimeError(f"翻译结果缺少索引 {cue.index}")
            translations.append(translation)
        return translations

    @staticmethod
    def __parse_indexed_translation_line(line: str) -> Optional[Tuple[int, str]]:
        patterns = [
            r"^\s*<id>(\d+)</id>\s*(.+)$",
            r"^\s*<index>(\d+)</index>\s*(.+)$",
            r"^\s*(\d+)\s*[\t|｜]+\s*(.+)$",
            r"^\s*(\d+)\s*[\.、:：\)\]-]\s*(.+)$",
        ]
        for pattern in patterns:
            match = re.match(pattern, line)
            if not match:
                continue
            index = int(match.group(1))
            translation = EmbeddedBilingualSubtitle.__clean_translation_line(match.group(2))
            if translation:
                return index, translation
        return None

    @staticmethod
    def __clean_translation_line(text: str) -> str:
        text = (text or "").strip()
        text = re.sub(r"^\s*(翻译结果|译文|translation)\s*[:：]\s*", "", text, flags=re.I)
        text = re.sub(r"^\s*(\d+)\s*[\.、:：\)\]-]\s*", "", text)
        text = re.sub(r"^\s*(\d+)\s*[\t|｜]+\s*", "", text)
        text = re.sub(r"^\s*[-*]\s*", "", text)
        text = text.strip().strip('"').strip("'").strip()
        return _normalize_text(text)

    def __build_output_path(self, file_path: Path) -> Path:
        suffix = (self._output_suffix or "zh.default").strip().strip(".")
        return file_path.with_name(f"{file_path.stem}.{suffix}.srt")

    def __record_history(self, result: ProcessResult):
        history = self.get_data("history") or []
        history.insert(
            0,
            {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "file_path": result.file_path,
                "status": result.status,
                "mode": result.mode,
                "reason": result.reason,
                "output_path": result.output_path,
                "english_stream": result.english_stream,
                "chinese_stream": result.chinese_stream,
            },
        )
        self.save_data("history", history[:200])

    def __notify_single_result(self, result: ProcessResult, source: str):
        if not self._notify:
            return
        if result.status == "skipped":
            return
        title = "内嵌双语字幕合成成功" if result.status == "success" else "内嵌双语字幕合成失败"
        text = (
            f"来源：{source}\n"
            f"文件：{result.file_path}\n"
            f"模式：{result.mode}\n"
            f"输出：{result.output_path or '-'}\n"
            f"原因：{result.reason or '-'}"
        )
        self.post_message(
            mtype=NotificationType.Plugin,
            title=title,
            text=text,
        )

    def __notify_batch_summary(self, results: List[ProcessResult], source: str):
        if not self._notify or not results:
            return
        success = len([item for item in results if item.status == "success"])
        failed = len([item for item in results if item.status == "failed"])
        skipped = len([item for item in results if item.status == "skipped"])
        if success == failed == skipped == 0:
            return
        lines = [
            f"来源：{source}",
            f"成功：{success}",
            f"失败：{failed}",
            f"跳过：{skipped}",
        ]
        failed_items = [item for item in results if item.status == "failed"][:5]
        for item in failed_items:
            lines.append(f"失败：{item.file_path} -> {item.reason}")
        self.post_message(
            mtype=NotificationType.Plugin,
            title="内嵌双语字幕批量处理结果",
            text="\n".join(lines),
        )

    def __update_config(self):
        self.update_config(
            {
                "enabled": self._enabled,
                "notify": self._notify,
                "monitor_transfer": self._monitor_transfer,
                "onlyonce": self._onlyonce,
                "test_onlyonce": self._test_onlyonce,
                "overwrite": self._overwrite,
                "keep_temp": self._keep_temp,
                "cron": self._cron,
                "scan_paths": self._scan_paths,
                "exclude_paths": self._exclude_paths,
                "custom_files": self._custom_files,
                "output_suffix": self._output_suffix,
                "ffmpeg_path": self._ffmpeg_path,
                "ffprobe_path": self._ffprobe_path,
                "translate_url": self._translate_url,
                "translate_api_key": self._translate_api_key,
                "translate_model": self._translate_model,
                "translate_batch_size": self._translate_batch_size,
                "translate_timeout": self._translate_timeout,
                "test_text": self._test_text,
            }
        )
