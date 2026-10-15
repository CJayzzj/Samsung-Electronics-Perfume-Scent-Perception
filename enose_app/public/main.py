# -*- coding: utf-8 -*-
"""机器人端公开主程序入口。

功能：从COM13接收回流1电子鼻原始数据，逐帧保存并转发给推理程序；终端把
周期监控、识别结果、恢复事件和错误告警分开显示。本文件不包含补偿、基线、
寻峰、特征工程或分类模型源码。
"""

import argparse
import json
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from common.protocol import TYPE_RECOVERY, TYPE_RESULT, TYPE_STATUS
from robot_app.inference_client import InferenceClient
from robot_app.sources import build_source
from robot_app.storage import AsyncFrameRecorder, InferenceFrameJoiner


def build_argument_parser():
    """创建命令行参数解析器，集中定义连接、保存和终端显示参数。"""
    parser = argparse.ArgumentParser(description="Samsung robot-side e-nose main program")
    parser.add_argument(
        "--sensor-transport",
        choices=("serial", "tcp", "stdin"),
        default="serial",
        help="electronic-nose input transport",
    )
    parser.add_argument("--serial-port", default="COM13")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument(
        "--serial-protocol",
        choices=("enose", "jsonl"),
        default="enose",
        help="串口数据协议：enose=真机二进制帧（默认），jsonl=JSONL文本",
    )
    parser.add_argument(
        "--device-id",
        default="回流1",
        help="设备编号；必须与推理端标定配置一致",
    )
    parser.add_argument(
        "--matched-resistance-kohm",
        type=float,
        default=10.0,
        help="ADC转电阻公式中的匹配电阻(kΩ)",
    )
    parser.add_argument("--sensor-host", default="127.0.0.1")
    parser.add_argument("--sensor-port", type=int, default=5005)
    parser.add_argument("--inference-host", default="127.0.0.1")
    parser.add_argument("--inference-port", type=int, default=5006)
    parser.add_argument(
        "--inference-executable",
        default="",
        help="optional inference executable to launch before connecting",
    )
    parser.add_argument("--sample-rate-hz", type=float, default=1.0)
    parser.add_argument("--queue-length", type=int, default=1000)
    parser.add_argument("--save-dir", default="data/raw")
    parser.add_argument("--save-format", choices=("csv", "xlsx"), default="csv")
    parser.add_argument("--rotate-rows", type=int, default=50000)
    parser.add_argument("--xlsx-flush-every", type=int, default=20)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument(
        "--monitor-interval-s",
        type=float,
        default=2.0,
        help="普通监控信息的终端输出间隔，默认2秒",
    )
    parser.add_argument(
        "--terminal-format",
        choices=("text", "jsonl"),
        default="text",
        help="终端输出格式，默认便于人工阅读的text",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="每帧输出监控信息，用于数值对齐调试",
    )
    return parser


class TerminalReporter:
    """按工业终端风格显示监控、结果、恢复与错误，并支持JSONL输出。"""

    def __init__(self, device_id, output_format="text", monitor_interval_s=2.0, verbose=False):
        """保存显示参数，并初始化事件计数、节流时间和线程打印锁。"""
        self.device_id = str(device_id)
        self.output_format = str(output_format)
        self.monitor_interval_s = float(monitor_interval_s)
        self.verbose = bool(verbose)
        self.recognition_count = 0
        self._last_monitor_time = None
        self._last_phase = None
        self._lock = threading.Lock()

    @staticmethod
    def _clock_text():
        """返回当前本地时间，统一用于所有终端记录。"""
        return datetime.now().strftime("%H:%M:%S")

    @staticmethod
    def _score_text(value):
        """Score只在终端保留两位小数，协议内部仍保留完整精度。"""
        return "--" if value is None else "%.2f" % float(value)

    @staticmethod
    def _seconds_text(value):
        """把秒数格式化为一位小数；尚未得到结果时显示横线。"""
        return "--" if value is None else "%.1fs" % float(value)

    def _emit(self, payload, text, error=False):
        """以text或JSONL输出一条完整记录，防止多线程内容交叉。"""
        with self._lock:
            if self.output_format == "jsonl":
                output = dict(payload)
                output.setdefault("terminal_timestamp", datetime.now().isoformat(timespec="milliseconds"))
                print(json.dumps(output, ensure_ascii=False, separators=(",", ":")), flush=True)
            else:
                print(text, file=sys.stderr if error else sys.stdout, flush=True)

    def system(self, message, level="INFO", code="MAIN", event_id="-"):
        """显示连接、采集和保存状态；ERROR/WARN包含明确错误码。"""
        payload = {
            "type": "log",
            "level": str(level),
            "code": str(code),
            "device_id": self.device_id,
            "event_id": str(event_id),
            "message": str(message),
        }
        text = "[%s][%s][%s][%s][%s] %s" % (
            self._clock_text(),
            level,
            self.device_id,
            event_id,
            code,
            message,
        )
        self._emit(payload, text, error=level in ("WARN", "ERROR"))

    def handle_status(self, payload):
        """阶段变化立即显示；普通current/peak Score默认每2秒显示一次。"""
        if payload.get("type") != TYPE_STATUS:
            return
        now = time.monotonic()
        phase = str(payload.get("phase") or "")
        phase_changed = phase != self._last_phase
        due = (
            self._last_monitor_time is None
            or now - self._last_monitor_time >= self.monitor_interval_s
        )
        if not (self.verbose or phase_changed or due):
            return
        self._last_monitor_time = now
        self._last_phase = phase
        current = self._score_text(payload.get("current_score", payload.get("score")))
        peak = self._score_text(payload.get("peak_score"))
        event_id = str(payload.get("event_id") or "-")
        state = str(payload.get("state") or phase)
        if phase == "recovering":
            tag = "RECOVERING"
            extra = " | 恢复计时=%s" % self._seconds_text(payload.get("recovery_elapsed_s"))
        else:
            tag = "MONITOR"
            extra = ""
        text = "[%s][%s][%s][%s] 状态=%s | 当前Score=%s | 本次峰值=%s%s" % (
            self._clock_text(), tag, self.device_id, event_id, state, current, peak, extra
        )
        self._emit(payload, text)

    def handle_result(self, payload):
        """每个气味事件只显示一次最终分类、置信度、峰值与响应时间。"""
        if payload.get("type") != TYPE_RESULT:
            return
        self.recognition_count += 1
        confidence = payload.get("confidence")
        confidence_text = "--" if confidence is None else "%.1f%%" % (100.0 * float(confidence))
        event_id = str(payload.get("event_id") or "-")
        decision_stage = str(payload.get("decision_stage") or "")
        binary_text = ""
        if decision_stage == "closed_binary":
            perfume_probability = payload.get("perfume_probability")
            binary_threshold = payload.get("binary_threshold")
            binary_text = " | 香水概率=%s | 判定阈值=%s" % (
                "--"
                if perfume_probability is None
                else "%.1f%%" % (100.0 * float(perfume_probability)),
                "--"
                if binary_threshold is None
                else "%.1f%%" % (100.0 * float(binary_threshold)),
            )
        text = (
            "[%s][RESULT][%s][%s] 标签=%s(%s) | 置信度=%s | 峰值Score=%s "
            "| 响应时间=%s | 推理耗时=%.2fms%s | 提示：%s"
            % (
                self._clock_text(),
                self.device_id,
                event_id,
                str(payload.get("label") or ""),
                str(payload.get("class_code") or "--"),
                confidence_text,
                self._score_text(payload.get("peak_score")),
                self._seconds_text(payload.get("response_time_s")),
                float(payload.get("inference_time_ms") or 0.0),
                binary_text,
                str(payload.get("hint") or "可取出香水试纸"),
            )
        )
        self._emit(payload, text)

    def handle_recovery(self, payload):
        """分别显示恢复完成和300秒超时告警；超时不代表停止监测。"""
        if payload.get("type") != TYPE_RECOVERY:
            return
        event_id = str(payload.get("event_id") or "-")
        status = str(payload.get("status") or "")
        if status == "complete":
            text = (
                "[%s][RECOVERED][%s][%s] 当前Score=%s | 恢复时间=%s "
                "| 主导通道=%s | 提示：可开始下一次检测"
                % (
                    self._clock_text(),
                    self.device_id,
                    event_id,
                    self._score_text(payload.get("current_score")),
                    self._seconds_text(payload.get("recovery_time_s")),
                    payload.get("dominant_sensor", "--"),
                )
            )
            self._emit(payload, text)
            return
        text = (
            "[%s][WARN][%s][%s][RECOVERY_TIMEOUT] 恢复已超过%s，继续监测且不开始下一事件"
            % (
                self._clock_text(),
                self.device_id,
                event_id,
                self._seconds_text(payload.get("recovery_time_s")),
            )
        )
        self._emit(payload, text, error=True)

    def handle_error(self, payload):
        """显示推理端返回的结构化错误，保留设备、事件和错误码。"""
        code = str(payload.get("code") or "UNKNOWN")
        event_id = str(payload.get("event_id") or "-")
        message = str(payload.get("message") or "")
        text = "[%s][ERROR][%s][%s][%s] %s" % (
            self._clock_text(), self.device_id, event_id, code, message
        )
        self._emit(payload, text, error=True)


def _launch_inference(executable, host, port):
    """按需启动本机推理程序；未指定可执行文件时由用户另开终端启动。"""
    if not executable:
        return None
    path = Path(executable).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("inference executable not found: %s" % path)
    return subprocess.Popen(
        [str(path), "--host", str(host), "--port", str(port)],
        stdout=None,
        stderr=None,
    )


def _connect_with_retry(client, child_process, timeout_s):
    """在限定时间内反复连接推理服务，并监测子进程是否异常退出。"""
    deadline = time.monotonic() + float(timeout_s)
    last_error = None
    while time.monotonic() < deadline:
        if child_process is not None and child_process.poll() is not None:
            raise RuntimeError(
                "inference executable exited with code %s" % child_process.returncode
            )
        try:
            return client.connect()
        except (ConnectionError, OSError, TimeoutError) as exc:
            last_error = exc
            time.sleep(0.2)
    raise TimeoutError("could not connect to inference service: %s" % last_error)


def run(args):
    """执行采集：连接推理端、收帧、缓存、保存、转发并安全退出。"""
    if float(args.sample_rate_hz) <= 0.0:
        raise ValueError("--sample-rate-hz must be greater than zero")
    if int(args.queue_length) <= 0:
        raise ValueError("--queue-length must be greater than zero")
    if float(args.monitor_interval_s) <= 0.0:
        raise ValueError("--monitor-interval-s must be greater than zero")

    reporter = TerminalReporter(
        args.device_id,
        args.terminal_format,
        args.monitor_interval_s,
        args.verbose,
    )
    child_process = _launch_inference(
        args.inference_executable, args.inference_host, args.inference_port
    )
    joiner = None

    # 推理消息晚于原始帧异步返回。先把消息交给保存关联器，再保持原有终端显示。
    def handle_result(payload):
        if joiner is not None:
            joiner.handle_result(payload)
        reporter.handle_result(payload)

    def handle_phase(payload):
        if joiner is not None:
            joiner.handle_status(payload)
        reporter.handle_status(payload)

    def handle_recovery(payload):
        if joiner is not None:
            joiner.handle_recovery(payload)
        reporter.handle_recovery(payload)

    def handle_error(payload):
        if joiner is not None:
            joiner.handle_error(payload)
        reporter.handle_error(payload)

    client = InferenceClient(
        args.inference_host,
        args.inference_port,
        on_result=handle_result,
        on_status=lambda text: reporter.system(text, "INFO", "INFERENCE_CONNECTION"),
        on_phase=handle_phase,
        on_recovery=handle_recovery,
        on_error=handle_error,
        connect_timeout_s=args.connect_timeout,
    )
    source = None
    recorder = None
    frame_window = deque(maxlen=int(args.queue_length))
    try:
        ready = _connect_with_retry(client, child_process, args.connect_timeout)
        inference_queue_length = int(ready.get("queue_length", 0))
        if inference_queue_length != int(args.queue_length):
            raise ValueError(
                "主程序queue_length=%d与推理程序queue_length=%d不一致"
                % (int(args.queue_length), inference_queue_length)
            )
        inference_sample_rate_hz = float(ready.get("sample_rate_hz", 0.0))
        if abs(inference_sample_rate_hz - float(args.sample_rate_hz)) > 1e-9:
            raise ValueError(
                "主程序sample_rate_hz=%g与推理程序sample_rate_hz=%g不一致"
                % (float(args.sample_rate_hz), inference_sample_rate_hz)
            )
        required_device_id = str(ready.get("required_device_id") or "").strip()
        if required_device_id and required_device_id != str(args.device_id):
            raise ValueError(
                "主程序device_id=%s与推理程序required_device_id=%s不一致"
                % (args.device_id, required_device_id)
            )
        reporter.system(
            "已连接推理程序 %s:%d，model_id=%s"
            % (
                args.inference_host,
                args.inference_port,
                (ready.get("model") or {}).get("model_id", "unknown"),
            ),
            "INFO",
            "INFERENCE_READY",
        )

        recorder = AsyncFrameRecorder(
            args.save_dir,
            output_format=args.save_format,
            rotate_rows=args.rotate_rows,
            xlsx_flush_every=args.xlsx_flush_every,
        )
        joiner = InferenceFrameJoiner(
            recorder,
            max_pending=args.queue_length,
            on_warning=lambda text: reporter.system(
                text, "WARN", "INFERENCE_STATUS_BACKLOG"
            ),
        )
        source = build_source(args)
        reporter.system(
            "开始接收电子鼻数据，端口=%s，采样频率=%gHz，内存窗口=%d帧"
            % (args.serial_port, float(args.sample_rate_hz), args.queue_length),
            "INFO",
            "SENSOR_STARTED",
        )
        for frame in source:
            frame_window.append(frame)
            # 必须先暂存再发送，防止推理结果过快返回而找不到对应原始帧。
            joiner.add_frame(frame)
            client.send(frame)
        client.ping(timeout_s=2.0)
    except KeyboardInterrupt:
        reporter.system("收到退出指令", "INFO", "STOP_REQUESTED")
        try:
            # pong位于此前全部推理消息之后，尽量让最后几帧完成状态匹配。
            client.ping(timeout_s=2.0)
        except (ConnectionError, OSError, TimeoutError) as exc:
            reporter.system(
                "退出前未能确认推理消息已处理完：%s" % exc,
                "WARN",
                "INFERENCE_DRAIN_TIMEOUT",
            )
    finally:
        if source is not None:
            source.close()
        client.close()
        if joiner is not None:
            unresolved = joiner.flush_unresolved()
            if unresolved:
                reporter.system(
                    "退出时有%d帧未收到推理状态，已保留原始数据并标记" % unresolved,
                    "WARN",
                    "INFERENCE_STATUS_MISSING",
                )
        if recorder is not None:
            recorder.close()
            reporter.system(
                "已保存%d帧原始数据" % recorder.rows_written,
                "INFO",
                "DATA_SAVED",
            )
        reporter.system(
            "本次运行共识别%d次" % reporter.recognition_count,
            "INFO",
            "SESSION_SUMMARY",
        )
        if child_process is not None and child_process.poll() is None:
            child_process.terminate()
            try:
                child_process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                child_process.kill()
    return 0


def cli_main(argv=None):
    """命令行入口：解析参数，并把异常转换成明确退出码。"""
    args = build_argument_parser().parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        reporter = TerminalReporter(
            getattr(args, "device_id", "-"),
            getattr(args, "terminal_format", "text"),
        )
        reporter.system("运行失败: %s" % exc, "ERROR", "MAIN_FAILED")
        return 2


if __name__ == "__main__":
    raise SystemExit(cli_main())
