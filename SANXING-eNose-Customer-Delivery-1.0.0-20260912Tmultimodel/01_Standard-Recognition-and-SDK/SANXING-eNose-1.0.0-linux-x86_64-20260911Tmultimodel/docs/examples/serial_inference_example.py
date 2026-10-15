#!/usr/bin/env python3
"""Read a real E-Nose serial device and forward frames for inference."""

import argparse
import json

from enose_sdk import InferenceClient
from enose_sdk.input_sources import SerialEnoseSource
from enose_sdk.serial_ports import available_serial_ports


def default_serial_port():
    ports = available_serial_ports()
    for port in ports:
        if port.startswith("/dev/ttyCH341USB"):
            return port
    return ports[0] if ports else "/dev/ttyCH341USB0"


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="电子鼻串口采集与推理 SDK 示例"
    )
    parser.add_argument("--serial-port", default=default_serial_port())
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--inference-host", default="127.0.0.1")
    parser.add_argument("--inference-port", type=int, default=5006)
    parser.add_argument("--connect-timeout-s", type=float, default=10.0)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="发送帧数上限；0 表示持续运行到 Ctrl+C",
    )
    return parser


def print_message(prefix, payload):
    print(
        "%s %s" % (
            prefix,
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        ),
        flush=True,
    )


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    client = InferenceClient(
        args.inference_host,
        args.inference_port,
        on_frame_response=lambda payload: print_message("frame_response:", payload),
        on_status=lambda message: print("status:", message, flush=True),
        on_error=lambda payload: print_message("error:", payload),
        connect_timeout_s=args.connect_timeout_s,
    )
    source = None
    sent = 0
    try:
        ready = client.connect()
        print_message("ready:", ready)
        sample_rate_hz = float(ready["sample_rate_hz"])
        if sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be greater than zero")
        source = SerialEnoseSource(
            port=args.serial_port,
            baudrate=args.baudrate,
            timeout_s=1.0,
            adc_full_scale=ready["input"]["adc_full_scale"],
            poll_interval_s=1.0 / sample_rate_hz,
        )
        for frame in source:
            client.send(frame)
            sent += 1
            if args.max_frames > 0 and sent >= args.max_frames:
                break
        client.ping(timeout_s=2.0)
    except KeyboardInterrupt:
        print("收到 Ctrl+C，正在停止。", flush=True)
    finally:
        if source is not None:
            source.close()
        client.close()
    print("已发送 %d 帧。" % sent, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
