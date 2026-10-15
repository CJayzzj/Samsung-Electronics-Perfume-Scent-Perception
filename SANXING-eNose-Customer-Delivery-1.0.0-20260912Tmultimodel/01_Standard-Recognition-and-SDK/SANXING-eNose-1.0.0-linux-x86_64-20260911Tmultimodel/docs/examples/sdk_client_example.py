#!/usr/bin/env python3
"""Connect to the inference service and send one example sensor frame."""

import argparse
import json
import threading
import time

from enose_sdk import InferenceClient


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="电子鼻 SDK 最小 TCP 推理示例"
    )
    parser.add_argument("--host", default="127.0.0.1", help="推理服务地址")
    parser.add_argument("--port", type=int, default=5006, help="推理服务端口")
    parser.add_argument("--adc", type=int, default=2048, help="16 路示例 ADC 值")
    parser.add_argument("--temperature-c", type=float, default=25.0)
    parser.add_argument("--humidity-rh", type=float, default=50.0)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    return parser


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    completed = threading.Event()
    callback_error = []

    def print_payload(prefix, payload):
        print(
            "%s %s" % (
                prefix,
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            ),
            flush=True,
        )

    def on_frame_response(payload):
        print_payload("frame_response:", payload)
        completed.set()

    def on_error(payload):
        print_payload("error:", payload)
        callback_error.append(payload)
        completed.set()

    client = InferenceClient(
        args.host,
        args.port,
        on_frame_response=on_frame_response,
        on_status=lambda message: print("status:", message, flush=True),
        on_error=on_error,
        connect_timeout_s=args.timeout_s,
    )
    try:
        ready = client.connect()
        print_payload("ready:", ready)
        adc_full_scale = float(ready["input"]["adc_full_scale"])
        if not 1 <= args.adc < adc_full_scale:
            raise ValueError(
                "--adc must satisfy 1 <= value < %g" % adc_full_scale
            )
        client.send(
            {
                "type": "frame",
                "sequence": 0,
                "timestamp_ms": int(time.time() * 1000),
                "adc": [args.adc] * 16,
                "temperature_c": args.temperature_c,
                "humidity_rh": args.humidity_rh,
            }
        )
        if not completed.wait(args.timeout_s):
            raise TimeoutError("未在限定时间内收到逐帧响应")
        if callback_error:
            return 2
        client.ping(timeout_s=args.timeout_s)
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
