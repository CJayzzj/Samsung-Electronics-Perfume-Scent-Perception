# -*- coding: utf-8 -*-
"""电子鼻数据模拟器。

本文件仅用于开发联调：按照正式JSONL协议生成16路电阻、温度和湿度，
可输出到终端，也可作为TCP服务端持续向公开主程序发送模拟帧。
它不参与正式采集、特征计算或模型分类。
"""

import argparse
import math
import random
import socket
import sys
import time
from pathlib import Path


# 模拟器复用公开端的协议实现，避免自行拼接出不符合规范的数据。
PUBLIC_ROOT = Path(__file__).resolve().parents[1] / "public"
if str(PUBLIC_ROOT) not in sys.path:
    sys.path.insert(0, str(PUBLIC_ROOT))

from common.protocol import (  # noqa: E402
    PROTOCOL_VERSION,
    SCHEMA,
    TYPE_SENSOR_FRAME,
    dumps,
    now_ms,
)


# 一次完整模拟气味脉冲：先上升到峰值，再逐步回落到空气基线。
PULSE_SHAPE = [
    0.00,
    0.01,
    0.03,
    0.06,
    0.10,
    0.16,
    0.24,
    0.34,
    0.45,
    0.58,
    0.70,
    0.80,
    0.72,
    0.62,
    0.50,
    0.38,
    0.26,
    0.16,
    0.08,
    0.03,
    0.00,
]


def build_parser():
    """定义模拟器命令行参数，并返回参数解析器。"""

    parser = argparse.ArgumentParser(description="Synthetic electronic-nose frame source")
    parser.add_argument("--mode", choices=("stdout", "tcp-server"), default="stdout")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--device-id", default="EN-DEV-001")
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--frames", type=int, default=0, help="0 means run until Ctrl+C")
    parser.add_argument("--pulse-every", type=int, default=100)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def frame_stream(args):
    """按指定频率连续产生符合协议的传感器帧。"""

    # 固定随机种子后，每次联调都能复现相同的微小噪声。
    rng = random.Random(args.seed)
    interval = 1.0 / float(args.sample_rate)
    sequence = 0
    next_deadline = time.monotonic()
    while args.frames <= 0 or sequence < args.frames:
        # 预热阶段只输出空气基线；之后按pulse_every周期插入一次气味脉冲。
        if sequence < int(args.warmup_frames):
            response = 0.0
        else:
            pulse_index = (sequence - int(args.warmup_frames)) % int(args.pulse_every)
            response = PULSE_SHAPE[pulse_index] if pulse_index < len(PULSE_SHAPE) else 0.0
        # 前8路模拟电阻下降，后8路模拟电阻上升，并加入通道差异和微小噪声。
        resistance = []
        for sensor_index in range(16):
            direction = -1.0 if sensor_index < 8 else 1.0
            scale = 0.75 + 0.03 * sensor_index
            noise = rng.uniform(-0.002, 0.002)
            resistance.append(10.0 * (1.0 + direction * response * scale + noise))
        # 字段名称和顺序要求由common.protocol统一定义。
        yield {
            "schema": SCHEMA,
            "protocol_version": PROTOCOL_VERSION,
            "type": TYPE_SENSOR_FRAME,
            "device_id": args.device_id,
            "sequence": sequence & 0xFFFFFFFF,
            "timestamp_ms": now_ms(),
            "resistance": resistance,
            "temperature_c": 25.0 + 0.2 * math.sin(sequence / 100.0),
            "humidity_rh": 50.0 + 0.5 * math.sin(sequence / 80.0),
        }
        # 使用单调时钟控制发送节拍，避免处理耗时造成间隔不断累积偏移。
        sequence += 1
        next_deadline += interval
        remaining = next_deadline - time.monotonic()
        if remaining > 0.0:
            time.sleep(remaining)


def run_stdout(args):
    """把每帧JSONL写到标准输出，便于管道或人工查看。"""

    for frame in frame_stream(args):
        print(dumps(frame), flush=True)


def run_tcp_server(args):
    """监听开发端口，主程序连接后持续发送模拟JSONL帧。"""

    # SO_REUSEADDR允许程序刚退出后较快地重新绑定同一开发端口。
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    print(
        "[sensor-simulator] listening on %s:%d" % (args.host, args.port),
        file=sys.stderr,
        flush=True,
    )
    # 当前模拟器一次只服务一个主程序连接，足以完成单机联调。
    client, address = server.accept()
    print(
        "[sensor-simulator] robot connected from %s:%d" % address,
        file=sys.stderr,
        flush=True,
    )
    try:
        for frame in frame_stream(args):
            client.sendall((dumps(frame) + "\n").encode("utf-8"))
    finally:
        client.close()
        server.close()


def main(argv=None):
    """校验参数并选择终端输出或TCP服务模式。"""

    args = build_parser().parse_args(argv)
    if args.sample_rate <= 0.0 or args.pulse_every <= len(PULSE_SHAPE):
        raise ValueError("sample-rate must be positive and pulse-every must exceed pulse length")
    # Ctrl+C属于正常结束，因此捕获后直接返回成功状态。
    try:
        if args.mode == "stdout":
            run_stdout(args)
        else:
            run_tcp_server(args)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
