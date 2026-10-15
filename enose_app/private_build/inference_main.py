# -*- coding: utf-8 -*-
"""实时推理程序的命令行入口。

本文件负责定位打包到程序内的模型与配置文件，解析启动参数，创建推理引擎，
并启动TCP JSONL服务。当前在Windows直接以Python源码运行；未来Linux交付时，
仍可把本入口及核心依赖编译进闭源程序。
"""

import argparse
import os
import sys


SERVICE_VERSION = "2.2.0-dev"


def resource_path(filename):
    """查找模型、类别映射或运行配置等随程序打包的资源文件。

    查找顺序兼容 PyInstaller 临时目录、源码运行目录和可执行文件所在目录，
    因此同一套代码既能直接用 Python 调试，也能在打包后运行。
    """
    candidates = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(os.path.join(meipass, "resources", filename))
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", filename))
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(sys.executable)), "resources", filename))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return candidates[0]


def build_argument_parser():
    """定义推理服务支持的命令行参数及默认值。"""
    parser = argparse.ArgumentParser(description="Samsung e-nose inference TCP service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5006)
    parser.add_argument("--model", default=resource_path("model.joblib"))
    parser.add_argument("--class-map", default=resource_path("class_map.json"))
    parser.add_argument("--runtime-config", default=resource_path("runtime_config.json"))
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--version", action="store_true")
    return parser


def cli_main(argv=None):
    """加载资源并运行推理服务；返回值可直接作为程序退出码。"""
    args = build_argument_parser().parse_args(argv)
    if args.version:
        print(SERVICE_VERSION)
        return 0

    from common.protocol import dumps, error_payload
    from inference.engine import InferenceEngine, RuntimeConfig
    from inference.model_adapter import ClassCatalog, ModelAdapter
    from inference.tcp_service import InferenceTcpServer

    # 启动前一次性加载并校验“类别映射、模型、运行配置”。任一文件不匹配，
    # 都立即停止，避免带着错误模型进入实时采集。
    try:
        catalog = ClassCatalog.load(args.class_map)
        adapter = ModelAdapter(args.model, catalog)
        config = RuntimeConfig.load(args.runtime_config)
        engine = InferenceEngine(adapter, config)
    except Exception as exc:
        print(dumps(error_payload("STARTUP_FAILED", str(exc), fatal=True)))
        print("[inference] 启动失败: %s" % exc, file=sys.stderr, flush=True)
        return 2

    if args.self_test:
        # 自检只验证资源能否正常加载，不占用 TCP 端口。
        print(dumps(engine.ready_payload()))
        return 0

    # 资源确认无误后持续监听机器人端连接，直到收到中断或关闭指令。
    server = InferenceTcpServer(engine, args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
