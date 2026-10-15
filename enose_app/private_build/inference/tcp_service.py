# -*- coding: utf-8 -*-
"""闭源推理程序使用的 TCP JSONL 服务模块。

本文件负责监听机器人端连接，逐行接收 JSON 消息，并把传感器帧交给推理引擎。
当引擎发现气味事件时，本模块立即把分类结果以一行 JSON 返回机器人端。
"""

import socket
import sys
import threading

from common.protocol import (
    TYPE_GET_STATUS,
    TYPE_PING,
    TYPE_RESET,
    TYPE_SENSOR_FRAME,
    TYPE_SHUTDOWN,
    dumps,
    envelope,
    error_payload,
    parse_json_line,
    validate_envelope,
)


class InferenceTcpServer:
    """单客户端 TCP 服务，负责通信调度，不直接执行特征和分类算法。"""

    def __init__(self, engine, host="127.0.0.1", port=5006, log_func=None):
        """保存推理引擎、监听地址和日志方法，并初始化服务状态。"""
        self.engine = engine
        self.host = str(host)
        self.port = int(port)
        self.log_func = log_func
        self.server_socket = None
        self.running = False
        self.bound_host = None
        self.bound_port = None
        self._started = threading.Event()

    def _log(self, text):
        """使用外部日志函数输出信息；未提供时打印到标准错误。"""
        if callable(self.log_func):
            self.log_func(str(text))
        else:
            print("[inference] %s" % text, file=sys.stderr, flush=True)

    @staticmethod
    def _send(stream, payload):
        """把一个字典序列化为单行 UTF-8 JSON，并立即发送。"""
        stream.write((dumps(payload) + "\n").encode("utf-8"))
        stream.flush()

    def serve_forever(self):
        """绑定端口并循环接受机器人连接，直到服务被停止。"""
        # SO_REUSEADDR 便于程序重启后快速重新绑定同一端口。
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.listen(1)
        sock.settimeout(0.5)
        self.server_socket = sock
        self.bound_host, self.bound_port = sock.getsockname()[:2]
        self.running = True
        self._started.set()
        self._log("监听 %s:%d" % (self.bound_host, self.bound_port))
        try:
            while self.running:
                # accept 设置短超时，使 stop() 后循环可以及时退出。
                try:
                    client, address = sock.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                self._log("机器人已连接 %s:%d" % (address[0], address[1]))
                try:
                    self._serve_client(client)
                finally:
                    try:
                        client.close()
                    except OSError:
                        pass
                    self._log("机器人连接已断开")
        finally:
            self.running = False
            try:
                sock.close()
            except OSError:
                pass

    def _serve_client(self, client):
        """处理一个机器人连接，并按消息类型执行推理、查询或控制操作。"""
        stream = client.makefile("rwb")
        # 新连接代表机器人主程序重新启动，清空旧设备状态，
        # 避免上一次会话的序号、基线和寻峰状态跨会话残留。
        self.engine.reset_all_devices()
        # 连接建立后先告知机器人模型、采样率和缓存长度等就绪信息。
        self._send(stream, self.engine.ready_payload())
        try:
            for line in stream:
                if not self.running:
                    break
                if not line.strip():
                    continue
                source_sequence = None
                try:
                    payload = validate_envelope(parse_json_line(line))
                    source_sequence = payload.get("sequence")
                    message_type = payload["type"]
                    # sensor_frame 是正常实时通路；其他类型用于状态检查和控制。
                    if message_type == TYPE_SENSOR_FRAME:
                        # 一帧可同时产生监控状态和识别/恢复事件，按列表顺序逐条返回。
                        messages = self.engine.process_frame(payload)
                        if isinstance(messages, dict):
                            messages = [messages]  # 兼容旧引擎的一条字典返回值。
                        for message in messages or []:
                            self._send(stream, message)
                    elif message_type == TYPE_PING:
                        self._send(stream, envelope("pong"))
                    elif message_type == TYPE_GET_STATUS:
                        self._send(
                            stream,
                            self.engine.device_status(payload.get("device_id", "")),
                        )
                    elif message_type == TYPE_RESET:
                        device_id = str(payload.get("device_id") or "").strip()
                        if not device_id:
                            raise ValueError("reset requires device_id")
                        self.engine.reset_device(device_id)
                        self._send(
                            stream,
                            envelope("ack", operation="reset", device_id=device_id),
                        )
                    elif message_type == TYPE_SHUTDOWN:
                        self._send(stream, envelope("ack", operation="shutdown"))
                        self.running = False
                        break
                    else:
                        raise ValueError("unsupported input message type: %s" % message_type)
                except Exception as exc:
                    # 单条坏消息只返回非致命错误，不主动断开整个实时连接。
                    self._send(
                        stream,
                        error_payload(
                            "INVALID_INPUT",
                            str(exc),
                            source_sequence=source_sequence,
                            fatal=False,
                        ),
                    )
        finally:
            try:
                stream.close()
            except OSError:
                pass

    def start_in_thread(self, timeout_s=5.0):
        """在守护线程中启动服务，主要供自动测试和本地联调使用。"""
        thread = threading.Thread(
            target=self.serve_forever,
            name="inference-tcp-server",
            daemon=True,
        )
        thread.start()
        if not self._started.wait(float(timeout_s)):
            raise TimeoutError("inference TCP server did not start")
        return thread

    def stop(self):
        """停止监听并关闭服务端套接字，可被重复安全调用。"""
        self.running = False
        sock = self.server_socket
        self.server_socket = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
