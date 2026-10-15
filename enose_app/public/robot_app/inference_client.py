# -*- coding: utf-8 -*-
"""机器人端连接推理程序的TCP客户端。

功能：建立并保持JSONL连接，发送电子鼻帧，在后台接收ready、status、result、
recovery、error和pong消息，再把各类信息交给主程序显示。
"""

import socket
import threading

from common.protocol import (
    TYPE_ERROR,
    TYPE_PING,
    TYPE_PONG,
    TYPE_READY,
    TYPE_RECOVERY,
    TYPE_RESULT,
    TYPE_STATUS,
    dumps,
    envelope,
    parse_json_line,
    validate_envelope,
)


class InferenceClient:
    """管理一条主程序到推理程序的长连接，并异步接收推理结果。"""

    def __init__(
        self,
        host,
        port,
        on_result=None,
        on_status=None,
        on_phase=None,
        on_recovery=None,
        on_error=None,
        connect_timeout_s=10.0,
    ):
        """保存连接参数，创建发送锁、后台线程和通信状态事件。"""
        self.host = str(host)
        self.port = int(port)
        self.on_result = on_result
        self.on_status = on_status
        self.on_phase = on_phase
        self.on_recovery = on_recovery
        self.on_error = on_error
        self.connect_timeout_s = float(connect_timeout_s)
        self.sock = None
        self.reader = None
        # 发送锁防止多个线程同时写同一个TCP连接，造成JSON行交叉。
        self._send_lock = threading.Lock()
        self._reader_thread = None
        # 事件用于跨线程等待ready、pong以及连接关闭通知。
        self._ready_event = threading.Event()
        self._pong_event = threading.Event()
        self._closed = threading.Event()
        self.ready_payload = None

    def _status(self, text):
        """如果上层提供了状态回调，就把连接状态通知给主程序。"""
        if callable(self.on_status):
            self.on_status(str(text))

    def connect(self):
        """连接推理服务、启动接收线程，并等待服务发回ready消息。"""
        self._closed.clear()
        self._ready_event.clear()
        self.ready_payload = None
        # 建立连接后取消读超时，由后台线程持续等待推理消息。
        self.sock = socket.create_connection(
            (self.host, self.port), timeout=self.connect_timeout_s
        )
        self.sock.settimeout(None)
        self.reader = self.sock.makefile("rb")
        # 后台读取可避免主采集循环因等待result而停止收取电子鼻帧。
        self._reader_thread = threading.Thread(
            target=self._read_loop,
            name="inference-result-reader",
            daemon=True,
        )
        self._reader_thread.start()
        # ready既是启动握手，也是推理配置检查的前提。
        if not self._ready_event.wait(self.connect_timeout_s):
            self.close()
            raise TimeoutError("inference service did not send ready in time")
        if self.ready_payload is None:
            self.close()
            raise ConnectionError("inference service disconnected before ready")
        return dict(self.ready_payload)

    def _read_loop(self):
        """后台逐行读取推理消息，并按消息类型更新状态或调用回调。"""
        try:
            for line in self.reader:
                if not line.strip():
                    continue
                # 所有入站消息先做JSON解析和协议版本校验。
                payload = validate_envelope(parse_json_line(line))
                message_type = payload.get("type")
                if message_type == TYPE_READY:
                    # ready唤醒connect，使主程序可以开始转发传感器数据。
                    self.ready_payload = payload
                    self._ready_event.set()
                    self._status("推理程序已就绪")
                elif message_type == TYPE_RESULT:
                    # 气味事件结果直接交给主程序注册的处理函数。
                    if callable(self.on_result):
                        self.on_result(payload)
                elif message_type == TYPE_STATUS:
                    # 阶段状态（识别中/恢复中）交给显示处理函数。
                    if callable(self.on_phase):
                        self.on_phase(payload)
                elif message_type == TYPE_RECOVERY:
                    # 恢复完成或超时是独立事件，不与普通监控状态混在一起。
                    if callable(self.on_recovery):
                        self.on_recovery(payload)
                elif message_type == TYPE_ERROR:
                    if callable(self.on_error):
                        self.on_error(payload)
                    else:
                        self._status(
                            "推理程序错误 %s: %s"
                            % (payload.get("code", "UNKNOWN"), payload.get("message", ""))
                        )
                elif message_type == TYPE_PONG:
                    # pong用于确认之前发送的全部TCP消息已按顺序处理。
                    self._pong_event.set()
                else:
                    self._status("收到推理消息: %s" % message_type)
        except Exception as exc:
            if not self._closed.is_set():
                self._status("推理连接中断: %s" % exc)
        finally:
            # 即使读取线程异常退出，也要唤醒正在等待ready的主线程。
            self._closed.set()
            self._ready_event.set()

    def send(self, payload):
        """把一个协议对象编码成单行UTF-8 JSON并完整发送。"""
        if self.sock is None or self._closed.is_set():
            raise ConnectionError("inference TCP connection is not available")
        data = (dumps(payload) + "\n").encode("utf-8")
        with self._send_lock:
            self.sock.sendall(data)

    def ping(self, timeout_s=2.0):
        """发送ping并等待pong，确认连接正常且此前消息已经处理。"""
        self._pong_event.clear()
        self.send(envelope(TYPE_PING))
        if not self._pong_event.wait(float(timeout_s)):
            raise TimeoutError("inference service did not answer ping")
        return True

    def close(self):
        """关闭套接字和读取流；重复调用也不会报错。"""
        self._closed.set()
        sock = self.sock
        self.sock = None
        # 先shutdown唤醒可能阻塞在读取操作中的后台线程，再关闭套接字。
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()
        if self.reader is not None:
            try:
                self.reader.close()
            except OSError:
                pass
