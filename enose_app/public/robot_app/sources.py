# -*- coding: utf-8 -*-
"""电子鼻数据源适配模块。

功能：把三种来源统一转换成经过协议校验的16路电阻、温度和湿度标准帧，
供机器人主程序使用：
- 串口（真机）：Modbus风格二进制帧协议，每秒请求一次、解析41字节响应帧；
- TCP / 标准输入：开发联调用的JSONL文本帧。
"""

import socket
import struct
import sys
import time

from common.protocol import (
    PROTOCOL_VERSION,
    SCHEMA,
    TYPE_SENSOR_FRAME,
    normalize_sensor_frame,
    now_ms,
    parse_json_line,
)


class EnoseBinaryFrameParser:
    """真机电子鼻二进制帧解析器（与采集工具 SerialThread 同口径换算下位机返回的帧数据）。
    解决串口数据源用错协议问题，保证16路电阻、温度和湿度的正确解析。

    上位机每秒发送读数命令 ``01 03 00 00 00 12 C5 C7``，下位机回一帧：

        ``01 03 24`` + ADC1..16(各2字节大端) + 温度原始值(2字节)
        + 湿度原始值(2字节) + CRC(2字节)，共41字节。

    换算公式：温度℃ = raw/100 - 40；湿度%RH = raw/100；
    电阻kΩ = 匹配电阻 * ADC / (4096 - ADC)。

    解析器内部维护粘包/断包缓冲：坏字节自动跳过，单帧异常不影响后续帧。
    """

    FRAME_LEN = 41
    HEADER = b"\x01\x03\x24"
    SEND_CMD = bytes.fromhex("01 03 00 00 00 12 C5 C7")
    SENSOR_COUNT = 16

    def __init__(self, matched_resistance_kohm=10.0):
        """保存ADC转电阻的匹配电阻值，并清空缓冲与统计。"""
        self.matched_resistance_kohm = float(matched_resistance_kohm)
        self._buffer = b""
        self.frame_count = 0
        self.skipped_bytes = 0

    def reset(self):
        """清空缓冲与统计，用于串口重连后重新同步。"""
        self._buffer = b""
        self.frame_count = 0
        self.skipped_bytes = 0

    def feed(self, chunk):
        """喂入一段串口字节流，返回本段解析出的帧字典列表。"""
        frames = []
        self._buffer += bytes(chunk)
        while True:
            index = self._buffer.find(self.HEADER)
            if index < 0:
                # 未发现帧头：只保留末尾2字节，避免帧头跨chunk时被误丢。
                keep = len(self.HEADER) - 1
                self.skipped_bytes += max(0, len(self._buffer) - keep)
                self._buffer = self._buffer[-keep:] if self._buffer else b""
                break
            if index > 0:
                # 帧头前的字节视为噪声，跳过并统计。
                self.skipped_bytes += index
                self._buffer = self._buffer[index:]
            if len(self._buffer) < self.FRAME_LEN:
                # 帧尚未收齐，保留缓冲等待下一段数据。
                break
            frame = self._buffer[: self.FRAME_LEN]
            self._buffer = self._buffer[self.FRAME_LEN:]
            parsed = self._parse_frame(frame)
            if parsed is not None:
                frames.append(parsed)
        return frames

    def _parse_frame(self, frame):
        """按固定字节位置解析一帧，返回16路电阻、温度和湿度。"""
        try:
            adc_values = [
                struct.unpack(">H", frame[3 + index * 2 : 5 + index * 2])[0]
                for index in range(self.SENSOR_COUNT)
            ]
            temp_raw = struct.unpack(">H", frame[35:37])[0]
            humid_raw = struct.unpack(">H", frame[37:39])[0]
            resistance = []
            for value in adc_values:
                if value >= 4096:
                    return None  # ADC越界说明帧错位，整帧丢弃。
                resistance.append(
                    self.matched_resistance_kohm * value / (4096.0 - value)
                )
            self.frame_count += 1
            return {
                "resistance": resistance,
                "temperature_c": round(temp_raw / 100.0 - 40.0, 2),
                "humidity_rh": round(humid_raw / 100.0, 2),
            }
        except (struct.error, TypeError, ValueError):
            return None


class StdinJsonlSource:
    """从标准输入读取JSONL传感器帧，主要用于管道输入和离线联调。"""

    def __init__(self, stream=None):
        """使用指定输入流；未指定时默认读取当前进程的标准输入。"""
        self.stream = stream or sys.stdin

    def __iter__(self):
        """逐行解析并产出标准传感器帧，自动跳过空行。"""
        source = self.stream.buffer if hasattr(self.stream, "buffer") else self.stream
        for line in source:
            if not line or not line.strip():
                continue
            yield normalize_sensor_frame(parse_json_line(line))

    def close(self):
        """标准输入由外部进程管理，因此这里不主动关闭。"""
        return None


class SerialJsonlSource:
    """从串口逐行读取UTF-8 JSONL传感器帧（备用协议，供JSONL固件联调）。"""

    def __init__(self, port, baudrate=115200, timeout_s=1.0):
        """打开8N1串口，并设置波特率和读取超时。"""
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("serial mode requires pyserial") from exc
        # 电子鼻约定使用8个数据位、无校验位、1个停止位。
        self.serial = serial.Serial(
            port=str(port),
            baudrate=int(baudrate),
            bytesize=8,
            parity="N",
            stopbits=1,
            timeout=float(timeout_s),
        )

    def __iter__(self):
        """持续读取串口行，校验成功后产出标准传感器帧。"""
        while self.serial.is_open:
            line = self.serial.readline()
            if not line:
                continue
            yield normalize_sensor_frame(parse_json_line(line))

    def close(self):
        """在串口仍打开时安全关闭设备连接。"""
        if self.serial.is_open:
            self.serial.close()


class SerialEnoseSource:
    """真机电子鼻串口数据源（二进制帧协议）。

    行为与采集工具 JL_sys_v3_6 的 SerialThread 保持一致：
    每约1秒发送一次读数命令，接收并解析41字节响应帧，换算16路电阻(kΩ)、
    温度(℃)和湿度(%RH)，再拼接成 enose.runtime.v1 的 sensor_frame 帧。
    """

    def __init__(
        self,
        port,
        baudrate=115200,
        timeout_s=1.0,
        device_id="回流1",
        matched_resistance_kohm=10.0,
        poll_interval_s=1.0,
        serial_obj=None,
    ):
        """打开8N1串口；测试时可注入serial_obj替换真实串口。"""
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("serial mode requires pyserial") from exc
        if serial_obj is not None:
            self.serial = serial_obj
        else:
            self.serial = serial.Serial(
                port=str(port),
                baudrate=int(baudrate),
                bytesize=8,
                parity="N",
                stopbits=1,
                timeout=float(timeout_s),
            )
        self.device_id = str(device_id)
        self.parser = EnoseBinaryFrameParser(matched_resistance_kohm)
        self.poll_interval_s = float(poll_interval_s)
        self.sequence = 0
        self._last_send_time = None

    def __iter__(self):
        """持续请求读数、解析响应，并产出标准传感器帧。"""
        while self.serial.is_open:
            # 周期性发送读数命令，触发下位机返回一帧数据。
            now = time.monotonic()
            if (
                self._last_send_time is None
                or now - self._last_send_time >= self.poll_interval_s
            ):
                self.serial.write(EnoseBinaryFrameParser.SEND_CMD)
                self._last_send_time = now

            # 只读取当前缓冲已有字节，避免阻塞；断包留在解析器内部等下次拼接。
            if self.serial.in_waiting:
                chunk = self.serial.read(self.serial.in_waiting)
            else:
                chunk = b""
            for parsed in self.parser.feed(chunk):
                yield {
                    "schema": SCHEMA,
                    "protocol_version": PROTOCOL_VERSION,
                    "type": TYPE_SENSOR_FRAME,
                    "device_id": self.device_id,
                    "sequence": self.sequence,
                    "timestamp_ms": now_ms(),
                    "resistance": parsed["resistance"],
                    "temperature_c": parsed["temperature_c"],
                    "humidity_rh": parsed["humidity_rh"],
                }
                self.sequence = (self.sequence + 1) & 0xFFFFFFFF
            # 短暂休眠避免空转占用CPU，与采集工具10ms节拍一致。
            time.sleep(0.01)

    def close(self):
        """在串口仍打开时安全关闭设备连接。"""
        if self.serial.is_open:
            self.serial.close()


class TcpJsonlSource:
    """从TCP JSONL流读取电子鼻帧，主要用于开发阶段模拟设备。"""

    def __init__(self, host, port, connect_timeout_s=5.0):
        """连接指定测试设备地址，并创建按行读取的二进制流。"""
        self.sock = socket.create_connection(
            (str(host), int(port)), timeout=float(connect_timeout_s)
        )
        self.stream = self.sock.makefile("rb")

    def __iter__(self):
        """逐行读取TCP数据，并产出校验后的标准传感器帧。"""
        for line in self.stream:
            if line.strip():
                yield normalize_sensor_frame(parse_json_line(line))

    def close(self):
        """先关闭读取流，再关闭底层TCP套接字。"""
        try:
            self.stream.close()
        finally:
            self.sock.close()


def build_source(args):
    """根据命令行配置创建对应的数据源对象。"""
    # 三种数据源最终都产出同一种标准帧，因此主循环无需区分来源。
    if args.sensor_transport == "stdin":
        return StdinJsonlSource()
    if args.sensor_transport == "serial":
        if not args.serial_port:
            raise ValueError("--serial-port is required for serial transport")
        protocol = getattr(args, "serial_protocol", "enose")
        if protocol == "enose":
            return SerialEnoseSource(
                args.serial_port,
                args.baudrate,
                device_id=getattr(args, "device_id", "回流1"),
                matched_resistance_kohm=getattr(
                    args, "matched_resistance_kohm", 10.0
                ),
            )
        return SerialJsonlSource(args.serial_port, args.baudrate)
    if args.sensor_transport == "tcp":
        return TcpJsonlSource(args.sensor_host, args.sensor_port)
    raise ValueError("unsupported sensor transport: %s" % args.sensor_transport)
