# -*- coding: utf-8 -*-
"""电子鼻实时通信协议。

功能：定义带版本号的JSONL消息格式，校验16路电阻、温湿度等字段，
并生成主程序与推理程序之间使用的标准消息和错误消息。
"""

import json
import math
import time


# 协议版本和单条消息上限；双方必须使用相同的版本才能通信。
SCHEMA = "enose.runtime.v1"
PROTOCOL_VERSION = 1
SENSOR_COUNT = 16
MAX_LINE_BYTES = 64 * 1024

# 支持的消息类型。status、result和recovery分别承载监控、识别与恢复事件。
TYPE_SENSOR_FRAME = "sensor_frame"
TYPE_READY = "ready"
TYPE_RESULT = "result"
TYPE_STATUS = "status"
TYPE_RECOVERY = "recovery"
TYPE_ERROR = "error"
TYPE_PING = "ping"
TYPE_PONG = "pong"
TYPE_GET_STATUS = "get_status"
TYPE_DEVICE_STATUS = "device_status"
TYPE_RESET = "reset"
TYPE_SHUTDOWN = "shutdown"


class ProtocolError(ValueError):
    """协议校验异常，携带机器可读错误码和可选的原始帧序号。"""

    def __init__(self, code, message, source_sequence=None):
        """保存错误码、说明及发生错误的数据帧序号。"""
        super().__init__(message)
        self.code = str(code)
        self.source_sequence = source_sequence


def now_ms():
    """返回当前Unix毫秒时间戳。"""
    return int(round(time.time() * 1000.0))


def _finite_float(value, field_name):
    """把字段转成有限浮点数，拒绝文本错误、无穷大和NaN。"""
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError("INVALID_FIELD", "%s must be numeric" % field_name) from exc
    if not math.isfinite(value):
        raise ProtocolError("INVALID_FIELD", "%s must be finite" % field_name)
    return value


def _finite_int(value, field_name, minimum=None, maximum=None):
    """把字段转成整数，并检查可选的最小值和最大值范围。"""
    if isinstance(value, bool):
        raise ProtocolError("INVALID_FIELD", "%s must be an integer" % field_name)
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError("INVALID_FIELD", "%s must be an integer" % field_name) from exc
    if isinstance(value, float) and not value.is_integer():
        raise ProtocolError("INVALID_FIELD", "%s must be an integer" % field_name)
    if minimum is not None and integer < minimum:
        raise ProtocolError("INVALID_FIELD", "%s is below its valid range" % field_name)
    if maximum is not None and integer > maximum:
        raise ProtocolError("INVALID_FIELD", "%s is above its valid range" % field_name)
    return integer


def _text(value, field_name, max_length=128):
    """清理并校验必填文本，防止空值或超长内容进入协议。"""
    text = str(value or "").strip()
    if not text:
        raise ProtocolError("INVALID_FIELD", "%s is required" % field_name)
    if len(text) > max_length:
        raise ProtocolError("INVALID_FIELD", "%s is too long" % field_name)
    return text


def parse_json_line(line):
    """解析一行UTF-8 JSONL数据，并保证顶层内容是JSON对象。"""
    # 先按字节检查大小，避免异常设备发送超大消息占用过多内存。
    raw = line if isinstance(line, bytes) else str(line).encode("utf-8")
    if len(raw) > MAX_LINE_BYTES:
        raise ProtocolError("MESSAGE_TOO_LARGE", "one JSON line may not exceed 64 KiB")
    # 网络和串口统一使用UTF-8编码。
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtocolError("INVALID_UTF8", "message is not valid UTF-8") from exc
    # 每一行只允许包含一个完整JSON对象。
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProtocolError("INVALID_JSON", "invalid JSON: %s" % exc.msg) from exc
    if not isinstance(payload, dict):
        raise ProtocolError("INVALID_MESSAGE", "one JSON line must contain one object")
    return payload


def dumps(payload):
    """把消息压缩成一行JSON文本，保留中文并禁止输出NaN。"""
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def envelope(message_type, **fields):
    """给业务字段补上协议名称、版本、消息类型和发送时间。"""
    payload = {
        "schema": SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "type": str(message_type),
        "timestamp_ms": now_ms(),
    }
    payload.update(fields)
    return payload


def validate_envelope(payload, allowed_types=None):
    """校验消息外层协议，并可限制当前位置允许接收的消息类型。"""
    # schema和protocol_version用于阻止不兼容版本静默混用。
    if not isinstance(payload, dict):
        raise ProtocolError("INVALID_MESSAGE", "message must be an object")
    if payload.get("schema") != SCHEMA:
        raise ProtocolError("UNSUPPORTED_SCHEMA", "schema must be %s" % SCHEMA)
    version = _finite_int(payload.get("protocol_version"), "protocol_version")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(
            "UNSUPPORTED_VERSION",
            "protocol_version must be %d" % PROTOCOL_VERSION,
        )
    # 消息类型统一转成小写，便于后续稳定比较。
    message_type = str(payload.get("type") or "").strip().lower()
    if allowed_types is not None and message_type not in set(allowed_types):
        raise ProtocolError("UNSUPPORTED_TYPE", "unsupported message type: %s" % message_type)
    payload = dict(payload)
    payload["type"] = message_type
    return payload


def normalize_sensor_frame(payload, fill_timestamp=True):
    """校验电子鼻帧，并返回字段顺序和数据类型统一的标准对象。"""
    payload = validate_envelope(payload, {TYPE_SENSOR_FRAME})
    # 基础身份与时序字段用于数据追踪、排序和错误定位。
    device_id = _text(payload.get("device_id"), "device_id")
    sequence = _finite_int(payload.get("sequence"), "sequence", 0, 0xFFFFFFFF)
    timestamp_value = payload.get("timestamp_ms")
    if timestamp_value is None and fill_timestamp:
        timestamp_value = now_ms()
    timestamp_ms = _finite_int(timestamp_value, "timestamp_ms", 0)

    # 电阻必须严格为16路，每一路都必须是大于0的有限数值。
    resistance = payload.get("resistance")
    if not isinstance(resistance, list) or len(resistance) != SENSOR_COUNT:
        raise ProtocolError(
            "INVALID_FRAME",
            "resistance must contain exactly 16 values",
            sequence,
        )
    resistance_values = []
    for index, value in enumerate(resistance, start=1):
        number = _finite_float(value, "resistance[%d]" % index)
        if number <= 0.0:
            raise ProtocolError(
                "INVALID_FRAME",
                "resistance values must be greater than zero",
                sequence,
            )
        resistance_values.append(number)

    # 温湿度与电阻同帧发送；相对湿度的物理范围限制为0%至100%。
    temperature_c = _finite_float(payload.get("temperature_c"), "temperature_c")
    humidity_rh = _finite_float(payload.get("humidity_rh"), "humidity_rh")
    if humidity_rh < 0.0 or humidity_rh > 100.0:
        raise ProtocolError("INVALID_FRAME", "humidity_rh must be in 0..100", sequence)

    # 只输出协议规定字段，避免未约定字段被继续传入推理核心。
    normalized = {
        "schema": SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "type": TYPE_SENSOR_FRAME,
        "device_id": device_id,
        "sequence": sequence,
        "timestamp_ms": timestamp_ms,
        "resistance": resistance_values,
        "temperature_c": temperature_c,
        "humidity_rh": humidity_rh,
    }
    if "sensor_time_ms" in payload:
        normalized["sensor_time_ms"] = _finite_int(
            payload.get("sensor_time_ms"), "sensor_time_ms", 0
        )
    return normalized


def error_payload(code, message, source_sequence=None, fatal=False):
    """创建标准错误消息，供接收端按错误码和帧序号定位问题。"""
    return envelope(
        TYPE_ERROR,
        code=str(code),
        message=str(message),
        source_sequence=source_sequence,
        fatal=bool(fatal),
    )
