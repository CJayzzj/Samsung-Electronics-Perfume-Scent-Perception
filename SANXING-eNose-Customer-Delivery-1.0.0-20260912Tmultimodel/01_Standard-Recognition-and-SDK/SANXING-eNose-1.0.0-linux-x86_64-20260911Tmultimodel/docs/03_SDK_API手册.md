# 设备接入 SDK API 手册

SDK 版本：1.0.0。支持 Python 3.8 或更高版本。

## 安装

在设备接入 SDK 解压目录执行：

```bash
python3 -m pip install python/enose_sdk-1.0.0-py3-none-any.whl
```

XLSX 支持：

```bash
python3 -m pip install "python/enose_sdk-1.0.0-py3-none-any.whl[xlsx]"
```

## 顶层公开接口

```python
from enose_sdk import (
    InferenceClient,
    PROTOCOL_VERSION,
    MODBUS_SENSOR_FRAME_HEADER,
    MODBUS_SENSOR_FRAME_LENGTH,
    extract_modbus_frames,
    modbus_crc16,
    valid_modbus_frame,
)
```

## InferenceClient

### 构造函数

```python
InferenceClient(
    host,
    port,
    on_frame_response=None,
    on_status=None,
    on_error=None,
    connect_timeout_s=10.0,
)
```

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `host` | `str` | 是 | 推理服务地址 |
| `port` | `int` | 是 | 推理服务端口 |
| `on_frame_response` | `callable(dict)` | 否 | 收到逐帧响应时调用 |
| `on_status` | `callable(str)` | 否 | 连接状态变化时调用 |
| `on_error` | `callable(dict)` | 否 | 收到协议错误时调用 |
| `connect_timeout_s` | `float` | 否 | 连接和等待 `ready` 的超时秒数，默认 10 秒 |

回调由 SDK 的接收线程调用。回调中应避免长时间阻塞；如需更新 GUI，应通过 GUI
框架的线程安全信号转发。

### connect()

```python
ready = client.connect()
```

连接服务并等待 `ready`。返回一个新字典：

| 字段 | 类型 | 说明 |
|---|---|---|
| `type` | `str` | 固定为 `ready` |
| `protocol_version` | `int` | 协议版本，当前为 1 |
| `device_id` | `str` | 当前模型对应的设备编号 |
| `sample_rate_hz` | `float` | 要求的采样频率 |
| `model_id` | `str` | 当前模型标识 |
| `algorithm_version` | `str` | 服务实现版本标识；按不透明字符串处理 |
| `classes` | `dict[str, str]` | 类别代码到显示名称的映射 |
| `input` | `dict` | 传感器数量、ADC 量程和匹配电阻参数 |

可能抛出 `OSError`、`ConnectionError` 或 `TimeoutError`。

### send(payload)

```python
client.send(payload)
```

发送一帧传感器数据，无返回值。输入字典必须包含：

| 字段 | 类型 | 必填 | 约束 |
|---|---|---|---|
| `type` | `str` | 是 | 固定为 `frame` |
| `sequence` | `int` | 是 | `0..4294967295`，连接内连续递增 |
| `timestamp_ms` | `int` | 是 | 非负 Unix 毫秒时间戳，不得倒退 |
| `adc` | `list[int]` | 是 | 正好 16 项，且 `1 <= value < adc_full_scale` |
| `temperature_c` | `int/float` | 是 | 有限数值，单位 ℃ |
| `humidity_rh` | `int/float` | 是 | `0..100`，单位 %RH |

字段校验失败时抛出 `enose_sdk.protocol.ProtocolError`；连接不可用时抛出
`ConnectionError` 或 `OSError`。

### on_frame_response(payload)

每个合法输入帧对应一次回调：

| 字段 | 类型 | 说明 |
|---|---|---|
| `type` | `str` | 固定为 `frame_response` |
| `sequence` | `int` | 对应的输入帧序号 |
| `gate_score` | `float` | 当前帧的服务状态评分；仅用于监控 |
| `event_peak_score` | `float/null` | 事件确认帧的服务状态评分，其他帧为 `null` |
| `result` | `dict/null` | 本帧产生最终识别结果时为字典 |

`result` 非空时包含 `event_id`、`class_code`、`label`，并可能包含
`response_time_s`。一次物理响应周期最多返回一次非空结果。客户端应以 `result`
作为识别业务依据，不应根据两个评分字段自行判断识别结果。

### on_error(payload)

```python
{
    "type": "error",
    "code": "INVALID_FRAME",
    "message": "request failed",
    "sequence": 123
}
```

`sequence` 仅在错误可关联到合法帧序号时存在。错误码见 `04_协议说明.md`。

### ping(timeout_s=2.0)

发送心跳并等待 `pong`。成功返回 `True`，超时抛出 `TimeoutError`，连接不可用时
抛出 `ConnectionError`。

### close()

关闭读取流和 TCP 连接，无返回值。可以重复调用。建议始终在 `finally` 中关闭。

## 完整调用示例

```python
import threading
import time

from enose_sdk import InferenceClient

received = threading.Event()


def on_response(payload):
    print(payload)
    received.set()


client = InferenceClient("127.0.0.1", 5006, on_frame_response=on_response)
try:
    ready = client.connect()
    client.send({
        "type": "frame",
        "sequence": 0,
        "timestamp_ms": int(time.time() * 1000),
        "adc": [2048] * 16,
        "temperature_c": 25.0,
        "humidity_rh": 50.0,
    })
    if not received.wait(5.0):
        raise TimeoutError("未收到逐帧响应")
finally:
    client.close()
```

## 串口发现

```python
from enose_sdk.serial_ports import available_serial_ports

ports = available_serial_ports()
```

返回 `list[str]`。Linux 下除 pyserial 枚举结果外，还会扫描
`/dev/ttyCH341USB*`。

## 电子鼻二进制数据源

```python
from enose_sdk.input_sources import SerialEnoseSource

source = SerialEnoseSource(
    port="/dev/ttyCH341USB0",
    baudrate=115200,
    timeout_s=1.0,
    adc_full_scale=ready["input"]["adc_full_scale"],
    poll_interval_s=1.0 / ready["sample_rate_hz"],
)
```

迭代 `source` 会持续产生符合 `send()` 要求的帧字典；结束时调用
`source.close()`。完整用法见 `examples/serial_inference_example.py`。

## Modbus 工具

### modbus_crc16(data)

输入 bytes-like 数据，返回 `0..65535` 的 Modbus RTU CRC-16 整数。

### valid_modbus_frame(frame, expected_header=None, expected_length=None)

校验帧 CRC，并可同时约束帧头和长度，返回 `bool`。

### extract_modbus_frames(buffer, expected_header=..., expected_length=41)

从 bytes-like 缓冲区中提取完整有效帧，返回：

```python
(frames, remaining_buffer, crc_error_count)
```

## 参考客户端命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--sensor-transport` | `serial` | 数据来源：`serial`、`tcp` 或 `stdin` |
| `--serial-port` | 自动发现；未发现回退 `/dev/ttyUSB0` | 电子鼻串口 |
| `--baudrate` | `115200` | 串口波特率 |
| `--serial-protocol` | `enose` | `enose` 二进制协议或 `jsonl` 文本协议 |
| `--sensor-host` | `127.0.0.1` | TCP 模拟数据源地址 |
| `--sensor-port` | `5005` | TCP 模拟数据源端口 |
| `--inference-host` | `127.0.0.1` | 推理服务地址 |
| `--inference-port` | `5006` | 推理服务端口 |
| `--inference-executable` | 空 | 连接前启动的本机推理程序路径 |
| `--save-dir` | 用户数据目录 | CSV/XLSX 保存目录 |
| `--save-format` | `csv` | `csv` 或 `xlsx` |
| `--rotate-rows` | `50000` | 每个数据文件的最大数据行数 |
| `--xlsx-flush-every` | `20` | XLSX 累计多少行刷新一次 |
| `--connect-timeout` | `10.0` | 推理服务连接超时秒数 |
| `--monitor-interval-s` | `1.0` | 普通监控输出间隔秒数 |
| `--terminal-format` | `text` | 终端输出为 `text` 或 `jsonl` |
| `--verbose` | 关闭 | 每帧显示监控信息 |

查看当前版本的最终参数定义：

```bash
./application/enose-client --help
```

## 数据文件字段

参考客户端保存以下列：`timestamp_ms`、`device_id`、`sequence`、
`ADC_Sensor1` 至 `ADC_Sensor16`、`Temperature`、`Humidity`、
`Recognition_Result`。没有最终结果的帧，其 `Recognition_Result` 为空。
