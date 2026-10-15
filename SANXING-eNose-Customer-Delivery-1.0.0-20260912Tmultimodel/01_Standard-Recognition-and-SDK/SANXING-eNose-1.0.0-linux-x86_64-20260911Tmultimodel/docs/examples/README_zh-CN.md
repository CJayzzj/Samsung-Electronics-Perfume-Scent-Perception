# SDK 示例

## sdk_client_example.py

连接已经启动的推理服务，发送一帧模拟 ADC 数据并打印 `ready` 和
`frame_response`。该示例用于验证 SDK 网络接口，不代表真实气味识别结果。

```bash
python3 sdk_client_example.py --host 127.0.0.1 --port 5006
```

## serial_inference_example.py

连接推理服务和真实电子鼻，持续转发传感器帧并打印推理响应：

```bash
python3 serial_inference_example.py \
  --serial-port /dev/ttyCH341USB0 \
  --inference-host 127.0.0.1 \
  --inference-port 5006
```

按 `Ctrl+C` 停止。同一时间只能有一个程序打开电子鼻串口。

## JSONL 示例

- `sensor_frame.jsonl`：客户端发送的传感器帧；
- `ready.jsonl`：服务连接就绪消息；
- `result.jsonl`：包含最终识别结果的响应；
- `adc_frame_example.jsonl`：ADC 输入帧示例。
