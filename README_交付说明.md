# 三星电子鼻 Linux 客户交付说明

## 交付内容

本目录包含三个相互独立的客户组件包：

- `packages/sanxing-enose-demo-1.0.0-linux-x86_64.tar.gz`：电子鼻识别演示客户端；
- `packages/sanxing-enose-inference-service-1.0.0-linux-x86_64.tar.gz`：电子鼻推理服务，内置多设备模型并默认使用 SX_01；
- `packages/sanxing-enose-device-sdk-1.0.0-linux-x86_64.tar.gz`：电子鼻设备接入 SDK、参考客户端、公开源码、协议和示例。

以上三个组件包面向 Linux x86_64，构建基线为 Ubuntu 20.04。

## 文档导航

- `docs/00_快速开始.md`：三种典型使用方式；
- `docs/01_安装与系统要求.md`：运行环境、串口权限和 Python 依赖；
- `docs/02_推理服务部署.md`：启动参数、模型选择和 systemd 部署；
- `docs/03_SDK_API手册.md`：Python 接口的入参、返回值、回调和异常；
- `docs/04_协议说明.md`：TCP JSONL 请求、响应和错误码；
- `docs/05_故障排查.md`：常见问题及处理方法；
- `docs/06_客户验收清单.md`：交付验收步骤与预期结果；
- `docs/examples/`：可直接运行的 Python 示例和 JSONL 示例数据。

## 完整性校验

在本目录执行：

```bash
sha256sum -c SHA256SUMS.txt
```

解压每个程序包后，还可以在其目录内执行 `sha256sum -c SHA256SUMS`，复核包内
逐文件校验和。

## 1. 电子鼻识别演示客户端

```bash
tar -xzf packages/sanxing-enose-demo-1.0.0-linux-x86_64.tar.gz
cd sanxing-enose-demo-1.0.0-linux-x86_64/application
./enose-demo
```

在界面中选择完整模型目录：

```text
model_registry/SX_01/20260911T142306_enose_core_v1
model_registry/SX_02/20260911T135526_enose_core_v1
```

随后加载模型、选择 `/dev/ttyCH341USB0`、连接设备，等待60秒预热结束后开始
推理。识别演示客户端直接读取电子鼻并在本机推理，不需要启动推理服务。

## 2. 电子鼻推理服务和设备接入参考客户端

先解压：

```bash
tar -xzf packages/sanxing-enose-inference-service-1.0.0-linux-x86_64.tar.gz
tar -xzf packages/sanxing-enose-device-sdk-1.0.0-linux-x86_64.tar.gz
```

终端一启动电子鼻推理服务：

```bash
cd sanxing-enose-inference-service-1.0.0-linux-x86_64/application
./enose-inference-service --self-test
./enose-inference-service --host 127.0.0.1 --port 5006
```

不指定 `--bundle` 时，推理服务加载随程序提供的默认 SX_01 模型。需要使用内置
的其他设备模型时，指定对应完整模型目录：

```bash
./enose-inference-service \
  --bundle model_registry/SX_02/20260911T135526_enose_core_v1 \
  --host 127.0.0.1 \
  --port 5006
```

模型目录必须包含 `manifest.json`、`model.joblib`、`class_map.json` 和
`runtime_config.json`。建议先使用相同的 `--bundle` 参数执行 `--self-test`，确认
模型包校验和加载成功后再启动服务。

终端二启动电子鼻设备接入参考客户端：

```bash
mkdir -p "$HOME/enose-client-data"
cd sanxing-enose-device-sdk-1.0.0-linux-x86_64/application
./enose-client \
  --serial-port /dev/ttyCH341USB0 \
  --inference-host 127.0.0.1 \
  --inference-port 5006 \
  --save-dir "$HOME/enose-client-data"
```

按 `Ctrl+C` 安全停止。参考客户端保存的 CSV/XLSX 位于 `--save-dir` 指定目录。

## 3. 电子鼻设备接入 Python SDK

Python 3.8 或更高版本可以安装 SDK：

```bash
python3 -m pip install \
  sanxing-enose-device-sdk-1.0.0-linux-x86_64/python/enose_sdk-1.0.0-py3-none-any.whl
```

SDK 依赖 `pyserial==3.5`；使用 XLSX 保存时还需要 `openpyxl==3.1.5`。免安装
参考客户端 `application/enose-client` 已自带运行依赖。

## 串口注意事项

`enose-demo` 和 `enose-client` 都会打开电子鼻串口，不能同时运行。推理服务
`enose-inference-service` 不直接打开串口，可以与 `enose-client` 同时运行。

如果当前用户没有串口权限，可执行：

```bash
sudo usermod -aG dialout "$USER"
```

随后注销并重新登录。不要从程序包中单独移动可执行文件；`application/_internal`
和模型目录必须与可执行文件保持原有相对位置。
