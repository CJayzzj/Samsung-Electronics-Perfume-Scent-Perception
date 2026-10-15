# 三星电子鼻交付程序工作区

此目录用于先在Windows跑通“公开机器人主程序＋开源推理程序”的实时链路。
Linux闭源构建留到数值、寻峰和真机通讯稳定以后再做。

## 目录作用

| 目录 | 作用 | 是否向甲方提供源码 |
| --- | --- | --- |
| `public` | 串口接收16路电阻和温湿度、逐帧保存、转发当前帧、显示result | 是 |
| `private_build` | 当前开源联调使用的补偿、动态基线、寻峰、特征、模型适配和TCP服务 | 当前开发使用 |
| `model_candidates` | 相互隔离的候选运行资源 | 否，最终只嵌入选中一套 |
| `docs` | 双方需要冻结的数据格式与结果格式 | 是 |
| `dev_tools` | 模拟电子鼻和检查模型包 | 否 |

## 关键文件

- `public/main.py`：甲方可见的主入口。
- `public/common/protocol.py`：公开的数据格式校验程序，是协议的代码实现。
- `public/robot_app/sources.py`：串口、开发TCP或stdin收数。
- `public/robot_app/storage.py`：异步逐帧保存CSV/XLSX并自动分卷。
- `public/robot_app/inference_client.py`：连接本机推理端口并接收result。
- `private_build/inference_main.py`：当前推理服务的Python启动入口；未来也可作为Linux闭源构建入口。
- `private_build/inference/engine.py`：补偿、基线、1000帧队列和持续寻峰。
- `private_build/inference/model_adapter.py`：按模型声明构造特征并校验模型包。
- `private_build/inference/score_model.py`：Score模型反序列化和推理所需实现。
- `private_build/inference/tcp_service.py`：本机TCP JSONL服务。
- `private_build/sensing`：基线和响应特征的唯一实现。

公开端和闭源端各带一份同版本`common/protocol.py`，因为交付后一个是Python源码、另一个已嵌入二进制。发布时必须保持两份文件完全一致并同时升级协议版本。

## 当前实时模型

本轮使用`0821_perfume_cigarette_cascade`级联模型：先用16路group_zscore＋
RBF-SVM用香烟样本判断香水/异味；异味对外返回`异味/O01`，香水再进入16路resp＋16路pattern的全天GBT四分类。
时间留出验证为46/46；全量固定六折OOF为138/138。该结果仍需新日期独立数据复验。
运行配置已绑定`回流1`温湿度补偿参数和16路1 ppm门限；设备号或标定缺失时会停止处理。

先检查三套文件：

```powershell
python .\dev_tools\verify_profiles.py
```

## 开发运行

终端一，从Python源码启动香水/香烟级联推理服务：

```powershell
Set-Location .\private_build
python .\inference_main.py --model ..\model_candidates\0821_perfume_cigarette_cascade\model.joblib --class-map ..\model_candidates\0821_perfume_cigarette_cascade\class_map.json --runtime-config ..\model_candidates\0821_perfume_cigarette_cascade\runtime_config.json
```

终端二，在公开端连接真实串口：

```powershell
Set-Location .\public
python .\main.py --sensor-transport serial --serial-port COM13 --baudrate 115200 --device-id 回流1 --inference-port 5006 --sample-rate-hz 1 --queue-length 1000 --save-format csv
```

真机串口协议说明：电子鼻下位机使用与采集工具 `JL_sys_v3_6` 一致的二进制帧协议（每秒请求一次读数命令 `01 03 00 00 00 12 C5 C7`，响应为41字节帧：帧头 + 16路ADC大端 + 温度 + 湿度）。主程序自动解析并换算为16路电阻(kΩ)、温度(℃)、湿度(%RH)后转发给推理端，不需要手动把串口数据转成JSONL。若需改用JSONL文本串口（联调备用），加参数 `--serial-protocol jsonl`。

数值对齐回归检查：

```powershell
python .\dev_tools\verify_v3_score_parity.py
```

该工具默认回放8月20日原始CSV，逐帧核对补偿电阻、基线、resp和Score。

## 尚未完成的交付门禁

1. 用COM13真机逐类验证每个气味事件只返回一次result和一次recovery。
2. 用新采集峰值核对实时峰顶帧与训练Excel特征是否同口径。
3. 按最终固件增加串口CRC、断线重连和推理TCP重连。
4. 收集咖啡等未知异味并重新评估开集拒识；当前只闭集识别香烟异味。
5. 最终再冻结模型、P01～P04映射、协议版本、文件哈希和甲方Linux环境。
