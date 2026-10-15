# 推理程序开发与未来构建区

当前阶段直接用`inference_main.py`在Windows运行开源推理服务，便于检查Score、
寻峰和恢复计时。Linux闭源构建将在实时链路稳定后进行，以下命令暂时保留备用。

Windows构建示例：

```powershell
python -m pip install -r requirements-build.txt
.\build_windows.ps1 ..\model_candidates\0821_perfume_cigarette_cascade
```

Linux构建示例：

```bash
python3 -m pip install -r requirements-build.txt
bash build_linux.sh ../model_candidates/0821_perfume_cigarette_cascade
```

必须在甲方目标Linux的同CPU架构、发行版和glibc环境构建并实机测试。构建脚本每次只嵌入一套模型，并在编译前检查三件运行文件能否成套加载。
