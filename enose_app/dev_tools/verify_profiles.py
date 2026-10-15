# -*- coding: utf-8 -*-
"""候选模型包自检工具。

本文件读取模型清单，逐套检查运行文件、模型哈希和model_id，随后调用
闭源推理入口执行真实加载自检，用来防止模型、类别映射或配置被误混用。
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


# 所有路径都由脚本位置计算，避免依赖用户当前所在目录。
APP_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = APP_ROOT / "private_build"
MANIFEST_PATH = APP_ROOT / "model_candidates" / "manifest.json"


def sha256(path):
    """分块计算文件SHA-256，较大的模型文件也不会一次读入内存。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    """检查清单中的全部候选模型；任一失败就返回退出码2。"""

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    failures = []
    for profile in manifest["profiles"]:
        # 每个profile必须成套包含模型、类别映射和运行配置三个文件。
        profile_dir = APP_ROOT / "model_candidates" / profile["directory"]
        paths = {
            name: profile_dir / name
            for name in ("model.joblib", "class_map.json", "runtime_config.json")
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            failures.append("%s 缺少文件：%s" % (profile["name"], missing))
            continue
        # 哈希变化通常意味着模型被替换、损坏，或清单没有同步更新。
        if sha256(paths["model.joblib"]) != profile["model_sha256"]:
            failures.append("%s model.joblib哈希不一致" % profile["name"])
            continue
        # 使用独立子进程走正式inference_main加载流程，比只读JSON更可靠。
        command = [
            sys.executable,
            str(PRIVATE_ROOT / "inference_main.py"),
            "--self-test",
            "--model",
            str(paths["model.joblib"]),
            "--class-map",
            str(paths["class_map.json"]),
            "--runtime-config",
            str(paths["runtime_config.json"]),
        ]
        # 中文Windows默认可能使用GBK；明确要求子进程输出UTF-8，避免ready中的
        # “回流1”等中文被读取线程按错误编码解码。
        child_env = os.environ.copy()
        child_env["PYTHONIOENCODING"] = "utf-8"
        completed = subprocess.run(
            command,
            cwd=str(PRIVATE_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            env=child_env,
        )
        if completed.returncode != 0:
            failures.append("%s 自检失败：%s" % (profile["name"], completed.stderr.strip()))
            continue
        # 自检成功时最后一行是ready消息，再核对其中实际加载的model_id。
        ready = json.loads(completed.stdout.strip().splitlines()[-1])
        actual_id = (ready.get("model") or {}).get("model_id")
        if actual_id != profile["model_id"]:
            failures.append("%s model_id不一致：%s" % (profile["name"], actual_id))
            continue
        print("[通过] %s -> %s" % (profile["name"], actual_id))
    # 集中打印全部问题，便于一次修完多个候选包。
    if failures:
        for message in failures:
            print("[失败] %s" % message, file=sys.stderr)
        return 2
    print("清单中的全部候选模型均通过成套加载自检。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
