#!/usr/bin/env bash
# Linux闭源推理程序构建脚本。
# 功能：检查一套模型资源是否完整且可加载，再用Nuitka将推理源码和该模型
# 封装为一个Linux可执行文件。脚本只嵌入指定的一套模型，不会混用候选模型。
# 用法：bash build_linux.sh ../model_candidates/0817_morning_gbt [程序名]

set -euo pipefail

# 无论从哪个目录调用，都先进入脚本自身所在的 private_build 目录。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# 参数1是候选模型目录；参数2可选，用于指定输出程序名。
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 MODEL_PROFILE_DIR [OUTPUT_NAME]" >&2
  exit 2
fi

# 解析模型绝对路径，并拼出运行必需的三个资源文件。
PROFILE_DIR="$(cd -- "$1" && pwd)"
OUTPUT_NAME="${2:-enose_infer}"
MODEL_FILE="${PROFILE_DIR}/model.joblib"
CLASS_MAP_FILE="${PROFILE_DIR}/class_map.json"
RUNTIME_CONFIG_FILE="${PROFILE_DIR}/runtime_config.json"

# 缺少任意运行文件时立即停止，避免构建出无法启动的程序。
for file in "${MODEL_FILE}" "${CLASS_MAP_FILE}" "${RUNTIME_CONFIG_FILE}"; do
  [[ -f "${file}" ]] || { echo "Missing runtime file: ${file}" >&2; exit 2; }
done

# 构建前先验证模型、类别映射和运行配置能够成套加载。
python3 inference_main.py --self-test \
  --model "${MODEL_FILE}" \
  --class-map "${CLASS_MAP_FILE}" \
  --runtime-config "${RUNTIME_CONFIG_FILE}" >/dev/null

# 执行Nuitka单文件构建，并把三项运行资源嵌入可执行程序内部。
# 依赖由入口自动追踪，不强制包含整个sklearn/joblib测试包。
python3 -m nuitka \
  --onefile \
  --follow-imports \
  --output-filename="${OUTPUT_NAME}" \
  --include-data-file="${MODEL_FILE}=resources/model.joblib" \
  --include-data-file="${CLASS_MAP_FILE}=resources/class_map.json" \
  --include-data-file="${RUNTIME_CONFIG_FILE}=resources/runtime_config.json" \
  inference_main.py
