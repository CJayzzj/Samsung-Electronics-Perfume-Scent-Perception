<#
Windows闭源推理程序构建脚本。

功能：接收一个完整的候选模型目录，先检查模型能否正常加载，再用Nuitka把
推理源码和选定模型封装成单个exe。构建时只嵌入一套模型，避免模型文件混用。

用法示例：
  .\build_windows.ps1 ..\model_candidates\0817_morning_gbt
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$ModelProfileDir,
    [string]$OutputName = "enose_infer.exe",
    # OneFile生成单个exe；Standalone生成exe加运行库目录，适合被安全软件误报时使用。
    [ValidateSet("OneFile", "Standalone")]
    [string]$BuildMode = "OneFile"
)

# 脚本从任意目录启动都可以；下面先切换到脚本自身所在目录。
$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $scriptDir
# Nuitka默认写入用户缓存目录；受限环境可能没有权限，因此固定到项目构建目录。
$nuitkaCacheDir = Join-Path $scriptDir ".nuitka_cache"
New-Item -ItemType Directory -Force -Path $nuitkaCacheDir | Out-Null
$env:NUITKA_CACHE_DIR = $nuitkaCacheDir
# 将模型目录解析成绝对路径，并拼出运行必需的三个文件。
$profileDir = (Resolve-Path -LiteralPath $ModelProfileDir).Path
$modelFile = Join-Path $profileDir "model.joblib"
$classMapFile = Join-Path $profileDir "class_map.json"
$runtimeConfigFile = Join-Path $profileDir "runtime_config.json"

# 构建前逐个检查文件，缺少任意一个就停止，避免生成不能运行的exe。
foreach ($file in @($modelFile, $classMapFile, $runtimeConfigFile)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Missing model runtime file: $file"
    }
}

# 调用推理入口的自检，确认模型、类别映射和运行配置属于同一套模型。
python .\inference_main.py --self-test `
    --model $modelFile `
    --class-map $classMapFile `
    --runtime-config $runtimeConfigFile
if ($LASTEXITCODE -ne 0) {
    throw "Model bundle self-test failed; build stopped."
}

# 组织Nuitka参数：由入口代码自动追踪实际依赖，并嵌入三个运行资源。
# 不强制包含整个sklearn/joblib包，否则会把测试和matplotlib界面模块也打入exe。
$nuitkaArgs = @(
    "-m", "nuitka",
    "--follow-imports",
    # 首次构建时自动下载Nuitka指定版本的MinGW64，避免脚本停在交互确认处。
    "--assume-yes-for-downloads",
    # 训练与推理不会用到测试、绘图和文档包，排除后可明显减小体积和构建时间。
    "--nofollow-import-to=sklearn.tests,sklearn.tests.*,scipy.tests,scipy.tests.*,numpy.tests,numpy.tests.*,joblib.test,joblib.test.*,matplotlib,matplotlib.*,sphinx,sphinx.*,numpydoc,numpydoc.*,IPython,IPython.*,pytest,pytest.*",
    "--output-filename=$OutputName",
    "--include-data-file=$modelFile=resources/model.joblib",
    "--include-data-file=$classMapFile=resources/class_map.json",
    "--include-data-file=$runtimeConfigFile=resources/runtime_config.json",
    ".\inference_main.py"
)
# 根据选择加入单文件或目录模式。两种模式都会编译核心Python代码，不会交付源码。
if ($BuildMode -eq "OneFile") {
    $nuitkaArgs = @("-m", "nuitka", "--onefile") + $nuitkaArgs[2..($nuitkaArgs.Count - 1)]
}
else {
    $standaloneRoot = Join-Path $scriptDir (([IO.Path]::GetFileNameWithoutExtension($OutputName)) + "_standalone")
    New-Item -ItemType Directory -Force -Path $standaloneRoot | Out-Null
    $nuitkaArgs = @("-m", "nuitka", "--standalone", "--output-dir=$standaloneRoot") + $nuitkaArgs[2..($nuitkaArgs.Count - 1)]
}
# 正式执行单文件构建，输出名称由 $OutputName 指定。
python @nuitkaArgs
if ($LASTEXITCODE -ne 0) {
    throw "Nuitka build failed."
}
