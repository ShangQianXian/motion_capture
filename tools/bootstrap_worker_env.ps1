<#
.SYNOPSIS
    Create the external worker virtual environment and install the inference stack.

.DESCRIPTION
    Follows docs/TECHNICAL_DESIGN.md section 3.2. Deep-learning packages are
    installed into <repo>/.venv and never into Blender's bundled Python.

    Install order matters: PyTorch first, then MMCV / MMDetection / MMPose via mim.

    This downloads several GB. Use -DryRun to print the commands only.

.EXAMPLE
    .\tools\bootstrap_worker_env.ps1 -DryRun
    .\tools\bootstrap_worker_env.ps1
    .\tools\bootstrap_worker_env.ps1 -CudaIndex https://download.pytorch.org/whl/cu121
    .\tools\bootstrap_worker_env.ps1 -CpuOnly
#>
[CmdletBinding()]
param(
    [string] $PythonVersion = '3.12',
    [string] $CudaIndex = 'https://download.pytorch.org/whl/cu128',
    [switch] $CpuOnly,
    [switch] $SkipOpenMMLab,
    [switch] $DryRun
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $repoRoot '.venv'
$venvPython = Join-Path $venvPath 'Scripts\python.exe'

function Invoke-Cmd {
    param([string] $Executable, [string[]] $Arguments)
    $display = "$Executable $($Arguments -join ' ')"
    if ($DryRun) {
        Write-Host "[dry-run] $display" -ForegroundColor Yellow
        return
    }
    Write-Host "> $display" -ForegroundColor Cyan
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "command failed ($LASTEXITCODE): $display" }
}

Write-Host "worker venv: $venvPath"

if (-not (Test-Path $venvPython)) {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        Invoke-Cmd $launcher.Source @("-$PythonVersion", '-m', 'venv', $venvPath)
    }
    else {
        $fallback = Get-Command python -ErrorAction SilentlyContinue
        if (-not $fallback) { throw 'No Python interpreter found. Install Python 3.10+ first.' }
        Write-Warning "py launcher not found; using $($fallback.Source) instead of Python $PythonVersion"
        Invoke-Cmd $fallback.Source @('-m', 'venv', $venvPath)
    }
}
else {
    Write-Host 'venv already exists, reusing it'
}

if ($DryRun -and -not (Test-Path $venvPython)) {
    Write-Host '[dry-run] remaining steps would use .venv\Scripts\python.exe'
    $venvPython = 'python'
}

Invoke-Cmd $venvPython @('-m', 'pip', 'install', '--upgrade', 'pip')

# 1) PyTorch first - the OpenMMLab packages compile against it.
if ($CpuOnly) {
    Invoke-Cmd $venvPython @('-m', 'pip', 'install', 'torch', 'torchvision')
}
else {
    Invoke-Cmd $venvPython @('-m', 'pip', 'install', 'torch', 'torchvision', '--index-url', $CudaIndex)
}

# 2) Everything the preview pipeline and media decoding need.
Invoke-Cmd $venvPython @('-m', 'pip', 'install', 'opencv-python', 'mediapipe', 'numpy', 'scipy', 'tqdm')

# 3) OpenMMLab stack for the high quality pipeline.
if (-not $SkipOpenMMLab) {
    Invoke-Cmd $venvPython @('-m', 'pip', 'install', '-U', 'openmim')
    Invoke-Cmd $venvPython @('-m', 'mim', 'install', 'mmengine', 'mmcv', 'mmdet', 'mmpose')
}

Write-Host ''
Write-Host 'verifying the environment' -ForegroundColor Cyan
if (-not $DryRun) {
    Push-Location $repoRoot
    try {
        & $venvPython -m backend_worker.cli --check-env
        & $venvPython -m backend_worker.cli --check-cuda
    }
    finally { Pop-Location }
}

Write-Host ''
Write-Host 'Next steps:'
Write-Host "  1. In Blender, set Worker Python to: $venvPython"
Write-Host "  2. Set Models Root to: $(Join-Path $repoRoot 'models')"
Write-Host '  3. Download the model files listed in docs/INSTALL.md'
# Keep this file ASCII-only: Windows PowerShell 5.1 misreads UTF-8 .ps1 files that
# have no BOM, which would garble any non-ASCII literal printed here.
Write-Host '  4. Click the "Check Model Environment" button in the Mocap sidebar'
