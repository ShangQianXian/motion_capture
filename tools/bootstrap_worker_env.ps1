<#
.SYNOPSIS
    Install isolated Python 3.10.0 x64 workers for Quality and Preview.
.EXAMPLE
    .\tools\bootstrap_worker_env.ps1 -PythonExe E:\SoftWare\Python\Python3.10.0\python.exe
    .\tools\bootstrap_worker_env.ps1 -Environment preview -DryRun
#>
[CmdletBinding()]
param(
    [string] $PythonExe,
    [ValidateSet('all', 'quality', 'preview')] [string] $Environment = 'all',
    [switch] $DryRun,
    [switch] $CpuOnly,
    [switch] $SkipOpenMMLab,
    [string] $PythonVersion = '3.10',
    [string] $CudaIndex = 'https://download.pytorch.org/whl/cu121'
)
$ErrorActionPreference = 'Stop'
if ($PythonVersion -notin @('3.10', '3.10.0')) { throw 'Only Python 3.10.0 x64 is supported by these locks.' }
if ($CudaIndex -ne 'https://download.pytorch.org/whl/cu121') { throw 'The Quality lock requires cu121 and its matching MMCV wheel.' }
if ($CpuOnly -or $SkipOpenMMLab) { $Environment = 'preview' }
if (-not $PythonExe) {
    $localInstall = 'E:\SoftWare\Python\Python3.10.0\python.exe'
    if (Test-Path -LiteralPath $localInstall) { $PythonExe = $localInstall }
    else {
        $launcher = Get-Command py -ErrorAction SilentlyContinue
        if (-not $launcher) { throw 'Pass -PythonExe with the absolute path to Python 3.10.0 x64.' }
        $PythonExe = & $launcher.Source -3.10 -I -c 'import sys;print(sys.executable)'
        if ($LASTEXITCODE -ne 0) { throw 'Python 3.10 not registered. Pass -PythonExe explicitly.' }
    }
}
$arguments = @((Join-Path $PSScriptRoot 'bootstrap_worker_env.py'), '--python-exe', $PythonExe, '--environment', $Environment)
if ($DryRun) { $arguments += '--dry-run' }
& $PythonExe -I @arguments
if ($LASTEXITCODE -ne 0) { throw "Worker setup failed (exit $LASTEXITCODE)." }
