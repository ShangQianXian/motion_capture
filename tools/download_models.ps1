<#
.SYNOPSIS
    Download model weights and complete configs from the project manifest.
.EXAMPLE
    .\tools\download_models.ps1 -Profile quality -DryRun
    .\tools\download_models.ps1 -Profile preview -Artifact mediapipe_hand
    .\tools\download_models.ps1 -Profile quality
#>
[CmdletBinding()]
param(
    [string] $PythonExe,
    [string] $ModelsRoot,
    [string] $Manifest,
    [ValidateSet('preview', 'quality', 'quality_plus', 'fallback_cpu', 'hand_enhanced')]
    [string[]] $Profile = @(),
    [string[]] $Artifact = @(),
    [switch] $All,
    [switch] $IncludeOptional,
    [switch] $ConfigsOnly,
    [switch] $DryRun,
    [switch] $List,
    [switch] $VerifyOnly,
    [switch] $Force,
    [double] $Timeout = 30,
    [int] $Retries = 3
)
$ErrorActionPreference = 'Stop'
if (-not $PythonExe) {
    $PythonExe = Join-Path (Split-Path -Parent $PSScriptRoot) '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $PythonExe)) { $PythonExe = 'E:\SoftWare\Python\Python3.10.0\python.exe' }
    if (-not (Test-Path -LiteralPath $PythonExe)) { throw 'Pass -PythonExe with a Python 3.10+ executable.' }
}
$arguments = @((Join-Path $PSScriptRoot 'download_models.py'), '--timeout', $Timeout.ToString([cultureinfo]::InvariantCulture), '--retries', "$Retries")
if ($ModelsRoot) { $arguments += @('--models-root', $ModelsRoot) }
if ($Manifest) { $arguments += @('--manifest', $Manifest) }
foreach ($item in $Profile) { $arguments += @('--profile', $item) }
foreach ($item in $Artifact) { $arguments += @('--artifact', $item) }
if ($All) { $arguments += '--all' }
if ($IncludeOptional) { $arguments += '--include-optional' }
if ($ConfigsOnly) { $arguments += '--configs-only' }
if ($DryRun -or $List) { $arguments += '--list' }
if ($VerifyOnly) { $arguments += '--verify-only' }
if ($Force) { $arguments += '--force' }
& $PythonExe -I @arguments
exit $LASTEXITCODE
