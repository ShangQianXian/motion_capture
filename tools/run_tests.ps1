<#
.SYNOPSIS
    Run every automated test: unit tests plus the Blender background tests.

.DESCRIPTION
    Unit tests need no third-party packages and run on any Python 3.10+.
    The Blender tests are skipped (not failed) when no Blender executable is found.

.EXAMPLE
    .\tools\run_tests.ps1
    .\tools\run_tests.ps1 -SkipBlender
    .\tools\run_tests.ps1 -Blender 'E:\SoftWare\Blender\blender-4.5.0-windows-x64\blender.exe'
#>
[CmdletBinding()]
param(
    [string] $Python,
    [string[]] $Blender = @(),
    [switch] $SkipBlender,
    [switch] $SkipWorker
)

# Native tools write progress to stderr; that must not abort the run.
$ErrorActionPreference = 'Continue'

$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $Python) {
    $Python = Join-Path $repoRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $Python)) { $Python = 'E:\SoftWare\Python\Python3.10.0\python.exe' }
    if (-not (Test-Path -LiteralPath $Python)) { throw 'Pass -Python with a Python 3.10.0 executable.' }
}
$failures = New-Object System.Collections.Generic.List[string]

function Invoke-Step {
    param([string] $Name, [scriptblock] $Body)
    Write-Host ''
    Write-Host "=== $Name ===" -ForegroundColor Cyan
    try { & $Body; $stepExit = $LASTEXITCODE }
    catch { Write-Host $_ -ForegroundColor Red; $stepExit = 1 }
    if ($stepExit -ne 0) {
        Write-Host "FAILED: $Name (exit $stepExit)" -ForegroundColor Red
        $script:failures.Add($Name)
    }
    else {
        Write-Host "passed: $Name" -ForegroundColor Green
    }
}

# ---------------------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------------------

Invoke-Step 'unit tests (unittest)' {
    Push-Location $repoRoot
    try {
        # unittest writes its progress to stderr. Stringifying each record keeps
        # Windows PowerShell 5.1 from rendering them as scary NativeCommandError
        # blocks; pass/fail is decided by $LASTEXITCODE, not by the stream.
        & $Python -m unittest discover -s tests/unit -t . 2>&1 |
            ForEach-Object { $_.ToString() } | Out-Host
    }
    finally { Pop-Location }
}

# pytest is optional; run it only when it is installed.
Push-Location $repoRoot
$hasPytest = $false
try {
    & $Python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('pytest') else 1)" 2>$null
    $hasPytest = ($LASTEXITCODE -eq 0)
}
finally { Pop-Location }

if ($hasPytest) {
    Invoke-Step 'unit tests (pytest)' {
        Push-Location $repoRoot
        try {
            & $Python -m pytest tests/unit -q 2>&1 |
                ForEach-Object { $_.ToString() } | Out-Host
        }
        finally { Pop-Location }
    }
}
else {
    Write-Host ''
    Write-Host 'skipped: pytest not installed (unittest already covered the suite)' -ForegroundColor Yellow
}

# ---------------------------------------------------------------------------------------
# Blender background tests
# ---------------------------------------------------------------------------------------

if (-not $SkipBlender) {
    $candidates = New-Object System.Collections.Generic.List[string]
    foreach ($explicit in $Blender) { $candidates.Add($explicit) }

    if ($candidates.Count -eq 0) {
        $onPath = Get-Command blender -ErrorAction SilentlyContinue
        if ($onPath) { $candidates.Add($onPath.Source) }
        foreach ($base in @('C:\Program Files\Blender Foundation', 'E:\SoftWare\Blender', 'D:\SoftWare\Blender')) {
            if (Test-Path $base) {
                Get-ChildItem -Path $base -Filter 'blender.exe' -Recurse -Depth 3 -ErrorAction SilentlyContinue |
                    Where-Object { $_.FullName -match '4\.5' } |
                    ForEach-Object { $candidates.Add($_.FullName) }
            }
        }
    }

    $found = $candidates | Where-Object { Test-Path $_ } | Select-Object -Unique
    if (-not $found) {
        Write-Host ''
        Write-Host 'skipped: no Blender executable found (pass -Blender <path>)' -ForegroundColor Yellow
    }

    foreach ($exe in $found) {
        foreach ($test in @('test_enable_addon.py', 'test_mock_retarget.py', 'test_retarget_coordinates.py', 'test_review_workflow.py')) {
            Invoke-Step "blender $test ($exe)" {
                Push-Location $repoRoot
                try {
                    & $exe --background --factory-startup --python-exit-code 1 --python "tests\blender\$test" 2>&1 |
                        Select-String -Pattern 'Blender \d.*Python|checks, \d+ failures|RESULT|FAILED:|FATAL' |
                        Out-Host
                }
                finally { Pop-Location }
            }
        }
    }
}

# ---------------------------------------------------------------------------------------
# Worker integration tests (no trained models required)
# ---------------------------------------------------------------------------------------
if (-not $SkipWorker) {
    foreach ($kind in @('quality', 'preview')) {
        $folder = if ($kind -eq 'quality') { '.venv' } else { '.venv-preview' }
        $workerPython = Join-Path $repoRoot "$folder\Scripts\python.exe"
        if (-not (Test-Path -LiteralPath $workerPython)) {
            Write-Host "skipped: $kind worker is not installed" -ForegroundColor Yellow
            continue
        }
        Invoke-Step "$kind adapter contracts" {
            Push-Location $repoRoot
            try { & $workerPython -m unittest discover -s tests/worker -t . 2>&1 | ForEach-Object { $_.ToString() } | Out-Host }
            finally { Pop-Location }
        }
        Invoke-Step "$kind environment" {
            Push-Location $repoRoot
            try { & $workerPython -m backend_worker.cli --check-env --profile $kind 2>&1 | ForEach-Object { $_.ToString() } | Out-Host }
            finally { Pop-Location }
        }
        if ($kind -eq 'quality') {
            Invoke-Step 'CUDA operators' {
                Push-Location $repoRoot
                try { & $workerPython -m backend_worker.cli --check-cuda 2>&1 | ForEach-Object { $_.ToString() } | Out-Host }
                finally { Pop-Location }
            }
        }
    }
}

# ---------------------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------------------

Write-Host ''
if ($failures.Count -gt 0) {
    Write-Host "$($failures.Count) step(s) failed:" -ForegroundColor Red
    foreach ($name in $failures) { Write-Host "  - $name" -ForegroundColor Red }
    exit 1
}
Write-Host 'ALL TESTS PASSED' -ForegroundColor Green
exit 0
