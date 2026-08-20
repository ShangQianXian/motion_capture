<#
.SYNOPSIS
    Build the distributable add-on zip.

.DESCRIPTION
    Produces dist/motion_capture-<version>.zip whose single top-level folder is
    "motion_capture", which is what Blender's "Install from Disk" expects.

    Included : add-on Python source, docs/, models/manifest.example.json, tools/
    Excluded : .venv, .git, .idea, __pycache__, dist, downloaded weights
               (*.pth, *.task, *.onnx), .mocap_jobs, logs and test media

.EXAMPLE
    .\tools\build_zip.ps1
    .\tools\build_zip.ps1 -OutputDir D:\releases
#>
[CmdletBinding()]
param(
    [string] $OutputDir
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$packageName = 'motion_capture'

if ((Split-Path -Leaf $repoRoot) -ne $packageName) {
    throw "Repository root must be named '$packageName' because it is the add-on package."
}

# Read the version straight from bl_info so the zip name can never drift.
$initText = Get-Content -Raw -Encoding UTF8 (Join-Path $repoRoot '__init__.py')
$match = [regex]::Match($initText, '"version"\s*:\s*\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)')
if (-not $match.Success) { throw 'Could not read "version" from bl_info in __init__.py' }
$version = '{0}.{1}.{2}' -f $match.Groups[1].Value, $match.Groups[2].Value, $match.Groups[3].Value

if (-not $OutputDir) { $OutputDir = Join-Path $repoRoot 'dist' }
$zipPath = Join-Path $OutputDir ("{0}-{1}.zip" -f $packageName, $version)

# Directory names that are never packaged.
$excludedDirs = @('.git', '.venv', '.idea', '.agents', '__pycache__', 'dist', '.mocap_jobs', '.pytest_cache')
# File patterns that are never packaged (model weights must stay out).
$excludedFilePatterns = @('*.pth', '*.task', '*.onnx', '*.pt', '*.ckpt', '*.zip', '*.log', '*.blend1', '*.mp4', '*.mov', '*.avi', '*.mkv')

Write-Host "packaging $packageName $version"

$staging = Join-Path ([System.IO.Path]::GetTempPath()) ("mocap_pkg_" + [System.Guid]::NewGuid().ToString('N'))
$stagedPackage = Join-Path $staging $packageName
New-Item -ItemType Directory -Force -Path $stagedPackage | Out-Null

$copied = 0
$skippedWeights = 0

Get-ChildItem -Path $repoRoot -Recurse -File -Force | ForEach-Object {
    $relative = $_.FullName.Substring($repoRoot.Length).TrimStart('\', '/')
    $parts = $relative -split '[\\/]'

    foreach ($part in $parts) {
        if ($excludedDirs -contains $part) { return }
    }
    foreach ($pattern in $excludedFilePatterns) {
        if ($_.Name -like $pattern) {
            if ($_.Name -like '*.pth' -or $_.Name -like '*.task') { $script:skippedWeights++ }
            return
        }
    }

    $destination = Join-Path $stagedPackage $relative
    $destinationDir = Split-Path -Parent $destination
    if (-not (Test-Path $destinationDir)) { New-Item -ItemType Directory -Force -Path $destinationDir | Out-Null }
    Copy-Item -LiteralPath $_.FullName -Destination $destination -Force
    $script:copied++
}

if (-not (Test-Path (Join-Path $stagedPackage '__init__.py'))) {
    throw 'Staging is missing __init__.py; refusing to build a broken package.'
}
if (-not (Test-Path (Join-Path $stagedPackage 'models\manifest.example.json'))) {
    throw 'Staging is missing models/manifest.example.json; refusing to build.'
}

if (-not (Test-Path $OutputDir)) { New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null }
if (Test-Path $zipPath) { Remove-Item -LiteralPath $zipPath -Force }

# Entries are written by hand rather than with Compress-Archive: on Windows
# PowerShell that cmdlet emits backslash separators, which violates the ZIP spec
# and can make Blender's "Install from Disk" create one oddly named file instead
# of a package directory.
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$archive = [System.IO.Compression.ZipFile]::Open($zipPath, [System.IO.Compression.ZipArchiveMode]::Create)
try {
    Get-ChildItem -Path $stagedPackage -Recurse -File -Force | ForEach-Object {
        $relative = $_.FullName.Substring($staging.Length).TrimStart('\', '/').Replace('\', '/')
        $entry = $archive.CreateEntry($relative, [System.IO.Compression.CompressionLevel]::Optimal)
        $entryStream = $entry.Open()
        try {
            $fileStream = [System.IO.File]::OpenRead($_.FullName)
            try { $fileStream.CopyTo($entryStream) }
            finally { $fileStream.Dispose() }
        }
        finally { $entryStream.Dispose() }
    }
}
finally { $archive.Dispose() }

Remove-Item -Recurse -Force $staging

# Verify the archive really uses forward slashes and has a single root folder.
$verify = [System.IO.Compression.ZipFile]::OpenRead($zipPath)
try {
    $entryNames = $verify.Entries | ForEach-Object { $_.FullName }
}
finally { $verify.Dispose() }

$backslashes = @($entryNames | Where-Object { $_.Contains('\') })
if ($backslashes.Count -gt 0) { throw "Archive contains backslash separators: $($backslashes[0])" }
# @() matters: PowerShell unwraps a single-element pipeline result to a scalar
# string, and indexing a string would yield its first character instead.
$roots = @($entryNames | ForEach-Object { ($_ -split '/')[0] } | Select-Object -Unique)
if ($roots.Count -ne 1 -or $roots[0] -ne $packageName) {
    throw "Archive must have exactly one top-level folder named '$packageName' (found: $($roots -join ', '))"
}
$leaked = @($entryNames | Where-Object { $_ -match '\.(pth|task|onnx|pt|ckpt)$' })
if ($leaked.Count -gt 0) { throw "Model weights leaked into the archive: $($leaked[0])" }

$size = [Math]::Round((Get-Item $zipPath).Length / 1KB, 1)
Write-Host "files packaged  : $copied"
if ($skippedWeights -gt 0) { Write-Host "weights skipped : $skippedWeights (third-party weights are never redistributed)" }
Write-Host "entries verified: $($entryNames.Count) (forward slashes, single root '$packageName')"
Write-Host "output          : $zipPath ($size KB)"
Write-Host ''
Write-Host 'Install in Blender: Preferences > Add-ons > Install from Disk, pick this zip.'
exit 0
