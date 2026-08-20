<#
.SYNOPSIS
    Link this repository into Blender's user add-on directory for development.

.DESCRIPTION
    The repository root *is* the add-on package, so a directory junction named
    "motion_capture" inside <user scripts>/addons is all Blender needs.
    Junctions do not require administrator rights on Windows.

.EXAMPLE
    .\tools\dev_install.ps1
    .\tools\dev_install.ps1 -BlenderVersion 4.0
    .\tools\dev_install.ps1 -Remove
#>
[CmdletBinding()]
param(
    [string[]] $BlenderVersion = @('4.0', '4.5'),
    [switch] $Remove
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$packageName = 'motion_capture'

if ((Split-Path -Leaf $repoRoot) -ne $packageName) {
    throw "Repository root must be named '$packageName' (found '$(Split-Path -Leaf $repoRoot)') because it is the add-on package."
}

foreach ($version in $BlenderVersion) {
    $addonsDir = Join-Path $env:APPDATA "Blender Foundation\Blender\$version\scripts\addons"
    $linkPath = Join-Path $addonsDir $packageName

    if ($Remove) {
        if (Test-Path $linkPath) {
            $item = Get-Item $linkPath -Force
            if ($item.LinkType) {
                $item.Delete()
                Write-Host "removed junction: $linkPath"
            }
            else {
                Write-Warning "not a junction, refusing to delete: $linkPath"
            }
        }
        else {
            Write-Host "nothing to remove for Blender $version"
        }
        continue
    }

    if (-not (Test-Path $addonsDir)) {
        New-Item -ItemType Directory -Force -Path $addonsDir | Out-Null
        Write-Host "created: $addonsDir"
    }

    if (Test-Path $linkPath) {
        $item = Get-Item $linkPath -Force
        if ($item.LinkType) {
            $item.Delete()
        }
        else {
            throw "A real directory already exists at $linkPath. Move it away first."
        }
    }

    New-Item -ItemType Junction -Path $linkPath -Target $repoRoot | Out-Null
    Write-Host "linked Blender $version : $linkPath -> $repoRoot"
}

if (-not $Remove) {
    Write-Host ''
    Write-Host 'Next: enable "Motion Capture for Rigify" in Blender Preferences > Add-ons.'
}
