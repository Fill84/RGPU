# RGPU Windows Installer Build Script
#
# Usage:
#   .\build-windows.ps1
#   .\build-windows.ps1 -Version 0.2.0
#   .\build-windows.ps1 -SkipBuild
#
# Prerequisites:
#   - Rust toolchain (rustup)
#   - NSIS 3.x (winget install NSIS.NSIS)
#   - EnvVarUpdate.nsh plugin in NSIS Include directory

param(
    [string]$Version = "0.1.0",
    [string]$NsisPath = "",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

# Find project root (two levels up from this script)
$ScriptDir = Split-Path -Parent $PSCommandPath
$ProjectRoot = (Resolve-Path "$ScriptDir\..\..").Path

Write-Host "=== RGPU Windows Installer Build ===" -ForegroundColor Cyan
Write-Host "Version: $Version"
Write-Host "Project: $ProjectRoot"
Write-Host ""

# Step 1: Build release (unless skipped)
if (-not $SkipBuild) {
    Write-Host "[1/4] Building release..." -ForegroundColor Yellow
    Push-Location $ProjectRoot
    cargo build --release
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: cargo build failed" -ForegroundColor Red
        Pop-Location
        exit 1
    }
    Pop-Location
    Write-Host "  Build successful." -ForegroundColor Green
} else {
    Write-Host "[1/4] Skipping build (--SkipBuild)" -ForegroundColor Gray
}

# Step 2: Verify artifacts
Write-Host "[2/4] Verifying build artifacts..." -ForegroundColor Yellow
$Artifacts = @(
    "$ProjectRoot\target\release\rgpu.exe",
    "$ProjectRoot\target\release\rgpu_cuda_interpose.dll",
    "$ProjectRoot\target\release\rgpu_vk_icd.dll"
)

foreach ($artifact in $Artifacts) {
    if (-not (Test-Path $artifact)) {
        Write-Host "  ERROR: Missing $artifact" -ForegroundColor Red
        Write-Host "  Run 'cargo build --release' first." -ForegroundColor Red
        exit 1
    }
    $size = (Get-Item $artifact).Length / 1MB
    Write-Host ("  Found: {0} ({1:N1} MB)" -f (Split-Path -Leaf $artifact), $size) -ForegroundColor Green
}

# Step 3: Stage files for NSIS
Write-Host "[3/4] Staging files..." -ForegroundColor Yellow
$StagingDir = "$ScriptDir\nsis\staging"
if (Test-Path $StagingDir) {
    Remove-Item -Recurse -Force $StagingDir
}
New-Item -ItemType Directory -Force -Path $StagingDir | Out-Null

Copy-Item "$ProjectRoot\target\release\rgpu.exe" "$StagingDir\"
Copy-Item "$ProjectRoot\target\release\rgpu_cuda_interpose.dll" "$StagingDir\"
Copy-Item "$ProjectRoot\target\release\rgpu_vk_icd.dll" "$StagingDir\"
Copy-Item "$ProjectRoot\packaging\config\rgpu.toml.template" "$StagingDir\"
Copy-Item "$ProjectRoot\icon.ico" "$StagingDir\"

Write-Host "  Staged to: $StagingDir" -ForegroundColor Green

# Step 4: Find and run NSIS
Write-Host "[4/4] Building NSIS installer..." -ForegroundColor Yellow

if ($NsisPath -eq "") {
    # Try common locations
    $NsisCandidates = @(
        "C:\Program Files (x86)\NSIS\makensis.exe",
        "C:\Program Files\NSIS\makensis.exe",
        "${env:ProgramFiles(x86)}\NSIS\makensis.exe",
        "${env:ProgramFiles}\NSIS\makensis.exe"
    )
    foreach ($candidate in $NsisCandidates) {
        if (Test-Path $candidate) {
            $NsisPath = $candidate
            break
        }
    }
    # Try PATH
    if ($NsisPath -eq "") {
        $NsisPath = (Get-Command makensis -ErrorAction SilentlyContinue).Source
    }
}

if ($NsisPath -eq "" -or -not (Test-Path $NsisPath)) {
    Write-Host "  ERROR: NSIS not found. Install with: winget install NSIS.NSIS" -ForegroundColor Red
    Write-Host "  Or specify path: .\build-windows.ps1 -NsisPath 'C:\path\to\makensis.exe'" -ForegroundColor Red
    exit 1
}

Write-Host "  Using NSIS: $NsisPath" -ForegroundColor Gray

& $NsisPath /DVERSION=$Version "$ScriptDir\nsis\rgpu-installer.nsi"
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: NSIS build failed" -ForegroundColor Red
    exit 1
}

# Verify output
$InstallerPath = "$ScriptDir\nsis\rgpu-${Version}-windows-x64-setup.exe"
if (Test-Path $InstallerPath) {
    $size = (Get-Item $InstallerPath).Length / 1MB
    Write-Host ""
    Write-Host "=== Build Complete ===" -ForegroundColor Cyan
    Write-Host ("  Installer: {0} ({1:N1} MB)" -f $InstallerPath, $size) -ForegroundColor Green
} else {
    Write-Host "  WARNING: Expected output not found at $InstallerPath" -ForegroundColor Yellow
    Write-Host "  Check NSIS output directory." -ForegroundColor Yellow
}
