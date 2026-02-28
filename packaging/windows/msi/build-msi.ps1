# RGPU Windows MSI Installer Build Script (WiX v4)
#
# Usage:
#   .\build-msi.ps1
#   .\build-msi.ps1 -Version 0.2.0
#   .\build-msi.ps1 -SkipBuild
#
# Prerequisites:
#   - Rust toolchain (rustup)
#   - .NET 8.0+ SDK
#   - WiX v4 CLI: dotnet tool install --global wix
#   - WiX extensions: wix extension add WixToolset.UI.wixext WixToolset.Util.wixext

param(
    [string]$Version = "0.1.0",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $PSCommandPath
$ProjectRoot = (Resolve-Path "$ScriptDir\..\..\..").Path

Write-Host "=== RGPU Windows MSI Build (WiX v4) ===" -ForegroundColor Cyan
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
    "$ProjectRoot\target\release\rgpu_nvenc_interpose.dll",
    "$ProjectRoot\target\release\rgpu_nvdec_interpose.dll",
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

# Step 3: Stage files
Write-Host "[3/4] Staging files..." -ForegroundColor Yellow
$StagingDir = "$ScriptDir\staging"
if (Test-Path $StagingDir) {
    Remove-Item -Recurse -Force $StagingDir
}
New-Item -ItemType Directory -Force -Path $StagingDir | Out-Null

Copy-Item "$ProjectRoot\target\release\rgpu.exe" "$StagingDir\"
Copy-Item "$ProjectRoot\target\release\rgpu_cuda_interpose.dll" "$StagingDir\"
Copy-Item "$ProjectRoot\target\release\rgpu_nvenc_interpose.dll" "$StagingDir\"
Copy-Item "$ProjectRoot\target\release\rgpu_nvdec_interpose.dll" "$StagingDir\"
Copy-Item "$ProjectRoot\target\release\rgpu_vk_icd.dll" "$StagingDir\"
Copy-Item "$ProjectRoot\packaging\config\rgpu.toml.template" "$StagingDir\"
Copy-Item "$ProjectRoot\icon.ico" "$StagingDir\"

Write-Host "  Staged to: $StagingDir" -ForegroundColor Green

# Step 4: Build MSI
Write-Host "[4/4] Building MSI installer..." -ForegroundColor Yellow

$WixPath = (Get-Command wix -ErrorAction SilentlyContinue).Source
if (-not $WixPath) {
    Write-Host "  ERROR: WiX CLI not found." -ForegroundColor Red
    Write-Host "  Install with: dotnet tool install --global wix" -ForegroundColor Red
    Write-Host "  Then run:     wix extension add WixToolset.UI.wixext WixToolset.Util.wixext" -ForegroundColor Red
    exit 1
}

Write-Host "  Using WiX: $WixPath" -ForegroundColor Gray

$OutputMsi = "$ScriptDir\rgpu-${Version}-windows-x64.msi"

wix build `
    -d VERSION=$Version `
    -arch x64 `
    -ext WixToolset.UI.wixext `
    -ext WixToolset.Util.wixext `
    -o $OutputMsi `
    "$ScriptDir\rgpu.wxs"

if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: WiX build failed" -ForegroundColor Red
    exit 1
}

# Verify output
if (Test-Path $OutputMsi) {
    $size = (Get-Item $OutputMsi).Length / 1MB
    Write-Host ""
    Write-Host "=== Build Complete ===" -ForegroundColor Cyan
    Write-Host ("  MSI Installer: {0} ({1:N1} MB)" -f $OutputMsi, $size) -ForegroundColor Green
} else {
    Write-Host "  WARNING: Expected output not found at $OutputMsi" -ForegroundColor Yellow
}
