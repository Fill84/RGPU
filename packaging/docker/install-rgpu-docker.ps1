# RGPU Docker Integration Installer for Windows
#
# Installs RGPU interpose libraries into Docker Desktop so that
# every container started with --gpus all sees both local and remote GPUs.
#
# Requirements:
# - Docker Desktop running with WSL2 backend
# - RGPU client daemon running with ipc_listen_address configured
#
# Usage:
#   .\install-rgpu-docker.ps1 -LibPath D:\tmp
#   .\install-rgpu-docker.ps1 -Uninstall

param(
    [string]$LibPath = "",
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"

$wslDistro = "docker-desktop"
$targetDir = "/usr/local/nvidia/lib64"

# Map of interpose lib → system lib name
$libs = @{
    "librgpu_nvml_interpose.so"  = "libnvidia-ml.so.1"
    "librgpu_cuda_interpose.so"  = "libcuda.so.1"
    "librgpu_nvenc_interpose.so" = "libnvidia-encode.so.1"
    "librgpu_nvdec_interpose.so" = "libnvcuvid.so.1"
}

if ($Uninstall) {
    Write-Host "Uninstalling RGPU Docker integration..."
    foreach ($sysName in $libs.Values) {
        wsl.exe -d $wslDistro -- rm -f "$targetDir/$sysName" 2>$null
    }
    Write-Host "Done. Restart Docker Desktop to restore original NVIDIA libraries."
    exit 0
}

if (-not $LibPath) {
    Write-Host "Usage: .\install-rgpu-docker.ps1 -LibPath <path-to-interpose-libs>"
    Write-Host ""
    Write-Host "The path should contain: librgpu_cuda_interpose.so, librgpu_nvml_interpose.so, etc."
    exit 1
}

# Verify libs exist
foreach ($lib in $libs.Keys) {
    $path = Join-Path $LibPath $lib
    if (-not (Test-Path $path)) {
        Write-Error "Missing: $path"
        exit 1
    }
}

# Check Docker Desktop is running
$dd = Get-Process "Docker Desktop" -ErrorAction SilentlyContinue
if (-not $dd) {
    Write-Error "Docker Desktop is not running"
    exit 1
}

Write-Host "Installing RGPU interpose libraries into Docker Desktop WSL2 VM..."

# Create target directory
wsl.exe -d $wslDistro -- mkdir -p $targetDir

# Convert Windows path to WSL mount path
$wslLibPath = $LibPath -replace '\\','/' -replace '^([A-Za-z]):','/mnt/host/$1'
$wslLibPath = $wslLibPath.ToLower() -replace '/mnt/host/([a-z])','/mnt/host/$1'

foreach ($entry in $libs.GetEnumerator()) {
    $src = "$wslLibPath/$($entry.Key)"
    $dst = "$targetDir/$($entry.Value)"
    Write-Host "  $($entry.Key) -> $($entry.Value)"
    wsl.exe -d $wslDistro -- cp "$src" "$dst"
}

Write-Host ""
Write-Host "Installed! Remote GPUs will be visible in Docker containers."
Write-Host ""
Write-Host "NOTE: You still need to pass -e RGPU_IPC_ADDRESS=host.docker.internal:9877"
Write-Host "      when running containers, OR set it in your Docker Compose file."
Write-Host ""
Write-Host "Example:"
Write-Host "  docker run --gpus all -e RGPU_IPC_ADDRESS=host.docker.internal:9877 nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi"
