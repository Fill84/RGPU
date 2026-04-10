# RGPU Docker Wrapper — transparently adds RGPU GPU support
#
# Usage: Set as alias in your PowerShell profile:
#   Set-Alias docker D:\Dev-Projects\RGPUv6\packaging\docker\Invoke-DockerRgpu.ps1
#
# Or run directly:
#   .\Invoke-DockerRgpu.ps1 run --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi

param([Parameter(ValueFromRemainingArguments)]$DockerArgs)

$RgpuLibs = $env:RGPU_LIB_PATH
if (-not $RgpuLibs) { $RgpuLibs = "D:\tmp" }

$docker = "C:\Program Files\Docker\Docker\resources\bin\docker.exe"

# Check if --gpus is in the arguments
$hasGpus = $DockerArgs -contains "--gpus"
$isRun = $DockerArgs.Count -gt 0 -and $DockerArgs[0] -eq "run"

if (-not $hasGpus -or -not $isRun) {
    # Pass through to real Docker
    & $docker @DockerArgs
    exit $LASTEXITCODE
}

# Find where to inject our flags (after 'run' and its flags, before image name)
# Strategy: inject right after --gpus and its value
$newArgs = @()
$i = 0
$injected = $false

while ($i -lt $DockerArgs.Count) {
    $arg = $DockerArgs[$i]
    $newArgs += $arg

    # After adding --gpus and its value, inject our flags
    if ($arg -eq "--gpus" -and -not $injected) {
        $i++
        if ($i -lt $DockerArgs.Count) {
            $newArgs += $DockerArgs[$i]  # the gpus value (e.g. "all")
        }
        # Inject RGPU mounts
        $newArgs += "-e"
        $newArgs += "RGPU_IPC_ADDRESS=host.docker.internal:9877"
        $newArgs += "-v"
        $newArgs += "${RgpuLibs}/librgpu_nvml_interpose.so:/usr/local/nvidia/lib64/libnvidia-ml.so.1:ro"
        $newArgs += "-v"
        $newArgs += "${RgpuLibs}/librgpu_cuda_interpose.so:/usr/local/nvidia/lib64/libcuda.so.1:ro"
        $newArgs += "-v"
        $newArgs += "${RgpuLibs}/librgpu_nvenc_interpose.so:/usr/local/nvidia/lib64/libnvidia-encode.so.1:ro"
        $newArgs += "-v"
        $newArgs += "${RgpuLibs}/librgpu_nvdec_interpose.so:/usr/local/nvidia/lib64/libnvcuvid.so.1:ro"
        $injected = $true
    }
    $i++
}

& $docker @newArgs
exit $LASTEXITCODE
