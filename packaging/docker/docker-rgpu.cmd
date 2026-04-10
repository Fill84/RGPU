@echo off
REM RGPU Docker Wrapper — transparently adds RGPU GPU support to all --gpus containers
REM
REM Install: copy this to a directory in your PATH *before* Docker's directory
REM          or rename Docker's docker.exe and use this as docker.cmd
REM
REM When --gpus is detected, automatically adds:
REM   -e RGPU_IPC_ADDRESS=host.docker.internal:9877
REM   -v <lib>:/usr/local/nvidia/lib64/<name>:ro  (for each interpose lib)

setlocal EnableDelayedExpansion

REM Path to RGPU interpose libraries (adjust to your installation)
set "RGPU_LIBS=%~dp0..\..\tmp"

REM Check if --gpus is in the arguments
set "HAS_GPUS=0"
for %%a in (%*) do (
    if "%%a"=="--gpus" set "HAS_GPUS=1"
)

if "%HAS_GPUS%"=="0" (
    REM No --gpus flag, pass through to real Docker
    "C:\Program Files\Docker\Docker\resources\bin\docker.exe" %*
    exit /b %ERRORLEVEL%
)

REM Add RGPU interpose mounts
"C:\Program Files\Docker\Docker\resources\bin\docker.exe" %* ^
  -e RGPU_IPC_ADDRESS=host.docker.internal:9877 ^
  -v "%RGPU_LIBS%\librgpu_nvml_interpose.so:/usr/local/nvidia/lib64/libnvidia-ml.so.1:ro" ^
  -v "%RGPU_LIBS%\librgpu_cuda_interpose.so:/usr/local/nvidia/lib64/libcuda.so.1:ro" ^
  -v "%RGPU_LIBS%\librgpu_nvenc_interpose.so:/usr/local/nvidia/lib64/libnvidia-encode.so.1:ro" ^
  -v "%RGPU_LIBS%\librgpu_nvdec_interpose.so:/usr/local/nvidia/lib64/libnvcuvid.so.1:ro"

exit /b %ERRORLEVEL%
