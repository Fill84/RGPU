#!/bin/bash
# nvidia-smi-rgpu — Drop-in nvidia-smi replacement using RGPU NVML interpose
# Shows both local and remote GPUs transparently

set -euo pipefail

python3 -c "
import ctypes, sys, time, os

# Load NVML
try:
    ml = ctypes.CDLL('libnvidia-ml.so.1')
except OSError:
    print('NVIDIA-SMI has failed because it couldn\\'t communicate with the NVIDIA driver.')
    sys.exit(1)

ml.nvmlInit_v2()

# Driver info
driver_ver = ctypes.create_string_buffer(80)
ml.nvmlSystemGetDriverVersion(driver_ver, 80)

nvml_ver = ctypes.create_string_buffer(80)
ml.nvmlSystemGetNVMLVersion(nvml_ver, 80)

count = ctypes.c_uint(0)
ml.nvmlDeviceGetCount_v2(ctypes.byref(count))

now = time.strftime('%a %b %d %H:%M:%S %Y')

print(f'{now}       ')
print('+-----------------------------------------------------------------------------------------+')
print(f'| NVIDIA-SMI (RGPU)          Driver Version: {driver_ver.value.decode():<15s} CUDA Version: N/A       |')
print('+-----------------------------------------+------------------------+----------------------+')
print('| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |')
print('| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |')
print('|                                         |                        |               MIG M. |')
print('|=========================================+========================+======================|')

for i in range(count.value):
    handle = ctypes.c_void_p()
    ml.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(handle))

    name = ctypes.create_string_buffer(256)
    ml.nvmlDeviceGetName(handle, name, 256)
    gpu_name = name.value.decode()

    temp = ctypes.c_uint(0)
    ml.nvmlDeviceGetTemperature(handle, 0, ctypes.byref(temp))

    class mem_t(ctypes.Structure):
        _fields_ = [('total', ctypes.c_ulonglong),
                     ('free', ctypes.c_ulonglong),
                     ('used', ctypes.c_ulonglong)]
    mem = mem_t()
    ml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem))
    total_mb = mem.total // (1024*1024)
    used_mb = mem.used // (1024*1024)

    pci_bus_id = ctypes.create_string_buffer(32)
    # Try to get PCI bus ID from pci info struct (offset varies)
    bus_id = '00000000:00:00.0'

    print(f'|   {i}  {gpu_name:<25s}   Off  |   {bus_id} Off |                  N/A |')
    print(f'|  N/A   {temp.value}C    P8               N/A  |   {used_mb:>5d}MiB / {total_mb:>5d}MiB |      N/A      Default |')
    print('|                                         |                        |                  N/A |')
    print('+-----------------------------------------+------------------------+----------------------+')

print('')
print('+-----------------------------------------------------------------------------------------+')
print('| Processes:                                                                              |')
print('|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |')
print('|        ID   ID                                                               Usage      |')
print('|=========================================================================================|')
print('|  No running processes found                                                             |')
print('+-----------------------------------------------------------------------------------------+')

ml.nvmlShutdown()
" 2>&1
