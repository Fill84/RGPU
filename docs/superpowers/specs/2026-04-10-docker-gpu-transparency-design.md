# Docker GPU Transparency — Design Spec

## Goal

Make remote RGPU GPUs completely invisible from local GPUs. Docker's `--gpus all`, `nvidia-smi`, CUDA applications, FFmpeg, and all other GPU consumers must see remote GPUs as if they are physically installed hardware. No configuration, no LD_PRELOAD, no special container flags — just `docker run --gpus all` and it works.

## Requirements

- Works on both native Linux and WSL2 (Docker Desktop)
- Remote GPU is indistinguishable from a local GPU (`nvidia-smi` shows "RTX 3070", not "RTX 3070 (Remote - RGPU)")
- Full functionality: CUDA compute, NVENC/NVDEC encoding, Vulkan rendering, NVML monitoring
- One-time admin installation (kernel module on Linux, signed driver on Windows), then automatic
- Docker `--gpus all` includes remote GPUs without any extra config

## Architecture

```
Docker: docker run --gpus all myapp
    │
    ▼
nvidia-container-toolkit (runs on HOST)
    │
    ├── Calls nvmlDeviceGetCount()
    │   └── HOST libnvidia-ml.so.1 (our NVML interpose)
    │       └── Returns local_count + remote_count
    │
    ├── For each GPU: nvmlDeviceGetMinorNumber()
    │   └── Local GPU → minor 0 (real /dev/nvidia0)
    │   └── Remote GPU → minor 1 (fake /dev/nvidia1 from our kernel module)
    │
    ├── Mounts into container:
    │   ├── /dev/nvidia0 (real device node)
    │   ├── /dev/nvidia1 (our kernel module's fake device node)
    │   ├── /dev/nvidiactl (real)
    │   ├── /dev/nvidia-uvm (real)
    │   ├── libcuda.so.1 (our CUDA interpose — IS the host lib)
    │   ├── libnvidia-ml.so.1 (our NVML interpose — IS the host lib)
    │   ├── libnvidia-encode.so.1 (our NVENC interpose — IS the host lib)
    │   └── libnvcuvid.so.1 (our NVDEC interpose — IS the host lib)
    │
    ▼
Container: sees 2 GPUs, everything transparent
    ├── cuDeviceGetCount() → 2
    ├── nvmlDeviceGetCount() → 2
    ├── nvidia-smi → shows both GPUs
    └── ffmpeg -c:v h264_nvenc → can target either GPU
```

## Component 1: Kernel Module (`rgpu_vgpu.c`)

### Current State

The existing kernel module creates `/dev/rgpu_gpu{N}` device nodes as misc devices. Docker doesn't recognize these because `nvidia-container-toolkit` looks for `/dev/nvidia{N}`.

### Required Changes

Modify the kernel module to create device nodes named `/dev/nvidia{N}` using NVIDIA's major number (195).

**Device node naming:**
- Query the highest existing real NVIDIA minor number on the system
- Assign remote GPUs the next sequential minor numbers
- Example: 1 local GPU at minor 0 → first remote GPU gets minor 1 → `/dev/nvidia1`

**Device node behavior:**
- `open()` → returns success (fd valid)
- `close()` → returns success
- `ioctl()` → returns -ENODEV or forwards to userspace (apps never reach this because interpose libs intercept first)
- `mmap()` → returns -ENODEV

**IOCTL interface to daemon (via `/dev/rgpu_control`):**
- `RGPU_IOCTL_ADD_GPU` — extended with `minor_number` field
- `RGPU_IOCTL_REMOVE_GPU` — unchanged
- `RGPU_IOCTL_LIST_GPUS` — extended with `minor_number` in response

**udev rules:**
```
KERNEL=="nvidia[0-9]*", SUBSYSTEM=="nvidia", MODE="0666", GROUP="video"
```
Our fake device nodes must match the same udev pattern as real NVIDIA devices.

**Registration approach:**
- Use `register_chrdev_region()` with major 195 and the specific minor number(s)
- Use `cdev_add()` to register the character device
- Use `device_create()` with the nvidia class to create `/dev/nvidia{N}`

**Safety:**
- Only register minor numbers that are NOT already taken by the real NVIDIA driver
- On module unload, all fake device nodes are removed
- If the real NVIDIA driver is not loaded, skip (no major 195 available)

### WSL2 Behavior

On WSL2, there are no `/dev/nvidia*` device files. GPU access goes through `/dev/dxg`. The kernel module is not needed on WSL2. Docker Desktop's nvidia-container-toolkit on WSL2 uses a different discovery path that relies more on NVML enumeration and less on device nodes.

For WSL2, the NVML interpose on the host (inside WSL2) is sufficient. The `nvidia-container-toolkit` will see the extra GPUs via NVML and pass the device information to the container.

## Component 2: NVML Interpose — Missing Functions

### Current State

The NVML interpose already merges local + remote GPU counts and handles device enumeration. However, several functions return `NVML_ERROR_NOT_SUPPORTED` for remote GPUs. The `nvidia-container-toolkit` and `nvidia-smi` call these functions and bail out if they fail.

### Functions to Implement for Remote GPUs

**Critical (nvidia-container-toolkit needs these):**

| Function | What to return for remote GPU |
|----------|-------------------------------|
| `nvmlDeviceGetMinorNumber(device, &minor)` | The minor number of our fake `/dev/nvidia{N}` device node |
| `nvmlDeviceGetUUID(device, uuid, len)` | A deterministic UUID: `GPU-RGPU-{server_id}-{device_index}-{hash}` formatted as standard GPU UUID |
| `nvmlDeviceGetPciInfo(device, &pci)` | Fake PCI info with unique bus/device/function, domain=0xRGPU |
| `nvmlDeviceGetIndex(device, &index)` | The pool index (local_count + remote_index) |

**Important (nvidia-smi and monitoring tools use these):**

| Function | What to return for remote GPU |
|----------|-------------------------------|
| `nvmlDeviceGetName(device, name, len)` | Raw device name without "(Remote - RGPU)" suffix |
| `nvmlDeviceGetMemoryInfo(device, &mem)` | Total/free/used from server query |
| `nvmlDeviceGetUtilizationRates(device, &util)` | GPU/memory utilization from server query |
| `nvmlDeviceGetTemperature(device, type, &temp)` | Temperature from server query |
| `nvmlDeviceGetPowerUsage(device, &power)` | Power from server query, or 0 if unavailable |
| `nvmlDeviceGetDriverVersion(device, ver, len)` | Server's driver version string |
| `nvmlDeviceGetCudaComputeCapability(device, &major, &minor)` | From GpuInfo.cuda_compute_capability |
| `nvmlDeviceGetComputeRunningProcesses(device, &count, infos)` | Empty list (no local processes on remote GPU) |

**UUID format:**
Must match NVIDIA's format: `GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`
Generate deterministically from server_id + device_index so it's stable across restarts.

**PCI info:**
Use realistic-looking PCI addresses to avoid tools rejecting them. Put remote GPUs on high bus numbers that are unlikely to conflict with real hardware:
```
domain:   0x0000
bus:      0x80 + server_id   (128+, real hardware rarely uses bus > 64)
device:   device_index
function: 0
busId:    "00000000:8S:0D.0"  (S=server_id hex, D=device_index hex)
```
Example: server_id=0, device_index=0 → `"00000000:80:00.0"`

## Component 3: Device Manager Updates

### Current State

`device_manager.rs` calls the kernel module via ioctl to create virtual devices. It filters for remote GPUs only (server_id != LOCAL_SERVER_ID).

### Required Changes

**Before creating virtual devices:**
1. Query the real NVIDIA driver for the highest used minor number
   - Read `/proc/driver/nvidia/gpus/*/information` or call real NVML
   - Or scan `/dev/nvidia*` for the highest existing minor
2. Assign sequential minor numbers starting from `highest_real + 1`

**Pass minor number to kernel module:**
- Extend the `RgpuGpuInfo` ioctl struct with a `minor_number` field
- The kernel module uses this to create `/dev/nvidia{minor_number}`

**Store mapping:**
- Keep a `HashMap<NetworkHandle, u32>` of remote GPU → assigned minor number
- Make this available to the NVML interpose (via shared memory or IPC query)

**Communicate minor numbers to NVML interpose:**
- Option A: Write to a well-known file (`/run/rgpu/gpu_map.json`) that the NVML interpose reads
- Option B: Expose via IPC query (daemon already has IPC)
- Recommendation: Option A — simple, no IPC round-trip for every NVML call

**GPU map file format (`/run/rgpu/gpu_map.json`):**
```json
{
  "gpus": [
    {
      "pool_index": 1,
      "minor_number": 1,
      "server_id": 0,
      "device_index": 0,
      "device_name": "NVIDIA GeForce RTX 3070",
      "uuid": "GPU-52475055-0000-0000-0000-000000000001",
      "pci_bus_id": "5247:0000:00.0",
      "total_memory": 8589934592
    }
  ]
}
```

## Component 4: NVML Interpose — Device Name

### Current State

Remote GPUs are displayed as "NVIDIA GeForce RTX 3070 (Remote - RGPU)". For full transparency, this suffix must be removed.

### Change

`nvmlDeviceGetName()` for remote handles: return the raw `device_name` from GpuInfo, without any suffix.

This is a one-line change in `nvml_interpose/lib.rs`.

## Component 5: Installer Updates

### Linux (deb/rpm)

The interpose libraries must be installed as the system NVIDIA libraries:

```
/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1     → our NVML interpose
/usr/lib/x86_64-linux-gnu/libnvidia-ml_real.so.1 → original NVIDIA NVML (renamed)
/usr/lib/x86_64-linux-gnu/libcuda.so.1           → our CUDA interpose
/usr/lib/x86_64-linux-gnu/libcuda_real.so.1      → original NVIDIA CUDA (renamed)
```

This is already implemented in the existing deb/rpm packaging.

The kernel module must be installed via DKMS (already configured in `drivers/linux/rgpu-vgpu/dkms.conf`). The module name stays `rgpu_vgpu` but creates NVIDIA-named device nodes.

### Windows (NSIS)

Already installs interpose DLLs to System32. No changes needed for Windows — Docker on Windows uses WSL2 which is Linux.

## Component 6: WSL2 Specific Handling

On WSL2, the approach is different because there are no `/dev/nvidia*` device files:

1. **No kernel module needed** — WSL2 GPU access goes through `/dev/dxg`
2. **NVML interpose on WSL2 host** — installed as `libnvidia-ml.so.1` in the WSL2 filesystem
3. **Docker Desktop** mounts WSL2's NVIDIA libraries into containers
4. **Detection**: Check for WSL2 via `/proc/sys/fs/binfmt_misc/WSLInterop` or `WSL_DISTRO_NAME` env var
5. **Skip kernel module** when running on WSL2, rely purely on NVML interpose

The daemon and interpose libs work the same way — the only difference is no fake device nodes needed.

## What Does NOT Change

- **CUDA interpose** — already works, daemon returns merged pool
- **NVENC interpose** — already works
- **NVDEC interpose** — already works
- **Vulkan ICD** — already works (registered via ICD JSON)
- **Server** — no changes
- **Protocol** — no changes
- **Transport** — no changes
- **Client daemon core** — no changes (pool manager already merges)

## Success Criteria

1. `nvidia-smi` on the host shows both local and remote GPUs with no "RGPU" branding
2. `docker run --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi` shows both GPUs
3. `docker run --gpus all` container can run CUDA on the remote GPU
4. `docker run --gpus all` container can run FFmpeg h264_nvenc on the remote GPU
5. No special Docker flags, env vars, or config needed — just `--gpus all`

## Risk: NVIDIA Driver Updates

When NVIDIA updates their driver, the real `libnvidia-ml.so.1` gets overwritten. Our interpose must be re-installed after driver updates. The DKMS hook can trigger this automatically on Linux. On Windows/WSL2, the user would need to re-run the installer.

Mitigation: Install a DKMS post-install hook or systemd path unit that watches for NVIDIA driver changes and re-symlinks our interpose.
