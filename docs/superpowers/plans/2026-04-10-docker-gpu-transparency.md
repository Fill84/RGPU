# Docker GPU Transparency — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make remote RGPU GPUs fully transparent to Docker, nvidia-smi, and all applications — `docker run --gpus all` shows remote GPUs as if they are local hardware.

**Architecture:** Modify the existing Linux kernel module to create `/dev/nvidia{N}` device nodes (major 195) instead of `/dev/rgpu_gpu{N}`. Extend the device manager to assign minor numbers based on existing real NVIDIA devices. Update the NVML interpose to return realistic PCI info, UUIDs, and device names without the "(Remote - RGPU)" suffix. Write a GPU map file so the NVML interpose knows which minor numbers were assigned.

**Tech Stack:** Linux kernel module (C), Rust (NVML interpose + device manager), Docker + nvidia-container-toolkit

**Spec:** `docs/superpowers/specs/2026-04-10-docker-gpu-transparency-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `drivers/linux/rgpu-vgpu/rgpu_vgpu.c` | Modify | Create `/dev/nvidia{N}` with major 195 instead of misc devices |
| `drivers/linux/rgpu-vgpu/99-rgpu.rules` | Modify | Match `nvidia[0-9]*` device pattern |
| `crates/rgpu-client/src/device_manager.rs` | Modify | Scan real NVIDIA minors, assign next available, pass to kernel, write GPU map file |
| `crates/rgpu-nvml-interpose/src/lib.rs` | Modify | Read GPU map file, fix device name, fix PCI info, fix minor number |
| `tests/docker/test_docker_gpu_visibility.sh` | Create | Test script that checks nvidia-smi and Docker GPU visibility |

---

### Task 1: Kernel Module — Switch from misc devices to NVIDIA character devices

**Files:**
- Modify: `drivers/linux/rgpu-vgpu/rgpu_vgpu.c`

The kernel module currently creates `/dev/rgpu_gpu{N}` as misc devices. We need to create `/dev/nvidia{N}` using NVIDIA's major number (195) with specific minor numbers passed from userspace.

- [ ] **Step 1: Update the ioctl struct to include minor_number**

In `drivers/linux/rgpu-vgpu/rgpu_vgpu.c`, replace the `rgpu_gpu_info` struct (lines 38-42):

```c
struct rgpu_gpu_info {
    char name[128];
    __u64 total_memory;
    __u32 index;        /* assigned by kernel on ADD, passed on REMOVE */
    __u32 minor_number; /* nvidia minor number to create (e.g. 1 → /dev/nvidia1) */
};
```

- [ ] **Step 2: Update the per-GPU state struct**

Replace the `rgpu_vgpu` struct (lines 56-64):

```c
struct rgpu_vgpu {
    bool active;
    char name[128];
    __u64 total_memory;
    __u32 index;
    __u32 nvidia_minor;       /* the /dev/nvidia{N} minor we created */
    struct platform_device *pdev;
    struct cdev cdev;         /* character device (replaces miscdevice) */
    struct device *dev;       /* sysfs device node */
    dev_t devno;              /* major:minor combo */
};
```

- [ ] **Step 3: Add NVIDIA major number constant and class**

Add after the module_param declarations (around line 34):

```c
#define NVIDIA_MAJOR 195

static struct class *nvidia_class;
```

- [ ] **Step 4: Rewrite rgpu_add_gpu to create /dev/nvidia{N}**

Replace the `rgpu_add_gpu` function (lines 107-186) with:

```c
static long rgpu_add_gpu(unsigned long arg)
{
    struct rgpu_gpu_info info;
    struct rgpu_vgpu *gpu;
    int i, slot = -1;
    dev_t devno;
    int ret;

    if (copy_from_user(&info, (void __user *)arg, sizeof(info)))
        return -EFAULT;

    /* Find a free slot */
    for (i = 0; i < max_gpus; i++) {
        if (!gpus[i].active) {
            slot = i;
            break;
        }
    }
    if (slot < 0)
        return -ENOSPC;

    gpu = &gpus[slot];
    memcpy(gpu->name, info.name, sizeof(gpu->name));
    gpu->name[sizeof(gpu->name) - 1] = '\0';
    gpu->total_memory = info.total_memory;
    gpu->nvidia_minor = info.minor_number;
    gpu->index = slot;

    /* Register character device with NVIDIA major + requested minor */
    devno = MKDEV(NVIDIA_MAJOR, gpu->nvidia_minor);
    gpu->devno = devno;

    cdev_init(&gpu->cdev, &rgpu_gpu_fops);
    gpu->cdev.owner = THIS_MODULE;

    ret = cdev_add(&gpu->cdev, devno, 1);
    if (ret) {
        pr_err("rgpu: cdev_add failed for nvidia%u: %d\n", gpu->nvidia_minor, ret);
        return ret;
    }

    /* Create /dev/nvidia{N} device node */
    gpu->dev = device_create(nvidia_class, NULL, devno, NULL,
                             "nvidia%u", gpu->nvidia_minor);
    if (IS_ERR(gpu->dev)) {
        ret = PTR_ERR(gpu->dev);
        pr_err("rgpu: device_create failed for nvidia%u: %d\n", gpu->nvidia_minor, ret);
        cdev_del(&gpu->cdev);
        return ret;
    }

    /* Create platform device for sysfs */
    gpu->pdev = platform_device_alloc("rgpu_vgpu", slot);
    if (gpu->pdev) {
        platform_device_add(gpu->pdev);
    }

    gpu->active = true;
    info.index = slot;

    if (copy_to_user((void __user *)arg, &info, sizeof(info)))
        pr_warn("rgpu: failed to copy index back to user\n");

    pr_info("rgpu: added virtual GPU %u as /dev/nvidia%u: %s (%llu MB)\n",
            slot, gpu->nvidia_minor, gpu->name,
            gpu->total_memory / (1024 * 1024));
    return 0;
}
```

- [ ] **Step 5: Rewrite rgpu_remove_gpu for character device cleanup**

Replace the `rgpu_remove_gpu` function (lines 190-218):

```c
static long rgpu_remove_gpu(unsigned long arg)
{
    struct rgpu_gpu_info info;
    struct rgpu_vgpu *gpu;

    if (copy_from_user(&info, (void __user *)arg, sizeof(info)))
        return -EFAULT;

    if (info.index >= max_gpus || !gpus[info.index].active)
        return -EINVAL;

    gpu = &gpus[info.index];

    device_destroy(nvidia_class, gpu->devno);
    cdev_del(&gpu->cdev);

    if (gpu->pdev) {
        platform_device_unregister(gpu->pdev);
        gpu->pdev = NULL;
    }

    pr_info("rgpu: removed virtual GPU %u (/dev/nvidia%u)\n",
            gpu->index, gpu->nvidia_minor);

    gpu->active = false;
    return 0;
}
```

- [ ] **Step 6: Update module init to create nvidia class**

Replace the module init function:

```c
static int __init rgpu_vgpu_init(void)
{
    int ret;

    gpus = kcalloc(max_gpus, sizeof(*gpus), GFP_KERNEL);
    if (!gpus)
        return -ENOMEM;

    /* Get or create the nvidia device class.
     * If the real NVIDIA driver is loaded, class_create will share the name.
     * We use our own class instance to avoid conflicts. */
    nvidia_class = class_create("nvidia");
    if (IS_ERR(nvidia_class)) {
        /* If class already exists (real NVIDIA driver loaded), look it up */
        nvidia_class = class_create("rgpu_nvidia");
        if (IS_ERR(nvidia_class)) {
            kfree(gpus);
            return PTR_ERR(nvidia_class);
        }
    }

    /* Register the control device */
    ret = misc_register(&control_dev);
    if (ret) {
        class_destroy(nvidia_class);
        kfree(gpus);
        return ret;
    }

    pr_info("rgpu: virtual GPU driver loaded (max %d devices)\n", max_gpus);
    return 0;
}
```

- [ ] **Step 7: Update module exit to destroy nvidia class**

Replace the module exit function:

```c
static void __exit rgpu_vgpu_exit(void)
{
    int i;

    for (i = 0; i < max_gpus; i++) {
        if (gpus[i].active) {
            device_destroy(nvidia_class, gpus[i].devno);
            cdev_del(&gpus[i].cdev);
            if (gpus[i].pdev)
                platform_device_unregister(gpus[i].pdev);
        }
    }

    misc_deregister(&control_dev);
    class_destroy(nvidia_class);
    kfree(gpus);
    pr_info("rgpu: virtual GPU driver unloaded\n");
}
```

- [ ] **Step 8: Add required includes**

Add at the top of the file (after existing includes):

```c
#include <linux/cdev.h>
```

- [ ] **Step 9: Commit**

```bash
git add drivers/linux/rgpu-vgpu/rgpu_vgpu.c
git commit -m "feat: kernel module creates /dev/nvidia{N} with major 195 for remote GPUs"
```

---

### Task 2: Update udev rules

**Files:**
- Modify: `drivers/linux/rgpu-vgpu/99-rgpu.rules`

- [ ] **Step 1: Update udev rules to match nvidia device nodes**

Replace the contents of `drivers/linux/rgpu-vgpu/99-rgpu.rules`:

```udev
# RGPU Virtual GPU udev rules
#
# Install: sudo cp 99-rgpu.rules /etc/udev/rules.d/
# Reload:  sudo udevadm control --reload-rules && sudo udevadm trigger

# /dev/rgpu_control — management interface (video group only)
KERNEL=="rgpu_control", SUBSYSTEM=="misc", MODE="0660", GROUP="video"

# /dev/nvidia* created by rgpu_vgpu module — match same permissions as real NVIDIA devices
KERNEL=="nvidia[0-9]*", DRIVERS=="rgpu_vgpu", MODE="0666", GROUP="video"
```

- [ ] **Step 2: Commit**

```bash
git add drivers/linux/rgpu-vgpu/99-rgpu.rules
git commit -m "feat: udev rules for RGPU-created /dev/nvidia* device nodes"
```

---

### Task 3: Device Manager — Scan real NVIDIA minors and assign next available

**Files:**
- Modify: `crates/rgpu-client/src/device_manager.rs`

The device manager must scan `/dev/nvidia*` to find the highest existing minor, then assign the next available minor to remote GPUs.

- [ ] **Step 1: Add minor number scanning function (Linux section)**

Add in the Linux platform section (after line 478):

```rust
/// Scan /dev/nvidia* to find the highest existing NVIDIA minor number.
/// Returns 0 if no NVIDIA devices exist.
fn find_highest_nvidia_minor() -> u32 {
    let mut highest: u32 = 0;
    if let Ok(entries) = std::fs::read_dir("/dev") {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if let Some(suffix) = name_str.strip_prefix("nvidia") {
                if let Ok(minor) = suffix.parse::<u32>() {
                    if minor > highest {
                        highest = minor;
                    }
                }
            }
        }
    }
    highest
}
```

- [ ] **Step 2: Update RgpuGpuInfo struct to include minor_number**

Update the Rust ioctl struct (lines 455-465):

```rust
#[repr(C)]
struct RgpuGpuInfo {
    name: [u8; 128],
    total_memory: u64,
    index: u32,
    minor_number: u32,
}
```

- [ ] **Step 3: Update create_device to pass minor number and write GPU map**

Replace the `create_device` function (lines 481-536):

```rust
fn create_device(gpu: &GpuInfo, assigned_minor: u32) -> Result<String, String> {
    let fd = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(RGPU_CONTROL_PATH)
        .map_err(|e| format!("failed to open {}: {}", RGPU_CONTROL_PATH, e))?;

    let mut info = RgpuGpuInfo {
        name: [0u8; 128],
        total_memory: gpu.total_memory,
        index: 0,
        minor_number: assigned_minor,
    };

    let display_name = &gpu.device_name;
    let name_bytes = display_name.as_bytes();
    let copy_len = name_bytes.len().min(127);
    info.name[..copy_len].copy_from_slice(&name_bytes[..copy_len]);

    let ret = unsafe {
        libc::ioctl(
            std::os::unix::io::AsRawFd::as_raw_fd(&fd),
            RGPU_IOCTL_ADD_GPU,
            &mut info as *mut RgpuGpuInfo,
        )
    };

    if ret < 0 {
        return Err(format!(
            "ioctl ADD_GPU failed: {}",
            std::io::Error::last_os_error()
        ));
    }

    let instance_id = format!("nvidia{}", assigned_minor);
    tracing::info!(
        "created virtual GPU /dev/{} for {} ({} MB)",
        instance_id,
        display_name,
        gpu.total_memory / (1024 * 1024)
    );

    Ok(instance_id)
}
```

- [ ] **Step 4: Update sync_devices to assign minor numbers and write GPU map file**

In the `sync_devices` method, add minor number assignment logic. After filtering remote GPUs and before calling `create_device`, add:

```rust
// Find next available NVIDIA minor number
let base_minor = find_highest_nvidia_minor() + 1;

// Track assigned minors for GPU map file
let mut gpu_map_entries: Vec<GpuMapEntry> = Vec::new();
let mut next_minor = base_minor;

for gpu in &new_gpus {
    let minor = next_minor;
    next_minor += 1;
    
    match create_device(gpu, minor) {
        Ok(instance_id) => {
            self.devices.push(VirtualDevice {
                instance_id,
                gpu_info: gpu.clone(),
            });
            gpu_map_entries.push(GpuMapEntry {
                pool_index: (self.local_gpu_count + gpu_map_entries.len() as u32),
                minor_number: minor,
                server_id: gpu.server_id,
                device_index: gpu.server_device_index,
                device_name: gpu.device_name.clone(),
                total_memory: gpu.total_memory,
            });
        }
        Err(e) => {
            tracing::warn!("failed to create virtual device for {}: {}", gpu.device_name, e);
        }
    }
}

// Write GPU map file for NVML interpose
write_gpu_map(&gpu_map_entries);
```

- [ ] **Step 5: Add GPU map file writing**

Add the GPU map types and write function:

```rust
#[derive(serde::Serialize)]
struct GpuMapEntry {
    pool_index: u32,
    minor_number: u32,
    server_id: u16,
    device_index: u32,
    device_name: String,
    total_memory: u64,
}

#[derive(serde::Serialize)]
struct GpuMapFile {
    gpus: Vec<GpuMapEntry>,
}

fn write_gpu_map(entries: &[GpuMapEntry]) {
    let map = GpuMapFile {
        gpus: entries.to_vec(),
    };
    
    let gpu_map_dir = if cfg!(target_os = "linux") {
        "/run/rgpu"
    } else {
        // Windows: use temp dir
        return; // GPU map only needed on Linux
    };
    
    let _ = std::fs::create_dir_all(gpu_map_dir);
    let path = format!("{}/gpu_map.json", gpu_map_dir);
    
    match serde_json::to_string_pretty(&map) {
        Ok(json) => {
            if let Err(e) = std::fs::write(&path, &json) {
                tracing::warn!("failed to write GPU map to {}: {}", path, e);
            } else {
                tracing::info!("wrote GPU map to {} ({} entries)", path, entries.len());
            }
        }
        Err(e) => tracing::warn!("failed to serialize GPU map: {}", e),
    }
}
```

- [ ] **Step 6: Commit**

```bash
git add crates/rgpu-client/src/device_manager.rs
git commit -m "feat: device manager assigns NVIDIA minor numbers and writes GPU map file"
```

---

### Task 4: NVML Interpose — Read GPU map and fix device info

**Files:**
- Modify: `crates/rgpu-nvml-interpose/src/lib.rs`

- [ ] **Step 1: Add GPU map reading**

Add after the NvmlState struct definition (around line 190):

```rust
#[derive(serde::Deserialize, Default)]
struct GpuMapFile {
    gpus: Vec<GpuMapEntry>,
}

#[derive(serde::Deserialize)]
struct GpuMapEntry {
    pool_index: u32,
    minor_number: u32,
    server_id: u16,
    device_index: u32,
    device_name: String,
    total_memory: u64,
}

fn read_gpu_map() -> GpuMapFile {
    let path = if cfg!(target_os = "linux") {
        "/run/rgpu/gpu_map.json"
    } else {
        return GpuMapFile::default();
    };
    
    match std::fs::read_to_string(path) {
        Ok(json) => serde_json::from_str(&json).unwrap_or_default(),
        Err(_) => GpuMapFile::default(),
    }
}
```

Add `gpu_map: GpuMapFile` field to `NvmlState`.

- [ ] **Step 2: Load GPU map during initialization**

In the nvmlInit function, after querying remote GPUs, add:

```rust
state.gpu_map = read_gpu_map();
```

- [ ] **Step 3: Fix nvmlDeviceGetMinorNumber to use GPU map**

Update the remote path in `nvmlDeviceGetMinorNumber_impl` (around line 820):

```rust
// Remote GPU — look up assigned minor from GPU map
let idx = remote_index(device);
let state = get_state().lock().expect("nvml state lock");
if let Some(entry) = state.gpu_map.gpus.iter().find(|e| {
    let remote_idx = (e.pool_index as usize).checked_sub(state.local_gpu_count as usize);
    remote_idx == Some(idx)
}) {
    *minor = entry.minor_number;
    return NVML_SUCCESS;
}
// Fallback: local_count + idx
*minor = state.local_gpu_count + idx as u32;
NVML_SUCCESS
```

- [ ] **Step 4: Fix nvmlDeviceGetName to remove "(Remote - RGPU)" suffix**

Update the remote path in `nvmlDeviceGetName_impl` (around line 476):

```rust
// Remote GPU — return raw device name (no suffix for transparency)
if let Some(gpu) = state.remote_gpus.get(idx) {
    let name_str = &gpu.device_name;
    write_c_string(name, length, name_str);
    return NVML_SUCCESS;
}
```

- [ ] **Step 5: Fix nvmlDeviceGetPciInfo to use realistic PCI addresses**

Update the remote path in `nvmlDeviceGetPciInfo_v3_impl` (around line 702):

```rust
// Remote GPU — generate realistic PCI info
let gpu = &state.remote_gpus[idx];
let bus = 0x80 + gpu.server_id as u32;  // High bus number (128+)
let device_slot = gpu.server_device_index;
let bus_id = format!("00000000:{:02X}:{:02X}.0", bus, device_slot);

// Zero the struct first
std::ptr::write_bytes(pci as *mut u8, 0, std::mem::size_of::<nvmlPciInfo_t>());

// Fill busIdLegacy (first 16 bytes) and busId (at offset 48, 32 bytes)
let bus_id_bytes = bus_id.as_bytes();
let legacy_len = bus_id_bytes.len().min(15);
std::ptr::copy_nonoverlapping(
    bus_id_bytes.as_ptr(),
    (*pci).busIdLegacy.as_mut_ptr() as *mut u8,
    legacy_len,
);
let full_len = bus_id_bytes.len().min(31);
std::ptr::copy_nonoverlapping(
    bus_id_bytes.as_ptr(),
    (*pci).busId.as_mut_ptr() as *mut u8,
    full_len,
);

(*pci).domain = 0;
(*pci).bus = bus;
(*pci).device = device_slot;
(*pci).pciDeviceId = ((gpu.device_id as u32) << 16) | gpu.vendor_id;
(*pci).pciSubSystemId = 0;
```

- [ ] **Step 6: Fix nvmlDeviceGetUUID to use NVIDIA-standard format**

Update the remote path in `nvmlDeviceGetUUID_impl` (around line 515):

```rust
// Generate deterministic UUID in NVIDIA's format: GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
let gpu = &state.remote_gpus[idx];
let hash_input = format!("RGPU-{}-{}", gpu.server_id, gpu.server_device_index);
// Simple deterministic hash from the input string
let mut hash: u64 = 0xcbf29ce484222325; // FNV-1a offset basis
for byte in hash_input.as_bytes() {
    hash ^= *byte as u64;
    hash = hash.wrapping_mul(0x100000001b3); // FNV-1a prime
}
let hash2 = hash.wrapping_mul(0x517cc1b727220a95);

let uuid = format!(
    "GPU-{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
    (hash >> 32) as u32,
    (hash >> 16) as u16,
    hash as u16,
    (hash2 >> 48) as u16,
    hash2 & 0xFFFFFFFFFFFF,
);
write_c_string(buffer, length, &uuid);
```

- [ ] **Step 7: Commit**

```bash
git add crates/rgpu-nvml-interpose/src/lib.rs
git commit -m "feat: NVML interpose reads GPU map, returns transparent device info"
```

---

### Task 5: Docker GPU Visibility Test

**Files:**
- Create: `tests/docker/test_docker_gpu_visibility.sh`

- [ ] **Step 1: Create the test script**

```bash
#!/bin/bash
# test_docker_gpu_visibility.sh — Verify Docker sees remote GPUs
#
# This test runs INSIDE a Docker container started with --gpus all.
# It checks that nvidia-smi and CUDA see the expected number of GPUs.
#
# Expected environment:
#   EXPECTED_GPU_COUNT — total GPUs (local + remote) expected

set -euo pipefail

echo "=== Docker GPU Visibility Test ==="

EXPECTED=${EXPECTED_GPU_COUNT:-2}
pass=0
fail=0

check() {
    local name="$1"
    local result="$2"
    if [ "$result" -eq 0 ]; then
        echo "  PASS: $name"
        pass=$((pass + 1))
    else
        echo "  FAIL: $name"
        fail=$((fail + 1))
    fi
}

# 1. nvidia-smi must be available
nvidia-smi > /dev/null 2>&1
check "nvidia-smi runs" $?

# 2. nvidia-smi must show expected GPU count
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "  INFO: nvidia-smi reports $GPU_COUNT GPU(s)"
if [ "$GPU_COUNT" -eq "$EXPECTED" ]; then
    echo "  PASS: GPU count matches expected ($EXPECTED)"
    pass=$((pass + 1))
else
    echo "  FAIL: GPU count $GPU_COUNT != expected $EXPECTED"
    fail=$((fail + 1))
fi

# 3. List all GPU names
echo "  INFO: GPUs found:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | while read line; do
    echo "    $line"
done

# 4. Check CUDA device count (if nvcc/cuda available)
if command -v python3 > /dev/null 2>&1; then
    CUDA_COUNT=$(python3 -c "
import ctypes
libcuda = ctypes.CDLL('libcuda.so.1')
libcuda.cuInit(0)
count = ctypes.c_int(0)
libcuda.cuDeviceGetCount(ctypes.byref(count))
print(count.value)
" 2>/dev/null || echo "0")
    echo "  INFO: CUDA reports $CUDA_COUNT device(s)"
    if [ "$CUDA_COUNT" -eq "$EXPECTED" ]; then
        echo "  PASS: CUDA device count matches expected ($EXPECTED)"
        pass=$((pass + 1))
    else
        echo "  FAIL: CUDA device count $CUDA_COUNT != expected $EXPECTED"
        fail=$((fail + 1))
    fi
fi

echo "=== Docker GPU Visibility: $pass passed, $fail failed ==="
exit $((fail > 0 ? 1 : 0))
```

- [ ] **Step 2: Commit**

```bash
git add tests/docker/test_docker_gpu_visibility.sh
chmod +x tests/docker/test_docker_gpu_visibility.sh
git commit -m "test: add Docker GPU visibility test script"
```

---

### Task 6: Integration Test — End-to-End Docker GPU Transparency

This task verifies the full pipeline on a Linux host with the kernel module loaded.

- [ ] **Step 1: Build and install kernel module**

```bash
cd drivers/linux/rgpu-vgpu
make
sudo insmod rgpu_vgpu.ko
```

- [ ] **Step 2: Start RGPU server on remote machine**

On the remote machine (e.g. 192.168.178.100):
```bash
rgpu server --config rgpu.toml
```

- [ ] **Step 3: Start RGPU client daemon**

```bash
rgpu client --config rgpu-client.toml
```

Expected log: `"created virtual GPU /dev/nvidia1 for NVIDIA GeForce RTX 3070"`

- [ ] **Step 4: Verify device node exists**

```bash
ls -la /dev/nvidia*
```

Expected: `/dev/nvidia0` (real) and `/dev/nvidia1` (RGPU virtual)

- [ ] **Step 5: Verify nvidia-smi shows both GPUs**

```bash
LD_PRELOAD=/usr/lib/rgpu/libnvidia-ml.so.1 nvidia-smi
```

Expected: Both GPUs listed, no "RGPU" branding

- [ ] **Step 6: Verify Docker sees both GPUs**

```bash
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi
```

Expected: Both GPUs visible inside the container

- [ ] **Step 7: Run FFmpeg NVENC in Docker on remote GPU**

```bash
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 bash -c "
  apt-get update -qq && apt-get install -y -qq ffmpeg > /dev/null 2>&1
  ffmpeg -f lavfi -i testsrc=duration=1:size=320x240:rate=30 -c:v h264_nvenc -gpu 1 /tmp/test.mp4 2>&1 | tail -5
"
```

Expected: Encoding succeeds on GPU 1 (remote RTX 3070)

- [ ] **Step 8: Commit any fixes**

```bash
git add -A
git commit -m "fix: integration test fixes for Docker GPU transparency"
```

---

### Task 7: Update project documentation

- [ ] **Step 1: Update claude_mem.md**

Add to current status:
```markdown
### Docker GPU Transparency
- Kernel module creates /dev/nvidia{N} with major 195 for remote GPUs
- Device manager scans real NVIDIA minors, assigns next available
- GPU map file (/run/rgpu/gpu_map.json) bridges device manager → NVML interpose
- NVML interpose returns transparent device info (no "RGPU" branding)
- Docker --gpus all sees remote GPUs as local hardware
```

- [ ] **Step 2: Update todos.md**

Mark "Architectuur: Virtual GPU Hardware" as completed with details.

- [ ] **Step 3: Commit**

```bash
git add claude_mem.md todos.md
git commit -m "docs: update project docs with Docker GPU transparency status"
```
