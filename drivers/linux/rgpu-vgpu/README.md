# RGPU Virtual GPU Kernel Module

A minimal Linux kernel module that creates virtual GPU device files in `/dev/` for remote GPUs managed by the RGPU daemon. After loading this module and having the daemon send `RGPU_IOCTL_ADD_GPU`, device files appear that discovery tools can find.

## What It Does

- Creates a control device `/dev/rgpu_control` that the RGPU daemon uses to manage virtual GPUs via ioctl
- Creates `/dev/rgpu_gpu0`, `/dev/rgpu_gpu1`, etc. for each virtual GPU added by the daemon
- Registers a `platform_device` per virtual GPU for sysfs visibility
- Supports up to `max_gpus` (default 16) virtual GPU devices

## Build Requirements

- Linux kernel headers for your running kernel
- GCC and Make

Install on Debian/Ubuntu:
```bash
sudo apt install linux-headers-$(uname -r) build-essential
```

Install on Fedora/RHEL:
```bash
sudo dnf install kernel-devel kernel-headers gcc make
```

## Building

```bash
make
```

This produces `rgpu_vgpu.ko`.

## Installation

### Manual (insmod)

```bash
sudo insmod rgpu_vgpu.ko
sudo insmod rgpu_vgpu.ko max_gpus=32  # custom max
```

### modprobe (after install)

```bash
sudo make install
sudo modprobe rgpu_vgpu

# Load at boot
echo "rgpu_vgpu" | sudo tee /etc/modules-load.d/rgpu.conf

# Set module parameters at boot
echo "options rgpu_vgpu max_gpus=32" | sudo tee /etc/modprobe.d/rgpu.conf
```

### DKMS (recommended)

DKMS automatically rebuilds the module when the kernel is updated:

```bash
sudo cp -r . /usr/src/rgpu-vgpu-0.1.0
sudo dkms add rgpu-vgpu/0.1.0
sudo dkms build rgpu-vgpu/0.1.0
sudo dkms install rgpu-vgpu/0.1.0
```

To remove:
```bash
sudo dkms remove rgpu-vgpu/0.1.0 --all
```

### udev Rules

Install the udev rules for proper device permissions:

```bash
sudo cp 99-rgpu.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```

This sets:
- `/dev/rgpu_control` to mode 0660, group `video`
- `/dev/rgpu_gpu*` to mode 0666 (accessible to all users)

## Daemon Communication (ioctl)

The RGPU daemon communicates with the module through ioctl calls on `/dev/rgpu_control`.

### RGPU_IOCTL_ADD_GPU

Creates a new virtual GPU device. The daemon sends a `struct rgpu_gpu_info` with the GPU name and VRAM size. The kernel assigns an index and creates `/dev/rgpu_gpuN`.

```c
struct rgpu_gpu_info {
    char name[128];        // e.g., "NVIDIA GeForce RTX 3070 (Remote - RGPU)"
    uint64_t total_memory; // VRAM in bytes
    uint32_t index;        // assigned by kernel on ADD, input on REMOVE
};

int fd = open("/dev/rgpu_control", O_RDWR);
struct rgpu_gpu_info info = {
    .name = "NVIDIA GeForce RTX 3070 (Remote - RGPU)",
    .total_memory = 8589934592ULL,  // 8 GB
};
ioctl(fd, RGPU_IOCTL_ADD_GPU, &info);
// info.index now contains the assigned GPU index
```

### RGPU_IOCTL_REMOVE_GPU

Removes a virtual GPU device by index.

```c
struct rgpu_gpu_info info = { .index = 0 };
ioctl(fd, RGPU_IOCTL_REMOVE_GPU, &info);
```

### RGPU_IOCTL_LIST_GPUS

Lists all active virtual GPUs.

```c
size_t sz = sizeof(struct rgpu_gpu_list) + 16 * sizeof(struct rgpu_gpu_info);
struct rgpu_gpu_list *list = calloc(1, sz);
list->max_count = 16;
ioctl(fd, RGPU_IOCTL_LIST_GPUS, list);
for (uint32_t i = 0; i < list->count; i++) {
    printf("GPU %u: %s (%llu MB)\n",
           list->infos[i].index,
           list->infos[i].name,
           list->infos[i].total_memory / (1024*1024));
}
```

## Module Parameters

| Parameter  | Default | Description                           |
|------------|---------|---------------------------------------|
| `max_gpus` | 16      | Maximum number of virtual GPU devices |

## Secure Boot / Module Signing

If Secure Boot is enabled, the kernel will refuse to load unsigned modules. You have two options:

1. **Sign the module** with your machine's MOK (Machine Owner Key):
   ```bash
   # Generate a signing key (one time)
   openssl req -new -x509 -newkey rsa:2048 -keyout MOK.priv -outform DER \
       -out MOK.der -nodes -days 36500 -subj "/CN=RGPU Module Signing/"

   # Enroll the key (requires reboot + MOK manager confirmation)
   sudo mokutil --import MOK.der

   # Sign the module after each build
   /usr/src/linux-headers-$(uname -r)/scripts/sign-file sha256 \
       MOK.priv MOK.der rgpu_vgpu.ko
   ```

2. **Use DKMS with sign_tool** (automated signing on kernel update):
   Add to `/etc/dkms/framework.conf`:
   ```
   sign_tool="/etc/dkms/sign_helper.sh"
   ```

## Unloading

```bash
sudo rmmod rgpu_vgpu
# or
sudo modprobe -r rgpu_vgpu
```

## Troubleshooting

Check if the module is loaded:
```bash
lsmod | grep rgpu
```

View kernel log messages:
```bash
dmesg | grep rgpu
```

Verify device files:
```bash
ls -la /dev/rgpu_*
```
