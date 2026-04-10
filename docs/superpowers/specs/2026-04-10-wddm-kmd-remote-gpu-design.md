# WDDM KMD Remote GPU Driver — Design Spec

## Goal

Make a remote GPU completely indistinguishable from a local GPU on Windows. Every application — Docker, Ollama, PyTorch, FFmpeg, nvidia-smi, Device Manager — sees and uses the remote GPU as if it is physically installed hardware. No special configuration, no library swapping, no volume mounts. NVIDIA's own user-mode drivers (nvcuda.dll, nvml.dll, nvEncodeAPI64.dll) load against our kernel driver and think they're talking to real hardware.

## Requirements

- Remote GPU appears in Device Manager as a real NVIDIA display adapter
- NVIDIA's own user-mode drivers load automatically (Windows finds matching vendor/device ID)
- `docker run --gpus all` includes the remote GPU — no extra flags
- `nvidia-smi` shows the remote GPU with correct name, memory, temperature
- CUDA, NVENC, NVDEC, Vulkan all work transparently
- Hot-swap: GPU appears when server connects, disappears when server disconnects
- Graceful: running apps get device-removed notification on disconnect
- Local GPU continues working normally alongside remote GPU(s)
- Test-signed driver for development; production signing later

## Architecture

```
Client PC                                         Server PC
┌─────────────────────────────────────┐          ┌──────────────────┐
│                                     │          │                  │
│  Applications (Docker, Ollama, ...) │          │  RGPU Server     │
│         │                           │          │  ├── RTX 3070    │
│         ▼                           │          │  ├── NVIDIA KMD  │
│  NVIDIA User-Mode Drivers           │          │  ├── nvcuda.dll  │
│  (nvcuda.dll, nvml.dll — NVIDIA's!) │          │  └── listens on  │
│         │                           │          │      :9876       │
│         ▼                           │          └──────────────────┘
│  Windows Dxgkrnl.sys                │                   ▲
│         │                           │                   │
│    ┌────┴────────────┐              │                   │
│    │                 │              │                   │
│    ▼                 ▼              │              TCP/IP
│  NVIDIA KMD      RGPU KMD          │                   │
│  (real HW)       (rgpu-kmd.sys)    │                   │
│    │                 │              │                   │
│    ▼                 ▼              │                   │
│  GTX 1060        Shared Memory     │                   │
│  (local)         Ring Buffer       │                   │
│                      │              │                   │
│                      ▼              │                   │
│                  RGPU Daemon ───────┼───────────────────┘
│                  (user-mode)        │
│                      │              │
│                      ▼              │
│                  RGPU Bus Driver    │
│                  (rgpu-bus.sys)     │
│                  Hot-swap control   │
│                                     │
└─────────────────────────────────────┘
```

## Component 1: Virtual PCI Bus Driver (`rgpu-bus.sys`)

### Purpose

Creates and removes child PCI devices on demand. When the RGPU daemon connects to a server and discovers remote GPUs, it tells the bus driver to enumerate new child devices. When the server disconnects, the bus driver removes them.

### Behavior

- Installs as a root-enumerated bus driver (`Root\RGPUBus`)
- Listens on a control device `\\.\RGPUBusControl` for IOCTLs from the daemon
- IOCTLs:
  - `IOCTL_RGPU_ADD_GPU`: daemon provides vendor ID, device ID, subsystem ID, revision, GPU name, memory size, register dump. Bus driver creates a child PDO (Physical Device Object) with those hardware IDs.
  - `IOCTL_RGPU_REMOVE_GPU`: bus driver triggers PnP surprise removal for that child device
  - `IOCTL_RGPU_LIST_GPUS`: returns currently enumerated child devices
- Child device hardware ID format: `PCI\VEN_10DE&DEV_XXXX&SUBSYS_XXXXXXXX&REV_XX`
- When Windows PnP sees the child device, it looks for a matching driver → finds rgpu-kmd.sys (our INF matches these IDs)

### Hot-swap Flow

```
Server connects:
  Daemon: IOCTL_RGPU_ADD_GPU → Bus driver
  Bus driver: IoInvalidateDeviceRelations(BusRelations)
  Windows PnP: enumerates new child → loads rgpu-kmd.sys
  Windows: "New display adapter detected"
  NVIDIA UMD loads automatically

Server disconnects:
  Daemon: IOCTL_RGPU_REMOVE_GPU → Bus driver
  Bus driver: marks child as removed → IoInvalidateDeviceRelations
  Windows PnP: surprise removal
  NVIDIA UMD unloads
  Device Manager: GPU disappears

Server reconnects:
  Same as "Server connects" — fresh enumeration
```

### Key Implementation Details

- Built using WDM (not WDF) for direct PnP control over child PDO creation
- Child PDOs report PCI-like hardware IDs but are not on a real PCI bus
- The bus driver does NOT handle any GPU operations — it only manages device lifecycle
- Maximum 16 child devices (configurable)

## Component 2: WDDM Display Miniport Driver (`rgpu-kmd.sys`)

### Purpose

WDDM 2.x display miniport driver that Windows loads for each remote GPU. NVIDIA's user-mode drivers communicate with this KMD through Dxgkrnl.sys, and our KMD forwards everything to the remote server via shared memory with the daemon.

### Initialization (`DxgkDdiStartDevice`)

When Windows loads our KMD for a child device:

1. KMD reads device properties from the bus driver (vendor ID, device ID, register dump)
2. KMD sets up shared memory ring buffer for daemon communication
3. KMD waits for daemon to connect (daemon monitors for new RGPU devices)
4. KMD populates GPU capabilities from the register dump:
   - `DXGK_DRIVERCAPS` — compute capability, memory size, engine count
   - BAR0 register space — cached register values from the real GPU
5. KMD reports success to Dxgkrnl → adapter is online

### BAR0 Register Emulation

NVIDIA's KMD/UMD reads GPU registers to detect architecture and capabilities. Our KMD emulates this:

- At server connect, daemon sends a register dump: ~4KB of key GPU registers from BAR0 of the real GPU
- Registers include: GPU ID, architecture, compute class, memory config, clock domains
- When NVIDIA's UMD calls `DxgkDdiEscape` to read registers, our KMD returns cached values
- When Dxgkrnl calls `DxgkDdiQueryAdapterInfo`, our KMD returns capabilities derived from the register dump

**Register dump protocol:**
- Server reads BAR0 registers 0x000-0xFFF via direct MMIO
- Sends as binary blob to client daemon
- Daemon passes to KMD via IOCTL at device startup
- KMD stores in non-paged pool memory

### GPU Memory Management

```
DxgkDdiCreateAllocation(size, type, flags)
  → KMD assigns local virtual address
  → Sends allocation request to daemon → server
  → Server: real cudaMalloc() on remote GPU
  → Returns remote GPU address
  → KMD stores mapping: local_va → remote_addr
  → Returns local_va to Dxgkrnl

DxgkDdiDestroyAllocation(local_va)
  → KMD looks up remote_addr
  → Sends free request to daemon → server
  → Server: real cudaFree()
  → KMD removes mapping
```

Allocation mapping table stored in non-paged pool, protected by spinlock.

### Command Buffer Submission

```
DxgkDdiSubmitCommand(DMA_buffer, allocation_list)
  → KMD copies DMA buffer to shared memory ring
  → Translates allocation list (local_va → remote_addr)
  → Signals daemon via KEVENT
  → Daemon reads from ring → sends via TCP to server
  → Server submits DMA buffer to real GPU hardware
  → Server waits for GPU completion
  → Server sends completion signal
  → Daemon writes completion to ring → signals KMD
  → KMD calls DxgkCbNotifyInterrupt(DXGK_INTERRUPT_DMA_COMPLETED)
  → Dxgkrnl unblocks the waiting thread
```

**Critical:** DMA buffers are forwarded as-is (opaque binary blobs). We do NOT parse or modify them. The remote GPU must be the exact same model/architecture as advertised, so the command format matches.

### DxgkDdiEscape (Vendor-Specific Communication)

NVIDIA's UMD uses escape calls for:
- Reading GPU registers
- Setting clocks/power state
- Querying GPU capabilities
- NVENC/NVDEC session management

Our KMD intercepts all escape calls:
- Register reads → return cached values
- State queries → forward to server and return response
- NVENC/NVDEC operations → forward to server

### Interrupt Emulation

Real GPUs signal completion via MSI interrupts. Our KMD emulates this:
- Daemon signals a KEVENT when server reports completion
- KMD's DPC (Deferred Procedure Call) fires
- DPC calls `DxgkCbNotifyInterrupt` with the appropriate interrupt type
- Dxgkrnl processes the completion

## Component 3: Shared Memory Ring Buffer

### Structure

```c
#define RING_SIZE (4 * 1024 * 1024)  // 4MB ring buffer

struct RgpuRingBuffer {
    volatile ULONG32 write_offset;     // KMD writes, daemon reads
    volatile ULONG32 read_offset;      // daemon writes, KMD reads
    volatile ULONG32 response_write;   // daemon writes responses
    volatile ULONG32 response_read;    // KMD reads responses
    KEVENT            request_event;   // KMD signals "new request"
    KEVENT            response_event;  // daemon signals "response ready"
    UCHAR             request_data[RING_SIZE];
    UCHAR             response_data[RING_SIZE];
};
```

### Request Format

```c
struct RgpuRequest {
    ULONG32 type;          // CREATE_ALLOC, SUBMIT_CMD, ESCAPE, READ_REG, ...
    ULONG32 sequence_id;   // For matching responses
    ULONG32 payload_size;  // Size of following data
    UCHAR   payload[];     // Type-specific data
};
```

### Setup

- KMD allocates ring buffer in non-paged pool during DxgkDdiStartDevice
- KMD creates an MDL (Memory Descriptor List) for the buffer
- Daemon maps the MDL into its address space via DeviceIoControl(IOCTL_RGPU_MAP_RING)
- Both KMD and daemon access the same physical memory — zero-copy

## Component 4: Daemon Extensions

### New Responsibilities

The existing RGPU daemon gets these additions:

1. **Bus driver communication**: sends IOCTL_RGPU_ADD/REMOVE_GPU when servers connect/disconnect
2. **KMD ring buffer**: maps shared memory, polls for requests, sends to server, writes responses
3. **Register dump relay**: at server connect, requests GPU register dump and passes to KMD
4. **Heartbeat monitoring**: 5-second pings, 3 missed → disconnect → GPU removal

### Register Dump Request

New protocol message:
```
Client → Server: DumpRegisters { device_index: u32 }
Server → Client: RegisterDump { 
    vendor_id: u32,
    device_id: u32, 
    subsystem_id: u32,
    revision: u8,
    bar0_data: Vec<u8>,    // ~4KB of BAR0 register values
    gpu_name: String,
    total_memory: u64,
    compute_capability: (i32, i32),
}
```

### Server-Side Register Reading

Server reads GPU registers via:
- Linux: `/sys/bus/pci/devices/XXXX:XX:XX.X/resource0` (BAR0 mmap)
- Windows: `MmMapIoSpace` on BAR0 physical address (requires kernel driver on server too) OR use NVML/CUDA driver API to query same information

Practical approach for Fase 1: use NVML + CUDA API to gather all needed info (device name, memory, compute capability, PCI info, clocks, etc.) instead of raw BAR0 register dump. Only dump actual BAR0 if NVIDIA's UMD specifically reads register offsets we can't emulate via API queries.

## Component 5: INF Files

### rgpu-bus.inf

```inf
[Manufacturer]
%RGPU%=RGPU,NTamd64

[RGPU.NTamd64]
%RGPUBus.DeviceDesc%=RGPUBus_Install, Root\RGPUBus

[RGPUBus_Install.NT]
CopyFiles=RGPUBus_CopyFiles

[RGPUBus_CopyFiles]
rgpu-bus.sys

[RGPUBus_Install.NT.Services]
AddService=RGPUBus,0x00000002,RGPUBus_Service

[RGPUBus_Service]
ServiceType=1
StartType=3
ErrorControl=1
ServiceBinary=%13%\rgpu-bus.sys
```

### rgpu-kmd.inf

```inf
[Manufacturer]
%RGPU%=RGPU,NTamd64

[RGPU.NTamd64]
; Match ALL NVIDIA GPU device IDs — our bus driver creates devices with these IDs
%RGPUGPU.DeviceDesc%=RGPUKMD_Install, PCI\VEN_10DE&DEV_2484  ; RTX 3070
%RGPUGPU.DeviceDesc%=RGPUKMD_Install, PCI\VEN_10DE&DEV_2684  ; RTX 4090
; ... etc for all supported GPUs, OR use a wildcard match

[RGPUKMD_Install.NT]
CopyFiles=RGPUKMD_CopyFiles
FeatureScore=F0  ; Lower than NVIDIA's real driver (FC) so we don't hijack real hardware

[RGPUKMD_CopyFiles]
rgpu-kmd.sys
```

**FeatureScore:** Windows uses this to pick between competing drivers for the same hardware ID. NVIDIA's driver has FeatureScore=FC. We use F0 (lower) so that on machines with REAL NVIDIA hardware, NVIDIA's driver wins. Our driver only loads for devices created by our bus driver (which have no real PCI backing, so NVIDIA's driver can't claim them).

## Phased Implementation

### Fase 1: CUDA Compute (this spec)
- `rgpu-bus.sys` — virtual bus with hot-swap
- `rgpu-kmd.sys` — WDDM miniport with:
  - Device registration and capability reporting
  - BAR0 register emulation (via API-gathered data, not raw dump)
  - DxgkDdiCreateAllocation / DxgkDdiDestroyAllocation → forwarded to server
  - DxgkDdiSubmitCommand → DMA buffer forwarding to server
  - DxgkDdiEscape → forwarded to server
  - Interrupt emulation for completion
- Daemon extensions — ring buffer, bus IOCTLs, register dump relay
- Protocol extension — RegisterDump message
- Server extension — register/capability query on behalf of clients

### Fase 2: NVENC/NVDEC
- DxgkDdiEscape handling for encode/decode session management
- Video memory allocation routing for encode/decode buffers
- Bitstream forwarding

### Fase 3: Vulkan Rendering
- DxgkDdiRender for Vulkan command buffer submission
- Presentation/swapchain support
- GPU memory mapping for shader resources

### Fase 4: Display Output
- Virtual monitor via IddCx on top of the WDDM adapter
- Framebuffer readback from server
- Display mode enumeration

## Development Environment

- **WDK (Windows Driver Kit)** — required for building WDDM drivers
- **Visual Studio 2022** — with WDK integration
- **Test machine** — separate PC or VM with test-signing enabled (`bcdedit -set TESTSIGNING ON`)
- **WinDbg** — kernel debugging via network or serial
- **Driver Verifier** — enabled during development to catch bugs early

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| BSOD during development | Machine crash | Develop on dedicated test machine, kernel debug via network |
| NVIDIA UMD reads unknown registers | Driver fails to init | Start with API-gathered capabilities, add register emulation incrementally based on what fails |
| DMA buffer format unknown | Commands don't execute | Forward as opaque blobs — server's real driver interprets them |
| Network latency kills performance | GPU operations slow | Batch commands, async completion, pipeline submissions |
| NVIDIA driver update breaks compatibility | Remote GPU stops working | Version-lock server/client driver versions, test with each NVIDIA update |
| FeatureScore conflict with real NVIDIA driver | Our driver hijacks real GPU | Use lower FeatureScore + INF flags to only match bus-driver-created devices |

## Success Criteria

1. Device Manager shows remote GPU as a real NVIDIA display adapter
2. `nvidia-smi` shows both local and remote GPU with correct info
3. CUDA app (`deviceQuery`) sees both GPUs and can allocate/compute on the remote one
4. `docker run --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi` shows both GPUs
5. Ollama in Docker uses the remote GPU for inference without any special configuration
6. Hot-swap: GPU appears/disappears cleanly when server connects/disconnects
7. Local GPU is completely unaffected
