# RGPU - Remote GPU Sharing

RGPU enables sharing GPUs over the network. It exposes local GPUs as remote resources accessible by clients, supporting both **Vulkan** and **CUDA** workloads. A single `rgpu` binary handles server, client, and GUI modes.

```
┌──────────────┐         TCP/QUIC+TLS          ┌──────────────┐
│  Client Host │ ◄───────────────────────────► │  GPU Server  │
│              │                               │              │
│  Application │                               │  NVIDIA GPU  │
│      ▼       │                               │  (RTX 3070)  │
│  CUDA / Vk   │                               │              │
│  Interpose   │                               │  CUDA Driver │
│  Libraries   │ ── IPC ──► Client Daemon ───► │  Vulkan RT   │
└──────────────┘                               └──────────────┘
```

## Features

- **Vulkan ICD Driver** - Presents remote GPUs as local Vulkan physical devices
- **CUDA Driver API Interpose** - Intercepts 200+ CUDA functions, forwards to remote GPUs
- **Multi-Server GPU Pool** - Aggregate GPUs from multiple servers into a single pool
- **Desktop GUI** - Real-time monitoring, server control, and configuration editor
- **Embedded Server** - Start/stop a GPU server directly from the UI
- **Dynamic Connections** - Add and remove server connections at runtime
- **QUIC Transport** - Optional QUIC (always TLS 1.3) alongside TCP+TLS
- **LZ4 Compression** - Automatic payload compression for large transfers
- **Zero-Copy Serialization** - rkyv-based wire protocol for minimal overhead
- **Token Authentication** - HMAC-SHA256 challenge-response security
- **Cross-Platform** - Windows, Linux, macOS

## Quick Start

### Build from Source

```bash
# Clone and build
git clone https://github.com/Fill84/RGPU.git
cd RGPU
cargo build --release
```

Build artifacts:
- `target/release/rgpu` (or `rgpu.exe`) - CLI binary
- `target/release/librgpu_cuda_interpose.so` (or `.dll` / `.dylib`) - CUDA interpose library
- `target/release/librgpu_vk_icd.so` (or `.dll` / `.dylib`) - Vulkan ICD driver

### 1. Generate a Token

```bash
rgpu token --name my-client
```

Output:
```
Generated RGPU token for 'my-client':

  a3f8b2c1d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1

Add this to your server's rgpu.toml:

  [[security.tokens]]
  token = "a3f8b2c1..."
  name = "my-client"
```

### 2. Start the Server

On the machine with the GPU:

```bash
# Simple start (no auth, development)
rgpu server

# With config file
rgpu server --config /etc/rgpu/rgpu.toml

# Custom port and bind address
rgpu server --port 9876 --bind 0.0.0.0
```

### 3. Query GPU Info

From any machine:

```bash
rgpu info --server gpu-server.local:9876 --token <your-token>
```

Output:
```
Connected to RGPU server at gpu-server.local:9876
Available GPUs:

  GPU 0: NVIDIA GeForce RTX 3070
    Type:     DiscreteGpu
    VRAM:     8192 MB
    Vulkan:   true
    CUDA:     true
    Compute:  8.6
```

### 4. Start the Client Daemon

On the machine that wants to use remote GPUs:

```bash
# Connect to one server
rgpu client --server gpu-server.local:9876 --token <your-token>

# Connect to multiple servers (GPU pool)
rgpu client \
  --server gpu-server-1.local:9876 \
  --server gpu-server-2.local:9876 \
  --token <your-token>
```

### 5. Launch the GUI

```bash
# Start with no pre-configured servers (add via Control panel)
rgpu ui

# Start with server connections
rgpu ui --server gpu-server.local:9876 --token <your-token>

# Custom poll interval
rgpu ui --poll-interval 5
```

The GUI has four tabs:
- **Control** - Start/stop an embedded server, manage connections
- **GPU Overview** - View all GPUs grouped by server
- **Metrics** - Live charts for connections, requests, CUDA/Vulkan commands
- **Config Editor** - Visual editor for `rgpu.toml`

## Using Remote GPUs

### CUDA Applications

The CUDA interpose library replaces the standard CUDA driver, forwarding all API calls to remote GPUs through the client daemon.

**Linux:**
```bash
# Start the client daemon first, then:
LD_PRELOAD=/path/to/librgpu_cuda_interpose.so ./my_cuda_app
```

**Windows:**
Place `rgpu_cuda_interpose.dll` (renamed to `nvcuda.dll`) in the same directory as your application, or earlier in the DLL search path than the real `nvcuda.dll`.

### Vulkan Applications

The Vulkan ICD driver registers with the Vulkan loader and presents remote GPUs as local physical devices.

**Linux:**
```bash
# Point Vulkan loader to the ICD manifest
VK_ICD_FILENAMES=/path/to/rgpu_icd.json ./my_vulkan_app
```

**Windows:**
The installer registers the ICD automatically via the Windows registry. After installation, Vulkan applications will see remote GPUs alongside local ones.

**Verify registration:**
```bash
vulkaninfo --summary
```

## Configuration

RGPU uses a TOML configuration file (`rgpu.toml`). All settings can also be overridden via CLI flags.

### Full Example

```toml
[server]
bind = "0.0.0.0"
port = 9876
server_id = 1
max_clients = 16
transport = "tcp"        # "tcp" or "quic"
# cert_path = "/etc/rgpu/cert.pem"
# key_path = "/etc/rgpu/key.pem"
# expose_gpus = [0, 1]  # Expose specific GPUs only (default: all)

[client]
gpu_ordering = "LocalFirst"  # "LocalFirst", "RemoteFirst", "ByCapability"
include_local_gpus = true

[[client.servers]]
address = "gpu-server-1.local:9876"
token = "your-token-here"
transport = "tcp"

[[client.servers]]
address = "gpu-server-2.local:9876"
token = "another-token"
transport = "quic"

[[security.tokens]]
token = "a3f8b2c1d4e5f6..."
name = "workstation-1"
# allowed_gpus = [0]       # Restrict to specific GPUs
# max_memory = 4294967296  # 4 GB memory limit

[[security.tokens]]
token = "b4c9d3e2f5a6b7..."
name = "render-farm"
```

### Configuration Reference

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `server` | `bind` | `0.0.0.0` | Bind address |
| `server` | `port` | `9876` | Listen port |
| `server` | `server_id` | `0` | Unique ID for multi-server pools |
| `server` | `max_clients` | `16` | Maximum concurrent connections |
| `server` | `transport` | `tcp` | Transport protocol (`tcp` or `quic`) |
| `server` | `cert_path` | - | TLS certificate (PEM) |
| `server` | `key_path` | - | TLS private key (PEM) |
| `server` | `expose_gpus` | all | GPU indices to expose |
| `client` | `gpu_ordering` | `LocalFirst` | GPU ordering in pool |
| `client` | `include_local_gpus` | `true` | Include local GPUs in pool |
| `client.servers` | `address` | - | Server `host:port` |
| `client.servers` | `token` | - | Authentication token |
| `client.servers` | `transport` | `tcp` | Per-server transport override |
| `security.tokens` | `token` | - | Token string |
| `security.tokens` | `name` | - | Human-readable name |
| `security.tokens` | `allowed_gpus` | all | GPU access restriction |
| `security.tokens` | `max_memory` | unlimited | Memory limit (bytes) |

## Multi-Server GPU Pool

RGPU can aggregate GPUs from multiple servers into a single pool. Each server has a unique `server_id`, and the client daemon merges all available GPUs.

```
┌─────────────┐     ┌───────────────────┐     ┌─────────────┐
│  Server 1   │     │   Client Daemon   │     │  Server 2   │
│  server_id=1│◄───►│                   │◄───►│  server_id=2│
│  2x RTX 4090│     │  GPU Pool:        │     │  1x A100    │
└─────────────┘     │   GPU 0: RTX 4090 │     └─────────────┘
                    │   GPU 1: RTX 4090 │
                    │   GPU 2: A100     │
                    └───────────────────┘
```

**GPU Ordering Options:**
- `LocalFirst` (default) - Local GPUs first, then remote
- `RemoteFirst` - Remote GPUs first
- `ByCapability` - Sorted by compute capability (highest first)

## Installation

### From Installers

Pre-built installers are available on the [Releases](https://github.com/Fill84/RGPU/releases) page:

| Platform | Format | File |
|----------|--------|------|
| Windows | Setup.exe | `rgpu-VERSION-windows-x64-setup.exe` |
| Linux (Debian/Ubuntu) | .deb | `rgpu_VERSION_amd64.deb` |
| Linux (Fedora/RHEL) | .rpm | `rgpu-VERSION.x86_64.rpm` |
| macOS | .pkg | `rgpu-VERSION-macos-x64.pkg` |

**Windows installer** includes:
- Binary added to system PATH
- CUDA interpose DLL
- Vulkan ICD with automatic registry registration
- Optional Windows Service
- Uninstaller via Add/Remove Programs

**Linux packages** include:
- Binary in `/usr/bin/`
- Libraries in `/usr/lib/rgpu/`
- Vulkan ICD manifest in `/usr/share/vulkan/icd.d/`
- systemd services (`rgpu-server`, `rgpu-client`)
- Default config in `/etc/rgpu/`

**macOS package** includes:
- Binary in `/usr/local/bin/`
- Libraries in `/usr/local/lib/rgpu/`
- Vulkan ICD manifest
- LaunchDaemon plists for server and client

### From Source

```bash
cargo build --release
```

### As a Linux Service

```bash
# After installing via .deb or .rpm:
sudo systemctl enable --now rgpu-server
sudo systemctl enable --now rgpu-client

# Check status
sudo systemctl status rgpu-server
```

## Architecture

RGPU is a Rust workspace with 11 crates:

```
rgpu-cli              CLI binary (server, client, token, info, ui)
rgpu-server           GPU discovery, CUDA/Vulkan executors, metrics
rgpu-client           Client daemon, IPC listener, connection pool
rgpu-protocol         Wire protocol (rkyv serialization, LZ4 compression)
rgpu-transport        TCP+TLS, QUIC (quinn), authentication
rgpu-core             Configuration (TOML), handle maps
rgpu-common           Logging (tracing), platform detection
rgpu-cuda-interpose   cdylib: CUDA Driver API interception (200+ functions)
rgpu-vk-icd           cdylib: Vulkan ICD (60+ dispatch entries)
rgpu-ui               egui/eframe desktop GUI
```

### Communication Flow

```
Application
    │
    ▼
CUDA Interpose / Vulkan ICD    (cdylib, loaded by app)
    │
    ▼ IPC (Unix socket / Named pipe)
    │
Client Daemon                   (connection pool, handle routing)
    │
    ▼ TCP+TLS / QUIC
    │
RGPU Server                     (GPU discovery, command execution)
    │
    ▼
Real GPU Driver (CUDA / Vulkan)
```

### Wire Protocol

- **Serialization**: rkyv 0.8 (zero-copy deserialization)
- **Compression**: LZ4 for payloads > 512 bytes
- **Authentication**: HMAC-SHA256 challenge-response
- **Transport**: TCP (optional TLS 1.3 via rustls) or QUIC (always TLS 1.3 via quinn)
- **Protocol version**: 3

## CLI Reference

```
rgpu <COMMAND>

Commands:
  server    Start the RGPU server
  client    Start the RGPU client daemon
  token     Generate an authentication token
  info      Query GPU information from a server
  ui        Launch the desktop GUI
  help      Print help
```

### `rgpu server`

```
rgpu server [OPTIONS]

Options:
  -p, --port <PORT>        Listen port [default: 9876]
  -b, --bind <BIND>        Bind address [default: 0.0.0.0]
      --cert <CERT>        TLS certificate file (PEM)
      --key <KEY>          TLS private key file (PEM)
  -c, --config <CONFIG>    Configuration file [default: rgpu.toml]
      --pid-file <PATH>    Write PID to file (for service managers)
```

### `rgpu client`

```
rgpu client [OPTIONS]

Options:
  -s, --server <SERVER>    Server address(es) (host:port), repeatable
  -t, --token <TOKEN>      Authentication token
  -c, --config <CONFIG>    Configuration file [default: rgpu.toml]
      --pid-file <PATH>    Write PID to file (for service managers)
```

### `rgpu token`

```
rgpu token [OPTIONS]

Options:
  -n, --name <NAME>    Name for this token [default: client]
```

### `rgpu info`

```
rgpu info [OPTIONS]

Options:
  -s, --server <SERVER>    Server address to query (host:port)
  -t, --token <TOKEN>      Authentication token
```

### `rgpu ui`

```
rgpu ui [OPTIONS]

Options:
  -s, --server <SERVER>       Server address(es) to monitor, repeatable
  -t, --token <TOKEN>         Authentication token
  -c, --config <CONFIG>       Configuration file [default: rgpu.toml]
      --poll-interval <SECS>  Metrics poll interval [default: 2]
```

## Examples

### Basic Setup: One Server, One Client

**Server machine** (has an NVIDIA GPU):
```bash
# Generate a token
rgpu token --name workstation
# Copy the token value

# Create config
cat > rgpu.toml << 'EOF'
[server]
port = 9876

[[security.tokens]]
token = "PASTE_TOKEN_HERE"
name = "workstation"
EOF

# Start server
rgpu server --config rgpu.toml
```

**Client machine** (no GPU, wants to use the remote one):
```bash
# Start client daemon
rgpu client --server 192.168.1.100:9876 --token PASTE_TOKEN_HERE

# Run a CUDA application with remote GPU
LD_PRELOAD=./librgpu_cuda_interpose.so python3 my_cuda_script.py
```

### Multi-Server Render Farm

```toml
# rgpu.toml on the client machine
[client]
gpu_ordering = "ByCapability"

[[client.servers]]
address = "render-node-1.local:9876"
token = "farm-token-abc123"

[[client.servers]]
address = "render-node-2.local:9876"
token = "farm-token-abc123"

[[client.servers]]
address = "render-node-3.local:9876"
token = "farm-token-abc123"
```

```bash
# Start client with pooled GPUs from 3 servers
rgpu client --config rgpu.toml
# Applications now see all GPUs from all 3 servers
```

### GUI-Only Mode (No CLI)

Start the GUI and configure everything from the Control panel:

```bash
rgpu ui
```

In the Control tab:
1. Fill in server configuration (port, bind, tokens)
2. Click **Start Server** to launch an embedded GPU server
3. Add remote connections via the Connections section
4. Monitor everything in GPU Overview and Metrics tabs

### QUIC Transport

For lower latency and built-in encryption:

```toml
# Server config
[server]
transport = "quic"
cert_path = "/etc/rgpu/cert.pem"
key_path = "/etc/rgpu/key.pem"

# Client config
[[client.servers]]
address = "gpu-server.local:9876"
token = "my-token"
transport = "quic"
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `RGPU_LOG` | Log level: `trace`, `debug`, `info`, `warn`, `error` |
| `VK_ICD_FILENAMES` | Override Vulkan ICD manifest path |
| `LD_PRELOAD` | Load CUDA interpose library (Linux) |

## Security

- **Authentication**: HMAC-SHA256 challenge-response with pre-shared tokens
- **Transport Encryption**: TLS 1.3 (TCP mode via rustls, QUIC mode via quinn)
- **Token Scoping**: Tokens can be restricted to specific GPUs and memory limits
- **Connection Limits**: Configurable `max_clients` per server
- **Service Hardening**: systemd units with `NoNewPrivileges`, `ProtectSystem=strict`, `ProtectHome=true`

> **Note**: In development mode (no cert/key configured), TCP connections are unencrypted. Always configure TLS certificates for production deployments.

## Supported CUDA Functions

RGPU intercepts 200+ CUDA Driver API functions, including:

- **Device Management**: `cuDeviceGet`, `cuDeviceGetCount`, `cuDeviceGetName`, `cuDeviceGetAttribute`, `cuDeviceTotalMem`, `cuDeviceGetUuid`, `cuDeviceComputeCapability`
- **Context**: `cuCtxCreate`, `cuCtxDestroy`, `cuCtxSetCurrent`, `cuCtxGetCurrent`, `cuCtxSynchronize`, `cuCtxPushCurrent`, `cuCtxPopCurrent`, primary context operations
- **Memory**: `cuMemAlloc`, `cuMemFree`, `cuMemcpyHtoD`, `cuMemcpyDtoH`, `cuMemcpyDtoD`, async variants, `cuMemsetD8/D16/D32`, host memory, managed memory, memory pools
- **Modules**: `cuModuleLoadData`, `cuModuleLoadDataEx`, `cuModuleGetFunction`, `cuModuleGetGlobal`, linker API
- **Execution**: `cuLaunchKernel`, `cuLaunchCooperativeKernel`, function attributes, occupancy queries
- **Streams**: `cuStreamCreate`, `cuStreamCreateWithPriority`, `cuStreamSynchronize`, `cuStreamWaitEvent`
- **Events**: `cuEventCreate`, `cuEventRecord`, `cuEventSynchronize`, `cuEventElapsedTime`
- **Pointer Queries**: `cuPointerGetAttribute`, `cuPointerGetAttributes`, `cuPointerSetAttribute`
- **Peer Access**: `cuCtxEnablePeerAccess`, `cuCtxDisablePeerAccess`
- **Process Address**: `cuGetProcAddress` with 253-entry dispatch table

## Supported Vulkan Functions

RGPU implements 60+ Vulkan functions as an ICD driver:

- **Instance/Device**: `vkCreateInstance`, `vkEnumeratePhysicalDevices`, `vkCreateDevice`, `vkGetDeviceQueue`
- **Memory/Buffers**: `vkAllocateMemory`, `vkMapMemory`, `vkCreateBuffer`, `vkBindBufferMemory`
- **Images**: `vkCreateImage`, `vkCreateImageView`, `vkBindImageMemory`, `vkGetImageMemoryRequirements`
- **Pipelines**: `vkCreateComputePipelines`, `vkCreateGraphicsPipelines`, `vkCreateShaderModule`, descriptor sets
- **Render Passes**: `vkCreateRenderPass`, `vkCreateFramebuffer`, `vkCmdBeginRenderPass`, `vkCmdDraw`
- **Commands**: `vkAllocateCommandBuffers`, `vkBeginCommandBuffer`, pipeline barriers, copy operations
- **Synchronization**: `vkCreateFence`, `vkCreateSemaphore`, `vkQueueSubmit`, `vkQueueWaitIdle`

## Building Installers

### Windows (NSIS)

```powershell
# Install NSIS
winget install NSIS.NSIS

# Build installer
.\packaging\windows\build-windows.ps1

# Output: packaging\windows\nsis\rgpu-0.1.0-windows-x64-setup.exe
```

### Linux (.deb + .rpm)

```bash
# Install tools
cargo install cargo-deb cargo-generate-rpm

# Build packages
./packaging/linux/build-linux.sh

# Output: target/debian/rgpu_0.1.0_amd64.deb
# Output: target/generate-rpm/rgpu-0.1.0.x86_64.rpm
```

### macOS (.pkg)

```bash
# Build on macOS (requires Xcode CLI tools)
./packaging/macos/build-macos.sh

# Output: target/rgpu-0.1.0-macos-x64.pkg
```

### Automated Releases (CI/CD)

Push a version tag to trigger cross-platform builds via GitHub Actions:

```bash
git tag v0.1.0
git push --tags
# Automatically builds Windows .exe, Linux .deb/.rpm, macOS .pkg
# and creates a GitHub Release with all artifacts
```

## License

Licensed under either of:

- [MIT License](LICENSE)
- [Apache License, Version 2.0](LICENSE)

at your option.
