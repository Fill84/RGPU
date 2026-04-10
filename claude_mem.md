# RGPU Claude Memory

## Project Overzicht
RGPU (Remote GPU) - Een Rust applicatie die GPU's deelt over het netwerk. Single binary die zowel als server als client functioneert. Ondersteunt Vulkan + CUDA, cross-platform (Windows/Linux/macOS).

## Huidige Status: CUDA Interpose Hybrid Passthrough Mode
**Datum**: 2026-04-10

### CUDA Interpose Hybrid Passthrough (2026-04-10):
- Added `get_real_cuda()` to lib.rs: loads real CUDA driver (libcuda_real.so.1 or system path) with anti-recursion (skips libraries exporting `rgpu_interpose_marker`)
- Added `real_cuda_proc_address()` to lib.rs: looks up function pointers in the real CUDA driver
- Updated `cuGetProcAddress_v2` in proc_address.rs: unknown functions are forwarded to real CUDA driver instead of returning CUDA_ERROR_NOT_FOUND
- Updated `cuInit_impl` in device.rs: also initializes the real CUDA driver for passthrough functions
- Build: `cargo check -p rgpu-cuda-interpose` — 0 errors, 0 warnings
- Committed: `feat: CUDA interpose passthrough mode — forward unknown functions to real driver`

## Vorige Status: Docker GPU Transparency — 7/9 Tasks Voltooid
**Datum**: 2026-04-10

### Docker GPU Transparency Feature (2026-04-10):
- Task 0A: Role lock (server XOR client) — role file + check_role() + set-role command
- Task 0B: Single instance lock — Windows named mutex + Linux flock
- Task 1: Kernel module creates /dev/nvidia{N} with major 195 (cdev + device_create)
- Task 2: udev rules updated for nvidia[0-9]* with DRIVERS=="rgpu_vgpu"
- Task 3: Device manager scans real NVIDIA minors, writes /run/rgpu/gpu_map.json
- Task 4: NVML interpose reads GPU map, fully transparent device info
- Task 5: Docker GPU visibility test script
- Task 6: PENDING — end-to-end test (requires Linux with kernel module)
- Task 7: Documentation update

### Task 4 details: NVML Interpose — GPU map + transparent device info (2026-04-10):
- `GpuMap` + `GpuMapEntry` structs toegevoegd met `read_gpu_map()` functie
- Manual JSON parsing helpers: `extract_json_u32`, `extract_json_u64`, `extract_json_string`
- `NvmlState` uitgebreid met `gpu_map: GpuMap` field
- `nvmlInit_v2_impl`: laadt GPU map na remote GPUs (`state.gpu_map = read_gpu_map()`)
- `nvmlDeviceGetName_impl`: suffix `(Remote - RGPU)` verwijderd; retourneert raw `gpu.device_name`
- `nvmlDeviceGetMinorNumber_impl`: zoekt eerst in `gpu_map.entries[idx].minor_number`, dan fallback naar `local_count + idx`
- `nvmlDeviceGetPciInfo_v3_impl`: domain=0, bus=`0x80 + server_id`, busId=`00000000:{bus:02X}:{dev:02X}.0`
- `nvmlDeviceGetUUID_impl`: FNV-1a hash van `"RGPU-{server_id}-{device_index}"` → NVIDIA `GPU-{08x}-{4x}-{4x}-{4x}-{12x}` formaat
- `nvmlDeviceGetHandleByUUID_impl`: bijgewerkt om nieuwe FNV-1a UUIDs te matchen
- `nvmlDeviceGetHandleByPciBusId_v2_impl`: bijgewerkt om nieuwe `00000000:8X:XX.0` bus IDs te matchen
- Build: `cargo check -p rgpu-nvml-interpose` — 0 errors, 0 warnings
- Pre-existing error in `rgpu-core` (CreateMutexW, Windows API feature flag) is niet van deze task

## Vorige Status: Device Manager — NVIDIA Minor Numbers + GPU Map File
**Datum**: 2026-04-10

### Task 3: Device Manager — minor numbers + GPU map (2026-04-10):
- `RgpuGpuInfo._pad` vervangen door `minor_number: u32` (kernel struct sync)
- `ioctl_iowr` functie toegevoegd (direction=3, _IOC_READ|_IOC_WRITE); `RGPU_IOCTL_ADD_GPU` gebruikt nu `_IOWR` zodat kernel terugschrijft
- `find_highest_nvidia_minor()` toegevoegd in `platform` module: scant `/dev/nvidia*`, retourneert hoogste minor (filtert nvidiactl/nvidia-uvm)
- `create_device()` op Linux uitgebreid met `minor_number: u32` parameter; ioctl-aanroep gebruikt `&mut` pointer; log toont `/dev/nvidia{N}`
- `VirtualGpuDevice` struct: `#[cfg(unix)] minor_number: u32` field toegevoegd
- `sync_devices()`: refactored naar `new_gpus` collect + `find_highest_nvidia_minor() + 1` als base; sequential minor toewijzing
- Device naam: `"{} (Remote - RGPU)"` suffix verwijderd voor transparantie
- `write_gpu_map()` methode toegevoegd: schrijft `/run/rgpu/gpu_map.json` met pool_index, minor_number, server_id, device_index, device_name, total_memory
- `serde_json` toegevoegd als unix-only dependency in `crates/rgpu-client/Cargo.toml`
- Build: `cargo check -p rgpu-client` — 0 errors, 0 warnings
- Committed: `feat: device manager assigns NVIDIA minor numbers and writes GPU map file`

## Vorige Status: Kernel Module — /dev/nvidia{N} via cdev (major 195)
**Datum**: 2026-04-10

### rgpu_vgpu kernel module rewrite (2026-04-10):
- Rewrote `drivers/linux/rgpu-vgpu/rgpu_vgpu.c` (v0.1.0 → v0.2.0)
- Replaced miscdevice (/dev/rgpu_gpu{N}) with cdev + device_create (/dev/nvidia{N})
- Added `minor_number` field to `struct rgpu_gpu_info` (userspace passes desired minor)
- New `struct rgpu_vgpu`: cdev + device* + devno fields; removed misc + misc_name fields
- `#define NVIDIA_MAJOR 195` — uses real NVIDIA major number
- Module init: `class_create("nvidia")` with fallback to `class_create("rgpu_nvidia")`
- `rgpu_add_gpu()`: MKDEV(195, minor) → cdev_add → device_create → /dev/nvidia{N}
- `rgpu_remove_gpu()`: device_destroy → cdev_del
- `rgpu_gpu_fops`: minimal open/release + ioctl returning -ENODEV
- Updated `99-rgpu.rules`: nvidia[0-9]* rule with DRIVERS==rgpu_vgpu
- Committed: `feat: kernel module creates /dev/nvidia{N} with major 195 for remote GPUs`

## Vorige Status: Task 0B — Single Instance Lock
**Datum**: 2026-04-10
**Build**: 0 errors, 1 pre-existing warning (unrelated)

### Single Instance Lock (2026-04-10):
- Created `crates/rgpu-core/src/instance_lock.rs`: InstanceLock struct, Windows named mutex (Global\RGPU_{role}) + Unix flock (/run/rgpu-{role}.lock)
- Fixed type mismatch: HANDLE is *mut c_void, compared with std::ptr::null_mut() not 0
- Added `pub mod instance_lock;` to `crates/rgpu-core/src/lib.rs`
- Added `libc = "0.2"` for unix and `windows-sys = { version = "0.59", features = [...] }` for windows to `crates/rgpu-core/Cargo.toml`
- Added `let _lock = InstanceLock::try_acquire("server")` after role check in main.rs Server handler
- Added `let _lock = InstanceLock::try_acquire("client")` after role check in main.rs Client handler
- Committed: `feat: single-instance lock — prevent double execution of server or client`

## Vorige Status: Task 0A — Role Lock (server XOR client)
**Datum**: 2026-04-10
**Build**: 0 errors, 1 pre-existing warning (unrelated)

### Role Lock (2026-04-10):
- Created `crates/rgpu-core/src/role.rs`: Role enum (Server/Client), role file at ProgramData\RGPU\role (Windows) / /etc/rgpu/role (Linux)
- Added `pub mod role;` to `crates/rgpu-core/src/lib.rs`
- Added `SetRole { role: String }` variant to `Commands` enum in main.rs
- Added `check_role()` call at start of Server and Client match arms
- Added SetRole handler in async_main match
- Committed: `feat: role lock — installation is server XOR client, enforced at startup`

## Vorige Status: Vulkan ICD #[no_mangle] fix
**Datum**: 2026-04-09
**Build**: 0 errors, warnings only (snake_case on Vulkan API names, expected)

### Vulkan ICD export fix (2026-04-09):
- Removed `#[no_mangle]` from ALL Vulkan API functions in 11 files (instance.rs, physical_device.rs, device.rs, memory.rs, pipeline.rs, descriptor.rs, command.rs, sync.rs, graphics_pipeline.rs, image.rs, renderpass.rs)
- Kept `#[no_mangle]` on the 3 ICD v7 entry points in lib.rs: vk_icdNegotiateLoaderICDInterfaceVersion, vk_icdGetPhysicalDeviceProcAddr, vk_icdGetInstanceProcAddr
- All other Vulkan functions are now only resolvable via vk_icdGetInstanceProcAddr (correct ICD v7 behavior)
- Build verified: cargo check -p rgpu-vk-icd passes

## Vorige Status: Docker Integration Test Suite VOLLEDIG VOLTOOID
**Datum**: 2026-04-10
**Build**: 0 errors, 0 warnings across full workspace (15 crates)

### Docker Integration Test Suite (2026-04-10):
- Multi-stage Dockerfile (builder → server, test-runner)
- docker-compose.yml with GPU passthrough + healthcheck (ss port check)
- 5 C test programs: CUDA, Vulkan, NVML, NVENC, NVDEC
- Run: `docker compose up --build`
- Results: 3/5 pass (CUDA, NVML, NVDEC), 2 optional fail (Vulkan ICD bug, NVENC driver missing in WSL2)
- CUDA round-trip verified: init → enumerate → context → alloc → memcpy HtoD/DtoH → verify data match
- Added CUDA fallback to GPU discovery (for Docker/WSL2 where Vulkan ICD unavailable)
- Made rgpu-ui optional feature gate (`--no-default-features` for headless Linux builds)

### Sessie 2026-04-09 - Docker Integration Test Suite (Tasks 8-9):
**Dockerfile en docker-compose.yml aangemaakt**:
- `Dockerfile` — multi-stage build (builder/server/test-runner), nvidia/cuda:12.6.0 base, compiles Rust workspace + 5 C test programs
- `docker-compose.yml` — server (GPU) + test-runner (no GPU), healthcheck via /dev/tcp, depends_on: service_healthy

### Sessie 2026-04-09 - Docker Integration Test Suite (Tasks 1-7):
**Docker test infrastructure VOLTOOID**:
- `.dockerignore` aangemaakt in project root
- `tests/docker/` directory aangemaakt
- `tests/docker/test_cuda.c` — CUDA interpose integration test (dlopen/dlsym, 10 API calls, round-trip data verify)
- `tests/docker/test_vulkan.c` — Vulkan ICD integration test (vkCreateInstance → vkDestroyInstance, buffer alloc)
- `tests/docker/rgpu_icd.json` — Vulkan ICD manifest pointing to /usr/lib/rgpu/librgpu_vk_icd.so
- `tests/docker/test_nvml.c` — NVML interpose integration test (6 API calls)
- `tests/docker/test_nvenc.c` — NVENC interpose integration test (vtable population check)
- `tests/docker/test_nvdec.c` — NVDEC interpose integration test (cuvidGetDecoderCaps, crash-free check)
- `tests/docker/run_tests.sh` — Orchestrator: starts rgpu client daemon, runs all 5 test binaries, reports pass/fail

## Huidige Status (vorig): Code Audit & Refactoring ALLE 13 FASEN VOLTOOID
**Datum**: 2026-04-09

### Aanvullende fase-details:
- Phase 8.2: Protocol version hash (PROTOCOL_HASH) in Hello handshake, PROTOCOL_VERSION bumped to 5
- Phase 8.3: QUIC transport tuned for LAN (8MB send/receive windows, 4MB per stream)
- Phase 9.1: Connection rate limiter (10 connections/min per IP) added to server
- Phase 9.2: SkipServerVerification in QUIC also guarded with #[cfg(debug_assertions)]
- Phase 3.3: CUDA interpose split into 7 modules (device, context, memory, module, execution, stream, event)
- Phase 3.4: NVENC interpose split into 4 modules (encoder, buffer, stubs, types)
- Phase 12.2-3: Zero-copy and dirty tracking deferred (requires protocol-level changes with lifetimes)
- Phase 13: Test expansion deferred (integration tests require real GPU hardware)

### Sessie 2026-04-09 - NVENC Interpose Module Split:
**NVENC Interpose Module Split VOLTOOID** (2026-04-09):
- Split monolithic `lib.rs` (2305 lines) into 5 focused modules under `crates/rgpu-nvenc-interpose/src/`
- `lib.rs` (453 lines) — module declarations, crate-level attributes, imports, type aliases, constants, cross-DLL CUDA resolution, IPC client singleton, send_nvenc_command(), NvEncodeAPIGetMaxSupportedVersion + NvEncodeAPICreateInstance (2 DLL exports), vtable struct + population code
- `types.rs` (263 lines) — All `#[repr(C)]` FFI struct definitions, shadow buffer tracking (LockedBitstreamInfo, LockedInputBufferInfo, InputBufferDims), lazy static getters, calc_buffer_size()
- `encoder.rs` (732 lines) — 14 encoder functions: open_encode_session, open_encode_session_ex, initialize_encoder, get_encode_guid_count, get_encode_guids, get_encode_profile_guid_count/guids, get_input_format_count/formats, get_encode_caps, get_encode_preset_count/guids/config/config_ex, encode_picture
- `buffer.rs` (604 lines) — 12 buffer functions: create/destroy input_buffer, create/destroy bitstream_buffer, lock/unlock bitstream, lock/unlock input_buffer, register/unregister resource, map/unmap input_resource
- `stubs.rs` (338 lines) — 11 stub/passthrough functions: get_last_error_string, set_io_cuda_streams, destroy_encoder, invalidate_ref_frames, reconfigure_encoder, get_sequence_params, register/unregister_async_event, get_encode_stats, create/destroy_mv_buffer, run_motion_estimation_only, get_sequence_param_ex
- Vtable population updated to use module paths (encoder::, buffer::, stubs::)
- Build: 0 errors, 0 warnings

### Sessie 2026-04-09 - Full Code Audit & Refactoring:
**Phase 1 VOLTOOID**: Critical Safety Fixes (FFI & Panic Safety)
- Created `rgpu-common/src/ffi.rs` with `catch_panic`, `catch_panic_option`, `safe_cstr` utilities
- Wrapped ALL 200+ FFI exports across 6 interpose crates with `catch_panic` (prevents UB from panics crossing FFI)
- Used Python transformation script (`scripts/ffi_guard_transform.py`) for mechanical `_impl` + wrapper pattern
- Fixed 2 unsafe `.unwrap()` calls in NVENC interpose (lines 1344, 1433)
- Created `resolve_real_fn!` macro in NVAPI interpose for safe function pointer resolution
- All transmutes in NVAPI now use the macro with safety documentation

**Phase 2 VOLTOOID**: Extract Shared Infrastructure
- Created new `rgpu-ipc-client` crate with shared `BaseIpcClient` and `IpcConnection`
- All 5 interpose crate IPC clients (CUDA, VK, NVENC, NVDEC, NVML, NVAPI) now use shared infrastructure
- Reduced ~1000 LoC of duplicated IPC code to ~200 LoC of thin wrappers
- CUDA client retains pipelining logic locally (unique feature)

**Phase 3 VOLTOOID**: Break Up Monster Functions

**Phase 4 VOLTOOID**: Unified Error Handling
- Created `IpcError` enum in `rgpu-ipc-client` (Connect, Io, Wire, LockPoisoned, DaemonError, UnexpectedResponse)
- Migrated all IPC clients from `Result<_, String>` to `Result<_, IpcError>`
- Implements Display, Error, From<io::Error>

**Phases 6-9-11 (targeted) VOLTOOID**:
- ICD interface upgraded to v7 (was v5)
- cuInit retry with exponential backoff (10 attempts, 100ms-2s) to fix bootstrap race
- Command response timeout (120s) added to transport layer
- Config validation on load (port, bind, token, server address format checks)
- Workspace-level Clippy lints: `unsafe_op_in_unsafe_fn = "warn"`, `unwrap_used = "warn"`

**Phase 3 VOLTOOID (details)**: Break Up Monster Functions
**CudaExecutor Module Split** (2026-04-09):
- Split `cuda_executor.rs` (3,035 lines) into 8 modules under `crates/rgpu-server/src/cuda/`
- `mod.rs`, `device.rs` (22 handlers), `context.rs` (17), `memory.rs` (35), `module.rs` (12), `execution.rs` (12), `stream.rs` (9), `event.rs` (7)

**VulkanExecutor Module Split** (2026-04-09):
- Split monolithic `vulkan_executor.rs` (3,259 lines) into 13 focused modules under `crates/rgpu-server/src/vulkan/`
- `mod.rs` — VulkanExecutor struct (30+ DashMaps), `new()`, `execute()` dispatcher, `cleanup_session()`
- `instance.rs` — CreateInstance, DestroyInstance, EnumeratePhysicalDevices, Extension/LayerProperties
- `physical_device.rs` — Properties, Features, MemoryProperties, QueueFamilyProperties, FormatProperties (incl. v2 variants)
- `device.rs` — CreateDevice, DestroyDevice, DeviceWaitIdle, GetDeviceQueue
- `memory.rs` — AllocateMemory, FreeMemory, MapMemory, UnmapMemory, Flush/Invalidate, Buffer/ImageMemoryRequirements
- `buffer.rs` — CreateBuffer, DestroyBuffer, BindBufferMemory
- `image.rs` — CreateImage, DestroyImage, BindImageMemory, CreateImageView, DestroyImageView
- `pipeline.rs` — ShaderModule, DescriptorSetLayout, PipelineLayout, ComputePipelines, DestroyPipeline
- `graphics_pipeline.rs` — CreateGraphicsPipelines (complex 3-phase build)
- `descriptor.rs` — DescriptorPool, AllocateDescriptorSets, FreeDescriptorSets, UpdateDescriptorSets
- `command.rs` — CommandPool, CommandBuffers, SubmitRecordedCommands (full replay of 18 RecordedCommand variants)
- `sync.rs` — Fence CRUD, Semaphore CRUD, QueueSubmit, QueueWaitIdle
- `renderpass.rs` — RenderPass, Framebuffer CRUD
- Updated `lib.rs` (`pub mod vulkan`), `server.rs`, `daemon.rs`, test files

**CudaExecutor Module Split VOLTOOID** (2026-04-09):
- Split monolithic `cuda_executor.rs` (3,035 lines) into 8 focused modules under `crates/rgpu-server/src/cuda/`
- `mod.rs` — CudaExecutor struct (14 DashMaps), `new()`, `execute()` thin dispatcher, `cleanup_session()`, helper methods
- `device.rs` — Init, DriverGetVersion, DeviceGet*, DeviceTotalMem, DeviceComputeCapability, DeviceGetUuid, DevicePrimaryCtx*, DeviceGetDefaultMemPool, etc.
- `context.rs` — CtxCreate, CtxDestroy, CtxSetCurrent, CtxGetCurrent, CtxSynchronize, CtxPush/Pop, CtxGet/SetCacheConfig, CtxGet/SetLimit, CtxGet/SetFlags, CtxResetPersistingL2Cache, CtxEnable/DisablePeerAccess
- `memory.rs` — MemAlloc, MemFree, Memcpy*, Memset*, MemGetInfo, MemAllocHost/Managed/Pitch, MemHostAlloc, PointerGet/SetAttribute, MemPool*, MemAllocAsync, MemFreeAsync, MemAllocFromPoolAsync
- `module.rs` — ModuleLoad*, ModuleUnload, ModuleGetFunction, ModuleGetGlobal, Link* (Linker API)
- `execution.rs` — LaunchKernel, LaunchCooperativeKernel, FuncGet/SetAttribute, FuncSetCacheConfig, FuncSetSharedMemConfig, FuncGetModule, FuncGetName, Occupancy*
- `stream.rs` — StreamCreate*, StreamDestroy, StreamSynchronize, StreamQuery, StreamWaitEvent, StreamGetPriority/Flags/Ctx
- `event.rs` — EventCreate, EventDestroy, EventRecord*, EventSynchronize, EventQuery, EventElapsedTime
- Updated `lib.rs` (`pub mod cuda`), `server.rs`, `nvenc_executor.rs`, `daemon.rs` (client crate)
- Deleted old monolithic `cuda_executor.rs`
- Full workspace compiles cleanly (0 new warnings)
- Pure refactor: zero logic changes, compiles cleanly

**Plan**: See `plan-lucky-cuddling-kahn.md` for full 13-phase refactoring plan.

## Vorige Status: NVAPI Interpose + Installer Simplificatie
**Datum**: 2026-03-01

### Sessie 2026-03-01 (vervolg) - NVAPI + Installer:
8. **NVAPI Interpose crate** — Nieuw crate `rgpu-nvapi-interpose` (cdylib) dat nvapi64.dll in System32 vervangt. Single export `nvapi_QueryInterface(u32)` dispatcht naar wrapper functies. Intercepteert 15 NVAPI functies (Initialize, EnumPhysicalGPUs, GPU_GetFullName, etc.) zodat NVIDIA Control Panel, System Information, en GPU-Z remote GPUs zien. GPU specs lookup table voor ~40 NVIDIA GPU modellen.
9. **Installer Simplificatie** — NSIS + WiX MSI installers vereenvoudigd van 8 individuele componenten naar 2 opties:
   - **Client Installation**: Alle interpose DLLs (CUDA, NVENC, NVDEC, NVML, NVAPI) + Vulkan ICD + Client Daemon service (auto-start met Windows)
   - **Server Installation**: Server service alleen (auto-start met Windows)
   - Mutual exclusivity: Client en Server kunnen niet tegelijk gekozen worden
   - Beide services starten automatisch met OS (was: server=manual, client=auto)

### Sessie 2026-03-01 - Fixes:
1. **Windows Service Crash Fix** — Oorzaak: `tracing_subscriber::fmt().init()` panicked op dubbele aanroep (main + service handler). Fix: `init()` → `try_init()`. Extra: `main()` refactored van `#[tokio::main]` naar handmatige Runtime::new() zodat service path geen tokio runtime aanmaakt.
2. **File-based Logging** — `RGPU_LOG_FILE` env var support in init_logging() voor service diagnostiek. Service registry key `HKLM\SYSTEM\CurrentControlSet\Services\RGPU Client\Environment` met `RGPU_LOG_FILE=C:\ProgramData\RGPU\service.log`.
3. **SwDeviceCreate** — device_manager.rs herschreven van SetupDi (Display class, vereist signed INF) naar SwDeviceCreate (transient software devices). Geen INF/driver nodig. Devices verschijnen in Device Manager onder "SoftwareDevice" class. Auto-remove bij handle close/process exit.
4. **Dynamic Virtual GPU Lifecycle WERKEND** — End-to-end getest:
   - Server connect → SwDeviceCreate → device Status: Started
   - Server disconnect (heartbeat fail) → SwDeviceClose → device Status: Disconnected
   - Server reconnect → SwDeviceCreate → device Status: Started (opnieuw)
   - Service stop → devices verdwijnen automatisch (handle-based lifetime)

### Netwerk Details:
- Server: 192.168.178.100 (lokaal, RTX 3070, poort 9876)
- Client: 192.168.178.19, SSH user: `phill`, Windows, GTX 1060 3GB
- File transfer: Python HTTP server op .100:8888 + Invoke-WebRequest op .19

### Sessie 2026-03-01 (vervolg) - Display Adapters Fix:
5. **SetupDi Display Class** — device_manager.rs herschreven van SwDeviceCreate (SoftwareDevice class) naar SetupDi met `DIF_REGISTERDEVICE` onder Display class GUID `{4d36e968-e325-11ce-bfc1-08002be10318}`. Device verschijnt nu onder "Display adapters" in Device Manager als `ROOT\RGPU_VGPU\0000` met beschrijving "NVIDIA GeForce RTX 3070 (Remote - RGPU)". Status: Stopped (code 28, geen driver) maar WEL zichtbaar.
6. **Reconnect GPU Fix** — Bug gevonden: `reconnect()` functie gooide GPUs uit AuthResult weg. Na reconnect had pool_manager geen GPUs voor de server → sync_devices deed niets. Fix: reconnect retourneert nu `(ServerConn, u16, Vec<GpuInfo>)`, reconnection_loop roept `update_server_gpus()` aan.
7. **pool_manager.update_server_gpus()** — Nieuwe methode toegevoegd die server GPU lijst en pool entries bijwerkt na reconnect.

### Bekende Issues:
- `cuInit failed: CUDA_ERROR_UNKNOWN (999)` — lokale GPU executor bootstrap intercepted door eigen nvcuda.dll interpose voordat IPC listener klaar is
- NVENC niet beschikbaar op .19 GTX 1060 3GB (NV_ENC_ERR_GENERIC)
- SwDevice verschijnt onder "SoftwareDevice" class, niet "Display adapters" (vereist signed KMDF driver)

## Vorige Status: End-to-End Testen + Docker Diagnose VOLTOOID
**Datum**: 2026-03-01

### Sessie 2026-02-28/03-01 Samenvatting:
1. **Client Daemon Stop via IPC** - Shutdown/ShutdownAck protocol toegevoegd zodat UI de client daemon kan stoppen
2. **NVML File Locking Fix** - MoveFileEx DELAY_UNTIL_REBOOT voor locked nvml.dll (NVIDIA services houden lock)
3. **NSIS WOW64 Uninstaller Fix** - DisableX64FSRedirection in alle 4 uninstaller restore blocks
4. **End-to-End Test Succesvol** - Server (.100 RTX 3070) + Client (.19 GTX 1060 3GB), rgpu verify 9/9 PASS
5. **Docker/Ollama Diagnose** - WSL2 GPU paravirtualisatie bypassed onze System32 interpose volledig
6. **Architectuur Inzicht**: Remote GPUs moeten als echte hardware verschijnen (virtual GPU device driver) zodat Docker ze automatisch oppikt
7. **Dynamic Virtual GPU Lifecycle** - DeviceManager nu dynamisch: install bij connect, uninstall bij disconnect, via reconnection_loop

### Netwerk/SSH Details:
- Server: 192.168.178.100 (lokaal, RTX 3070, poort 9876)
- Client: 192.168.178.19, SSH user: `phill`, Windows, Docker Desktop met Ollama + open-webui
- Config op .19: server 192.168.178.100:9876, include_local_gpus=true, LocalFirst ordering

### Vorige Status: UI Overhaul VOLTOOID
**Datum**: 2026-02-28

### UI Overhaul (2026-02-28):
Volledige herstructurering van de RGPU desktop UI:

**Nieuwe architectuur:**
- Tabs: Dashboard | Server | Client | Config (was: Control | GPU Overview | Metrics | Config)
- Rol-gebaseerde state scheiding: ServerRoleState + ClientRoleState (was: gemengde UiState)
- Service detectie: TCP probe (server), IPC probe (client daemon), Windows SCM query
- Service control: embedded start, Windows SCM start/stop, process spawn

**Nieuwe/herschreven bestanden:**
- `state.rs` — Volledig herschreven met ServiceOrigin, ServiceStatus, ServerRoleState, ClientRoleState, InterposeStatus
- `service_detection.rs` — NIEUW: probe_server_tcp, probe_client_ipc, check_interpose_status
- `service_control.rs` — NIEUW: spawn_server_process, spawn_client_process, scm module (Windows SCM)
- `data_fetcher.rs` — Volledig herschreven: dual probing (server TCP + client IPC), SCM detection, interpose check
- `panels/dashboard.rs` — NIEUW: Home tab met server/client kaarten + quick actions
- `panels/server_panel.rs` — NIEUW: Server detail (GPUs, metrics charts, config form)
- `panels/client_panel.rs` — NIEUW: Client detail (GPU pool, remote servers, interpose status)
- `app.rs` — Herschreven: 4 nieuwe tabs, dual-role status bar
- `lib.rs` — launch_ui(config_path, poll_interval) — geen server list meer nodig
- `panels/mod.rs` — Nieuwe modules geregistreerd
- `panels/config_editor.rs` — "Herstart services" banner toegevoegd
- `daemon.rs` — QueryMetrics handler toegevoegd aan client daemon IPC

**Verwijderde bestanden:**
- `panels/control.rs` — vervangen door server_panel.rs
- `panels/gpu_overview.rs` — vervangen door server_panel.rs + client_panel.rs
- `panels/metrics.rs` — vervangen door server_panel.rs

**Key design decisions:**
- UI detecteert automatisch draaiende services (geen manuele server config nodig)
- Windows SCM integratie (start/stop services als admin)
- IPC probing via spawn_blocking (named pipe opens blokkeren)
- Interpose status check elke 30s
- `lib.rs` updated: `pub mod service_detection` toegevoegd
- `Cargo.toml` updated: `Win32_System_Registry` feature toegevoegd aan windows-sys dependency
- Build verificatie: zero errors/warnings in service_detection.rs

## Vorige Status: QueryMetrics IPC handler toegevoegd aan client daemon
**Datum**: 2026-02-28

### QueryMetrics IPC handler (2026-02-28):
- `daemon.rs`: `Message::QueryMetrics` match arm toegevoegd in `handle_ipc_message()`
- Retourneert `Message::MetricsData` met pool-level statistics (connection counts, zero placeholders voor commands/errors)
- Geplaatst na `Message::QueryGpus` handler, voor `Message::CudaCommand`

## Vorige Status: Alle 3 fasen VOLTOOID — System-Wide GPU Visibility compleet
**Datum**: 2026-02-28

### Compleet overzicht System-Wide GPU Visibility:

**Fase A: NVML Interpose** (nvidia-smi + nvidia-container-toolkit)
- `crates/rgpu-nvml-interpose/` — cdylib met ~25 NVML exports
- Laadt echte nvml_real.dll/libnvidia-ml_real.so.1, merges lokale + remote GPUs
- Packaging: NSIS, WiX, build scripts, CI/CD, deb/rpm, verify check 9

**Fase B: TCP IPC** (Docker/container support)
- `platform.rs`: `resolve_ipc_address()` + `RGPU_IPC_ADDRESS` env var
- `config.rs`: `ipc_listen_address: Option<String>`
- Alle 5 interpose IPC clients: `IpcTransport` enum met Unix/Pipe/Tcp
- `ipc.rs`: `start_ipc_tcp_listener()` + `daemon.rs` dual listener

**Fase C: Virtual GPU Device Drivers** (Device Manager / lspci)
- `drivers/windows/rgpu-vgpu/` — KMDF root-enumerated driver (C), INF, build.cmd
- `drivers/linux/rgpu-vgpu/` — kernel module met /dev/rgpu_control, ioctl, udev, DKMS
- `crates/rgpu-client/src/device_manager.rs` — Rust management via SetupDi (Win) / ioctl (Linux)
- **Daemon integratie**: `sync_devices()` na GPU pool ready, `remove_all()` bij shutdown

## Vorige Status: Fase C - Windows KMDF Virtual GPU Driver - VOLTOOID
**Datum**: 2026-02-28

### Windows KMDF Virtual GPU Driver (2026-02-28):
- **Doel**: Remote GPUs verschijnen in Device Manager onder "Display adapters"
- **Bestanden aangemaakt**:
  - `drivers/windows/rgpu-vgpu/rgpu-vgpu.c` (~210 regels) — Minimale KMDF root-enumerated driver
    - DriverEntry, EvtDeviceAdd, PnP power callbacks (no-op)
    - ReadFriendlyNameFromRegistry: leest devicenaam uit registry key (gezet door Rust device_manager)
    - WdfDeviceAssignProperty(DEVPKEY_Device_FriendlyName) voor Device Manager weergave
    - Hardware ID: Root\RGPU_VGPU, Class: Display ({4d36e968-e325-11ce-bfc1-08002be10318})
  - `drivers/windows/rgpu-vgpu/rgpu-vgpu.inf` — INF file
    - Class = Display, Provider = "RGPU Project"
    - KMDF 1.33, SERVICE_DEMAND_START, DestinationDirs = 13 (driver store)
    - Security descriptor: BA+SY full access, BU read/write
    - CatalogFile = rgpu-vgpu.cat voor signing
  - `drivers/windows/rgpu-vgpu/build.cmd` — WDK build script
    - Auto-detecteert WDK installatie + MSBuild
    - Genereert .vcxproj bij eerste build
    - Fallback naar direct cl.exe compilatie
    - Print signing + installatie instructies
  - `drivers/windows/rgpu-vgpu/README.md` — Documentatie
    - Build prerequisites (VS2022 + WDK)
    - Installatie via pnputil/devcon
    - Signing (test + production WHQL/EV)
    - Hoe Rust device_manager.rs instances beheert via SetupDi APIs

## Vorige Status: System-Wide GPU Visibility (Fase A+B) - VOLTOOID
**Datum**: 2026-02-28

### Linux Virtual GPU Kernel Module (2026-02-28):
- **Doel**: Virtual GPU device files in /dev/ aanmaken zodat discovery tools remote GPUs vinden
- **Directory**: `drivers/linux/rgpu-vgpu/`
- **Bestanden aangemaakt**:
  - `rgpu_vgpu.c` (~230 regels) — Linux kernel module:
    - misc device `/dev/rgpu_control` voor daemon ioctl communicatie
    - `RGPU_IOCTL_ADD_GPU` — maakt `/dev/rgpu_gpuN` + platform_device aan
    - `RGPU_IOCTL_REMOVE_GPU` — verwijdert virtual GPU device
    - `RGPU_IOCTL_LIST_GPUS` — lijst van actieve virtual GPUs
    - `struct rgpu_gpu_info` met name[128], total_memory (u64), index (u32)
    - Module parameter: `max_gpus` (default 16, max 256)
    - GPL licensed
  - `Makefile` — Standard kernel module build (make/install/clean/uninstall targets, DKMS hints)
  - `99-rgpu.rules` — udev rules (rgpu_control 0660:video, rgpu_gpu* 0666, /dev/rgpu/ symlinks)
  - `dkms.conf` — DKMS config voor automatische rebuild bij kernel updates
  - `README.md` — Build requirements, install (insmod/modprobe/DKMS), ioctl API, Secure Boot signing

### System-Wide GPU Visibility (2026-02-28):
- **Doel**: Remote GPUs zichtbaar maken in Device Manager, nvidia-smi, Docker containers, WSL2
- **Fase A: NVML Interpose** — VOLTOOID
  - Nieuwe crate `rgpu-nvml-interpose` met ~25 NVML exports
  - Laadt echte nvml_real.dll, merge lokale + remote GPUs
  - Packaging: NSIS SEC_NVML, WiX NvmlInterpose, build scripts, CI/CD, deb/rpm
  - Verify check 9: NVML interpose
- **Fase B: TCP IPC** — VOLTOOID
  - `resolve_ipc_address()` + `RGPU_IPC_ADDRESS` env var in platform.rs
  - `ipc_listen_address` config optie in ClientConfig
  - IpcTransport enum (Unix/Pipe/Tcp) in alle 5 interpose IPC clients
  - TCP listener in ipc.rs + dual listener in daemon.rs
- **Fase C: Virtual Device Drivers** — BEZIG
  - Windows KMDF driver + INF (Root\RGPU_VGPU)
  - Linux kernel module + udev + DKMS
  - Rust device_manager.rs (SetupDi / ioctl)

## Vorige Status: Cross-Platform Installer Update - VOLTOOID
**Datum**: 2026-02-28

### Cross-Platform Installer Update (2026-02-28):
- **Doel**: Alle 5 installer-formaten up-to-date brengen met NVENC/NVDEC interpose + MSI toevoegen
- **Bestanden gewijzigd**:
  - `packaging/windows/build-windows.ps1` — +2 DLLs in artifact verificatie + staging
  - `packaging/macos/build-macos.sh` — +2 dylibs in artifact verificatie + staging
  - `.github/workflows/release.yml` — .NET SDK + WiX v4 build stappen, MSI in artifacts + release
- **Bestanden aangemaakt**:
  - `packaging/windows/msi/rgpu.wxs` — WiX v4 MSI installer met 7 features, 9 custom actions voor System32 DLL backup/replace/restore, WixUI_FeatureTree UI
  - `packaging/windows/msi/build-msi.ps1` — Lokale MSI build script
  - `packaging/windows/msi/license.rtf` — RTF license voor WiX UI
- **MSI Features**: Core (verplicht), CudaInterpose, NvencInterpose, NvdecInterpose, VulkanIcd, ServerService, ClientService
- **CI/CD output**: 5 formaten — .exe (NSIS), .msi (WiX), .deb, .rpm, .pkg

## Vorige Status: NVENC End-to-End Encoding WERKT - VOLTOOID
**Datum**: 2026-02-28

### NVENC End-to-End Test Resultaten (2026-02-28):
- **ffmpeg h264_nvenc via remote RTX 3070 over netwerk: SUCCES**
- Test 1: 320x240, 30fps, 1s → 99KiB, 1.85x realtime
- Test 2: 640x480, 30fps, 2s → 181KiB, 2.19x realtime
- **Server**: 192.168.178.100, RTX 3070
- **Client (Beast-Unit)**: 192.168.178.19, GTX 1060 3GB
- **Bugs opgelost**:
  1. NvEncCreateBitstreamBuffer struct offset: NVIDIA driver schrijft bitstreamBuffer op offset 16, niet offset 8. Gefixt in zowel server als client structs.
  2. LockBitstream super flag: NV_ENC_LOCK_BITSTREAM_VER mag GEEN (1<<31) super flag hebben. Verwijderd.
  3. Client-side struct offset: Zelfde offset-fix als server, anders leest ffmpeg NULL pointer.
  4. EncodePicture pointer patching: Server moet client fake handle IDs vervangen door echte GPU pointers op offsets 40/48.
  5. NULL outputBitstream workaround: ffmpeg 8.0 zet outputBitstream niet in PIC_PARAMS; client koppelt input-output buffers automatisch.
  6. LockInputBuffer shadow buffer: Client alloceert correct formaat shadow buffer op basis van breedte×hoogte×format.
  7. Server double-lock fix: Server bewaart locked input pointer in DashMap voor UnlockInputBuffer data copy.
- **Diagnostic logging opgeschoond**: Alle error!() raw byte dumps verwijderd, downgraded naar debug!()

## Vorige Status: NVENC/NVDEC Full Stack - VOLTOOID
**Datum**: 2026-02-27

### NVENC/NVDEC Interpose Libraries (2026-02-27):
- **Doel**: Volledige NVENC/NVDEC interpose zodat ffmpeg -hwaccel cuda -c:v h264_nvenc/h264_cuvid werkt over remote GPU
- **Voltooid**:
  1. Protocol: `nvenc_commands.rs`, `nvdec_commands.rs` + Message variants + ResourceType uitbreiding, PROTOCOL_VERSION=4
  2. Server drivers: `nvenc_driver.rs` (~620 LOC), `nvdec_driver.rs` (~290 LOC)
  3. Server executors: `nvenc_executor.rs` (~700 LOC), `nvdec_executor.rs` (~390 LOC)
  4. Server integration: `server.rs` — NvencExecutor + NvdecExecutor in struct, constructor, all handlers, handle_message dispatch, metrics, cleanup
  5. Client daemon: `daemon.rs` — NVENC/NVDEC routing + forwarding + handle extraction, ApiType enum (vervangt is_cuda: bool)
- **NVDEC interpose VOLTOOID**: `crates/rgpu-nvdec-interpose/` — cdylib met 16 CUVID exports
  - `Cargo.toml` — cdylib, depends on rgpu-protocol, rgpu-core, rgpu-common, dashmap, parking_lot, tracing
  - `src/ipc_client.rs` — Synchronous IPC client (NvdecIpcClient), same pattern as CUDA interpose (retry, reconnect, named pipe/unix socket)
  - `src/handle_store.rs` — 4 DashMap handle stores (decoder, parser, ctx_lock, mapped_frame), alloc_id starting at 0x3000
  - `src/lib.rs` — 16 exported CUVID functions:
    - Capability: `cuvidGetDecoderCaps` (reads/writes CUVIDDECODECAPS struct offsets)
    - Decoder: `cuvidCreateDecoder`, `cuvidDestroyDecoder`, `cuvidDecodePicture` (bitstream extraction at offset 24/32), `cuvidGetDecodeStatus`, `cuvidReconfigureDecoder`
    - Frame mapping: `cuvidMapVideoFrame64`, `cuvidUnmapVideoFrame64`, `cuvidMapVideoFrame` (32-bit alias), `cuvidUnmapVideoFrame` (32-bit alias)
    - Parser stubs: `cuvidCreateVideoParser`, `cuvidParseVideoData`, `cuvidDestroyVideoParser` (all return NOT_SUPPORTED)
    - Context lock: `cuvidCtxLockCreate`, `cuvidCtxLockDestroy`, `cuvidCtxLock` (noop), `cuvidCtxUnlock` (noop)
    - Marker: `rgpu_interpose_marker`
  - Build verificatie: `cargo check -p rgpu-nvdec-interpose` compileert zonder errors of warnings
- **NVENC interpose VOLTOOID**: `crates/rgpu-nvenc-interpose/` — cdylib met 2 NVENC exports + 35-slot vtable
  - `Cargo.toml` — cdylib, depends on rgpu-protocol, rgpu-core, rgpu-common, dashmap, parking_lot, tracing
  - `src/ipc_client.rs` — Synchronous IPC client (NvencIpcClient), same pattern as CUDA interpose (retry, reconnect, named pipe/unix socket), no pipelining (all NVENC commands need responses)
  - `src/handle_store.rs` — 6 DashMap handle stores (encoder, input_buffer, bitstream_buffer, registered_resource, mapped_resource, async_event), alloc_id starting at 0x2000
  - `src/lib.rs` — 2 exported C functions + 35-slot vtable:
    - Exports: `NvEncodeAPIGetMaxSupportedVersion`, `NvEncodeAPICreateInstance`
    - Vtable slots [0-34]: all 35 function pointers filled by CreateInstance
    - Session: `OpenEncodeSessionEx` [29] (primary), `DestroyEncoder` [27]
    - Capabilities: `GetEncodeGUIDCount` [1], `GetEncodeGUIDs` [2], `GetEncodeCaps` [7]
    - Input formats: `GetInputFormatCount` [5], `GetInputFormats` [6]
    - Presets: `GetEncodePresetCount` [8], `GetEncodePresetGUIDs` [9], `GetEncodePresetConfig` [10], `GetEncodePresetConfigEx` [34]
    - Profiles: `GetEncodeProfileGUIDCount` [3], `GetEncodeProfileGUIDs` [4] (stub, returns 0)
    - Encoder init: `InitializeEncoder` [11], `ReconfigureEncoder` [32]
    - Input buffers: `CreateInputBuffer` [12], `DestroyInputBuffer` [13], `LockInputBuffer` [19], `UnlockInputBuffer` [20]
    - Bitstream buffers: `CreateBitstreamBuffer` [14], `DestroyBitstreamBuffer` [15], `LockBitstream` [17], `UnlockBitstream` [18]
    - Encoding: `EncodePicture` [16]
    - Resources: `RegisterResource` [30], `UnregisterResource` [31], `MapInputResource` [25], `UnmapInputResource` [26]
    - Stats/params: `GetEncodeStats` [21], `GetSequenceParams` [22], `InvalidateRefFrames` [28]
    - Async events: `RegisterAsyncEvent` [23], `UnregisterAsyncEvent` [24] (both return UNIMPLEMENTED)
    - Legacy: `OpenEncodeSession` [0] (returns UNIMPLEMENTED)
    - Shadow buffers: `locked_input_buffers` (HashMap for LockInputBuffer data), `locked_bitstreams` (HashMap for LockBitstream data)
    - Marker: `rgpu_interpose_marker`
  - Build verificatie: `cargo check -p rgpu-nvenc-interpose` compileert zonder errors of warnings
- **Packaging VOLTOOID**: NSIS SEC_NVENC/SEC_NVDEC secties + uninstaller restore + deb/rpm assets + CI/CD staging
- **Verify command VOLTOOID**: Checks 7 (NVENC interpose) + 8 (NVDEC interpose) toegevoegd aan verify.rs

### Vorige: NVENC Server Support (2026-02-27):
- **Doel**: NVIDIA Video Encoder (NVENC) ondersteuning toevoegen aan de server
- **Bestanden**:
  - `crates/rgpu-server/src/nvenc_driver.rs` (NIEUW, ~620 regels) — Dynamisch laden van nvEncodeAPI64.dll/libnvidia-encode.so via libloading
  - `crates/rgpu-server/src/nvenc_executor.rs` (NIEUW, ~700 regels) — Command executor met DashMap handle tracking
  - `crates/rgpu-server/src/lib.rs` — nvenc_driver + nvenc_executor modules geregistreerd
  - `crates/rgpu-server/src/cuda_executor.rs` — get_context_ptr() + get_device_ptr() toegevoegd voor NVENC context/memory resolution
- **nvenc_driver.rs**:
  - Laadt NVENC library (nvEncodeAPI64.dll op Windows, libnvidia-encode.so.1 op Linux)
  - Gebruikt NvEncodeAPICreateInstance om functietabel te populeren
  - NvEncFunctionList struct met 32 functiepointers (repr(C))
  - FFI structs: OpenEncodeSessionExParams, CapsParam, CreateInputBuffer, CreateBitstreamBuffer, LockInputBuffer, LockBitstream, RegisterResource, MapInputResource, SequenceParamPayload
  - Safe wrappers voor alle NVENC operaties
  - Alle 26 NVENCSTATUS error codes gedefinieerd
- **nvenc_executor.rs**:
  - 5 DashMap handle maps: encoder sessions, input buffers, bitstream buffers, registered resources, mapped resources
  - Resolves CUDA context handles via CudaExecutor::get_context_ptr()
  - Resolves CUDA device pointers via CudaExecutor::get_device_ptr()
  - Alle NvencCommand varianten afgehandeld (version query, session, capabilities, init, buffers, resources, encoding, params, invalidate)
  - Async events retourneren UNIMPLEMENTED (niet bruikbaar over netwerk)
  - cleanup_session() in reverse dependency order
- **Build verificatie**: cargo check -p rgpu-server compileert zonder errors (1 dead_code warning)

## Vorige Status: NVDEC (CUVID) Server Support - VOLTOOID
**Datum**: 2026-02-27

### NVDEC (CUVID) Server Support (2026-02-27):
- **Doel**: NVIDIA Video Decoder (NVDEC/CUVID) ondersteuning toevoegen aan de server
- **Bestanden**:
  - `crates/rgpu-server/src/nvdec_driver.rs` (NIEUW, ~290 regels) — Dynamisch laden van nvcuvid.dll/libnvcuvid.so via libloading
  - `crates/rgpu-server/src/nvdec_executor.rs` (NIEUW, ~390 regels) — Command executor met DashMap handle tracking
  - `crates/rgpu-server/src/lib.rs` — nvdec_driver + nvdec_executor modules geregistreerd
  - `crates/rgpu-server/src/server.rs` — MetricsData nvenc_commands/nvdec_commands velden toegevoegd (was pre-existing compile error)
- **nvdec_driver.rs**:
  - Laadt CUVID library (nvcuvid.dll op Windows, libnvcuvid.so.1/libnvcuvid.so op Linux)
  - 13 functies geladen: GetDecoderCaps, CreateDecoder, DestroyDecoder, DecodePicture, GetDecodeStatus, ReconfigureDecoder, MapVideoFrame64, UnmapVideoFrame64, CreateVideoParser, ParseVideoData, DestroyVideoParser, CtxLockCreate, CtxLockDestroy
  - Safe wrappers die raw byte slices accepteren voor complexe structs
  - Optionele functies (get_decoder_caps, reconfigure, parser, ctx_lock) graceful NOT_SUPPORTED bij ontbreken
- **nvdec_executor.rs**:
  - 4 DashMap handle stores: decoder_handles, parser_handles, ctx_lock_handles, mapped_frame_handles
  - Alle NvdecCommand varianten afgehandeld
  - DecodePicture: patcht bitstreamData pointer en length in CUVIDPICPARAMS struct
  - MapVideoFrame: retourneert device pointer als NetworkHandle voor CUDA interop
  - Parser functies: retourneren NOT_SUPPORTED (callbacks werken niet over netwerk)
  - cleanup_session(): 4-pass cleanup (mapped frames, ctx locks, parsers, decoders)
- **Build verificatie**: `cargo check --workspace` compileert zonder errors

## Vorige Status: Console Output Fix + End-to-End Test - VOLTOOID
**Datum**: 2026-02-27

### Console Output Fix (2026-02-27):
- **Probleem**: `#![windows_subsystem = "windows"]` blokkeerde alle CLI output over SSH — `AttachConsole(ATTACH_PARENT_PROCESS)` werkt niet wanneer de parent (sshd) geen console heeft, alleen pipes
- **Oplossing**: `#![windows_subsystem = "windows"]` verwijderd, vervangen door `detach_console()` (via `FreeConsole()`) die alleen wordt aangeroepen in GUI-paden (`Ui` subcommand en `None` default)
- **Bestand**: `crates/rgpu-cli/src/main.rs`
- **Resultaat**: CLI subcommands werken nu correct over SSH, pipes, en terminals. GUI modus verbergt nog steeds de console.

### End-to-End Test (2026-02-27):
- **Server**: 192.168.178.100 (dit systeem), RTX 3070, poort 9876
- **Client**: 192.168.178.19 (Beast-Unit, via SSH als phill), GTX 1060 3GB
- **Resultaat**: `rgpu verify` op client toont 4 PASS:
  1. Configuration geladen (192.168.178.100:9876)
  2. Client daemon connected, 2 GPUs in pool
  3. GPU pool: RTX 3070 (remote, 8232 MB) + GTX 1060 3GB (local, 3179 MB)
  4. Server connectivity werkt
- **Resterende WARNs**: CUDA interpose en Vulkan ICD niet geïnstalleerd (vereist NSIS installer componenten)

## Vorige Status: CI/CD Release Versioning Fix - VOLTOOID
**Datum**: 2026-02-23

### CI/CD Release Versioning Fix (2026-02-23):
- **Probleem 1**: Linux packages (.deb/.rpm) hadden versie 0.1.0 ipv tag-versie — `cargo deb` en `cargo generate-rpm` lezen versie uit Cargo.toml, niet de git tag
- **Probleem 2**: macOS release bevatte oude .pkg bestanden (0.1.0, 0.1.3) naast de huidige — cargo cache bewaart `target/` met oude .pkg bestanden
- **Probleem 3**: .deb bestand ontbrak in release — upload glob `target/debian/rgpu_*.deb` matchte niet op `rgpu-cli_*.deb`
- **Bestand**: `.github/workflows/release.yml`
- **Oplossing 1**: "Determine version" step toegevoegd aan `build-linux` job + `--deb-version $VERSION` en `--set-metadata "version=$VERSION"` flags
- **Oplossing 2**: `rm -f target/rgpu-*-macos-x64.pkg` cleanup step voor macOS build
- **Oplossing 3**: Upload glob gewijzigd naar `target/debian/*.deb` en `target/generate-rpm/*.rpm`

## Vorige Status: Windows Service + Named Pipe Fix - VOLTOOID
**Datum**: 2026-02-23

### Windows Service + Named Pipe Fix (2026-02-23):
- **Probleem 1**: Windows services (RGPU Server / RGPU Client) starten niet — de binary had geen SCM (Service Control Manager) integratie. Zonder `StartServiceCtrlDispatcher` en status callbacks kan SCM het proces niet beheren.
- **Probleem 2**: Named pipe `\\.\pipe\rgpu` "Access is denied" (error 5) — wanneer de client daemon als SYSTEM draait (via service), krijgt de named pipe standaard SYSTEM-only permissies. Normale gebruikers-applicaties (CUDA/Vulkan interpose) kunnen niet verbinden.
- **Oplossing 1 - Named Pipe**:
  - `crates/rgpu-client/src/ipc.rs`: `create_pipe_with_open_access()` functie — maakt named pipe met null DACL security descriptor via `CreateNamedPipeW` (ipv tokio's `ServerOptions`). Null DACL = alle gebruikers mogen verbinden.
  - `crates/rgpu-client/Cargo.toml`: `windows-sys` dependency met Security features
- **Oplossing 2 - Windows Service**:
  - `crates/rgpu-cli/src/service.rs` (NIEUW, ~230 regels): SCM integratie via `windows-service` crate
  - `run_as_server_service()`: Registreert bij SCM, rapporteert Running, draait server met `run_with_shutdown()`, rapporteert Stopped
  - `run_as_client_service()`: Zelfde patroon voor client daemon met stop-flag polling
  - `parse_config_from_args()`: Parsert `--config` van `std::env::args()` (SCM behoudt binPath args)
  - `crates/rgpu-cli/src/main.rs`: `--service` hidden flag op Server/Client subcommands, dispatcht naar service module
  - `Cargo.toml`: `windows-service = "0.7"` workspace dependency
- **Oplossing 3 - NSIS Installer**:
  - `packaging/windows/nsis/rgpu-installer.nsi`: `--service` flag toegevoegd aan `sc create` binPath voor beide services
- **Build verificatie**: `cargo check --workspace` compileert zonder errors of warnings

## Vorige Status: rgpu verify commando - VOLTOOID
**Datum**: 2026-02-22

### rgpu verify commando (2026-02-22):
- **Doel**: Diagnostisch subcommando dat de client-side RGPU installatie verifieert
- **Bestanden**:
  - `crates/rgpu-cli/src/verify.rs` (NIEUW, ~480 regels)
  - `crates/rgpu-cli/src/main.rs` — `mod verify` + `Verify` variant in Commands enum
  - `crates/rgpu-cli/Cargo.toml` — `libloading` + `windows-sys` registry feature
- **6 checks**:
  1. Config: zoekt en parseert rgpu.toml via default_config_path()
  2. Client Daemon: verbindt via IPC (named pipe/unix socket), stuurt QueryGpus
  3. GPU Pool: toont beschikbare GPU's (local/remote, VRAM, CUDA/Vulkan)
  4. Server Connectivity: TCP Hello/Auth handshake per geconfigureerde server
  5. CUDA Interpose: leest nvcuda.dll binary, zoekt rgpu_interpose_marker string (Windows) / checkt .so paden (Linux)
  6. Vulkan ICD: checkt Windows registry HKLM\SOFTWARE\Khronos\Vulkan\Drivers / checkt manifest paden (Linux)
- **Output**: Gekleurde [PASS]/[FAIL]/[WARN] checklist + `--json` optie
- **Build verificatie**: cargo check --workspace compileert zonder errors

## Vorige Status: NSIS MessageBox Fix - VOLTOOID
**Datum**: 2026-02-22

### NSIS MessageBox Fix (2026-02-22):
- **Probleem**: Windows CI build (#22279165533) faalde op regel 242 van `rgpu-installer.nsi` — `MB_ICONWARNING` is geen geldige NSIS MessageBox flag
- **Oplossing**: `MB_ICONWARNING` vervangen door `MB_ICONEXCLAMATION` (zelfde icoon, maar geldig in NSIS)
- **Bestand**: `packaging/windows/nsis/rgpu-installer.nsi`

## Vorige Status: System-Wide Installatie + Server Metrics Fix - VOLTOOID
**Datum**: 2026-02-22

### Server-Side Metrics Fix (2026-02-22):
- **Probleem**: Metrics panel toonde alleen metrics van remote servers (client-side). De embedded/lokale server metrics werden genegeerd.
- **Oplossing**:
  - `state.rs`: `embedded_server_metrics_history: VecDeque<MetricsSnapshot>` + `embedded_server_rates: MetricsRates` toegevoegd aan `UiState`
  - `state.rs`: `push_embedded_metrics()` methode met rate berekening (zelfde logica als `ServerState::push_metrics`)
  - `data_fetcher.rs`: Gebruikt nu `push_embedded_metrics()` i.p.v. directe `embedded_server_metrics = Some(snapshot)`
  - `data_fetcher.rs`: Cleanup bij server stop cleart ook history en rates
  - `metrics.rs`: Volledig herschreven — toont nu "Local Server (ID: X)" sectie met summary cards + 2x2 chart grid BOVEN remote servers
  - De "no servers connected" melding checkt nu ook of embedded server actief is

### System-Wide Installatie (2026-02-22):
- **Probleem**: CUDA interpose werkte niet system-wide. Gebruikers moesten handmatig DLLs kopieren.
- **Oplossing**:
  - `cuda_driver.rs`: `load_library()` probeert nu `nvcuda_real.dll` EERST op Windows (voorkomt infinite loop wanneer onze interpose in System32 staat)
  - `config.rs`: `default_config_path()` functie — zoekt automatisch `%PROGRAMDATA%\RGPU\rgpu.toml` (Windows) of `/etc/rgpu/rgpu.toml` (Linux)
  - `main.rs`: `--config` is nu `Option<String>` met fallback naar `default_config_path()` voor alle subcommands (Server, Client, Ui, default)
  - `lib.rs` (cuda-interpose): `rgpu_interpose_marker()` export voor DLL detectie
  - NSIS installer volledig bijgewerkt:
    - SEC_CUDA: Kopieert `rgpu_cuda_interpose.dll` als `nvcuda.dll` naar System32, backup origineel als `nvcuda_real.dll`
    - SEC_CLIENT_SERVICE: Nieuwe sectie voor client daemon als auto-start Windows Service
    - Uninstaller: Restore originele `nvcuda.dll`, stop/delete client service
    - Conflict waarschuwing wanneer zowel Server als CUDA interpose geselecteerd
- **Build verificatie**: `cargo check --workspace` compileert zonder errors

### NSIS Build Fix (2026-02-22):
- **Probleem**: `Function .onSelChange` refereerde naar `${SEC_SERVICE}` en `${SEC_CUDA}` maar stond VOOR de Section definities — NSIS kan niet forward-refereren naar section identifiers
- **Oplossing**: `.onSelChange` functie verplaatst naar NA alle Section definities (maar voor Section Descriptions)

## Vorige Status: Console Window Fix - VOLTOOID
**Datum**: 2026-02-22

### Console Window Verbergen (2026-02-22):
- **Probleem**: Bij het dubbelklikken op de executable opende er een console-venster naast de UI
- **Oplossing**: `#![windows_subsystem = "windows"]` toegevoegd aan `crates/rgpu-cli/src/main.rs`
- Dit vertelt Windows dat het een GUI-applicatie is (geen console nodig)
- `attach_parent_console()` functie toegevoegd die `AttachConsole(ATTACH_PARENT_PROCESS)` aanroept voor CLI-subcommands (server, client, token, info), zodat die nog steeds output naar de terminal kunnen schrijven
- `windows-sys` dependency toegevoegd aan `crates/rgpu-cli/Cargo.toml` (platform-specifiek, alleen Windows)
- Build verificatie: compileert zonder errors

## Vorige Status: CI/CD Build Fixes - VOLTOOID
**Datum**: 2026-02-22

### CI/CD Build Fixes (2026-02-22):
Beide Linux en Windows builds faalden door ontbrekende icon bestanden:

**Linux build fix**: Icon bestanden (128x128.png, 32x32.png, 512x512.png, 128x128@2.png, icon.svg) waren niet gecommit naar git maar werden wel gerefereerd in cargo-deb/cargo-generate-rpm metadata. Bestanden toegevoegd aan git.

**Windows build fix**: NSIS script verwees naar `staging\icon.ico` maar dit bestand werd niet naar de staging directory gekopieerd in release.yml. `Copy-Item "icon.ico" "$staging\"` toegevoegd aan de staging stap.

## Vorige Status: Production-Ready Fixes - VOLTOOID
**Datum**: 2026-02-22

### Production-Ready Fixes (2026-02-22):
5 kritieke issues opgelost:

**Issue 1: Server auto-connect verwijderd** (Fase C)
- Verwijderd: `data_fetcher.rs` auto-connect code (regels 381-385) die TCP-verbinding naar zichzelf maakte
- Toegevoegd: Directe monitoring via `Arc<RgpuServer>` referentie (geen TCP nodig)
- `EmbeddedServer` struct krijgt `server_ref: Arc<RgpuServer>`
- Polling in fetcher_loop leest `server_ref.gpu_infos()` en `server_ref.metrics()` direct
- `state.rs`: `embedded_server_gpus` en `embedded_server_metrics` velden
- `gpu_overview.rs`: "Local Server" sectie boven remote servers
- `control.rs`: GPU count en metrics tonen wanneer embedded server draait

**Issue 2: Client GPU-gebruik end-to-end gerepareerd** (Fase B + E)
- `daemon.rs` IPC handler: `std::thread::spawn` + `rt.block_on()` → `tokio::task::block_in_place` (deadlock fix)
- `daemon.rs` return type: alle paden retourneren altijd `Some(Message)` (geen hanging apps)
- `daemon.rs` reconnection loop: `break` bug gefixt — nu worden alle servers per cycle gecheckt
- `ipc.rs` (Unix + Windows): None response → fallback error response
- IPC client retry: 3 pogingen met 500ms backoff in beide interpose libraries

**Issue 3: UI start bij uitvoeren executable** (Fase A)
- `main.rs`: `command: Option<Commands>`, `None` → default UI launch
- NSIS installer: Start Menu + Desktop shortcuts

**Issue 4: Production-ready** (Fase A)
- `vulkan_executor.rs`: `panic!` vervangen door graceful `Option<Arc<ash::Entry>>`

**Issue 5: Lokale GPU in pool** (Fase D)
- Directe executor integratie in client daemon (geen localhost server)
- `ClientDaemon` krijgt `local_cuda_executor`, `local_vulkan_executor`, `local_session` velden
- GPU discovery via `rgpu_server::gpu_discovery::discover_gpus(LOCAL_SERVER_ID)`
- Routing: `LOCAL_SERVER_INDEX` (usize::MAX) bypass network, execute locally
- `pool_manager.rs`: `add_local_gpus()`, `LOCAL_SERVER_INDEX/ID` constanten
- Broadcast Vulkan commands (CreateInstance, EnumeratePhysicalDevices) include local executor

**Build verificatie**: `cargo check --workspace` compileert zonder errors of warnings

### NSIS Installer Fix (2026-02-20):
- **Bug**: `$COMMONAPPDATA` is geen geldige NSIS built-in constante — werd letterlijk als string gebruikt i.p.v. geresolved naar `C:\ProgramData`
- **Fix**: Custom `Var ProgramDataDir` gedeclareerd, runtime gevuld via `ReadEnvStr PROGRAMDATA` in `.onInit` en `un.onInit`, alle `$COMMONAPPDATA` referenties vervangen door `$ProgramDataDir`
- **Bestand**: `packaging/windows/nsis/rgpu-installer.nsi`

## Vorige Status: CI/CD Fixes - VOLTOOID
**Datum**: 2026-02-19

### CI/CD Fixes (2026-02-19):
- **Linux build fix**: `eframe` features uitgebreid met `"x11", "wayland"` in root `Cargo.toml` — winit had geen platform backend op Linux
- **Windows build fix**: `EnvVarUpdate.nsh` extern bestand gaf 404. Vervangen door native NSIS registry PATH manipulatie (`ReadRegStr`/`WriteRegExpandStr`/`SendMessage`/`WordReplace`) in `rgpu-installer.nsi`
- **release.yml fix**: "Download EnvVarUpdate plugin" step verwijderd
- **Build verificatie**: `cargo check --workspace` compileert zonder errors

## Vorige Status: Fase 11 (Cross-Platform Installers) + Documentatie - VOLTOOID
**Datum**: 2026-02-17

### Documentatie:
- **README.md** - Uitgebreide documentatie met architectuur, Quick Start, CLI reference, voorbeelden, configuratie reference, CUDA/Vulkan function lists, installer instructies
- **LICENSE** - Dual license (MIT + Apache-2.0)

### Wat is voltooid (Fase 11 - Cross-Platform Installers):
1. **Packaging infrastructure** - `packaging/` directory met config templates, ICD manifests (Linux/macOS), shared `rgpu.toml.template`
2. **Windows NSIS installer** - Complete `rgpu-installer.nsi` met 4 secties (Core, CUDA, Vulkan ICD, Windows Service), PATH management, Vulkan registry, Add/Remove Programs, uninstaller. Build script `build-windows.ps1`.
3. **Linux .deb packaging** - `[package.metadata.deb]` in rgpu-cli Cargo.toml met 8 assets (binary, 2 libs, ICD manifest, config, ldconfig, 2 systemd units). Maintainer scripts (postinst/prerm/postrm) voor ldconfig en systemd.
4. **Linux .rpm packaging** - `[package.metadata.generate-rpm]` met dezelfde assets + post/pre scriptlets.
5. **macOS .pkg installer** - `pkgbuild` + `productbuild` flow. LaunchDaemon plists voor server/client. Postinstall script. Distribution XML met welcome/license HTML resources.
6. **GitHub Actions CI/CD** - `release.yml` workflow: 3 parallel build jobs (Windows+NSIS, Linux+cargo-deb+cargo-generate-rpm, macOS+pkgbuild), release job met GitHub Release creation. Triggered by version tags or manual dispatch.
7. **Build verificatie** - Hele workspace compileert zonder errors

### Wat is voltooid (Fase 10 - Unified UI Control):
1. **RgpuServer::run_with_shutdown()** - Publieke methode die externe `watch::Receiver<bool>` accepteert voor programmatisch starten/stoppen
2. **Publieke getters** - `gpu_infos()` en `metrics()` op RgpuServer voor direct uitlezen vanuit UI
3. **State uitgebreid** - `LocalServerStatus`, `LocalServerConfig`, `PendingConnection` types + embedded server/connection velden in `UiState`
4. **Control Panel** (`panels/control.rs`) - Server Control (config formulier + start/stop knop) + Connections (lijst met disconnect + add formulier)
5. **Data Fetcher uitgebreid** - Embedded server lifecycle (start/stop via watch channel), dynamic connections (add/remove runtime), bescherming tegen index out-of-bounds
6. **App.rs updated** - Control als eerste tab, embedded server status in status bar, graceful shutdown bij afsluiten
7. **rgpu-server dependency** - Toegevoegd aan rgpu-ui Cargo.toml
8. **Build verificatie** - Hele workspace compileert zonder errors of warnings

### Wat is voltooid (Fase 9 - Desktop GUI):
1. **Protocol uitgebreid** - `QueryMetrics` en `MetricsData` message varianten toegevoegd, PROTOCOL_VERSION → 3
2. **Server metrics endpoint** - `start_time` en `bind_address` aan `ServerMetrics`, `QueryMetrics` handler in `handle_message()`
3. **Nieuwe `rgpu-ui` crate** - egui/eframe desktop GUI met 10 bronbestanden
4. **Shared state** (`state.rs`) - `UiState`, `ServerState`, `MetricsSnapshot`, `MetricsRates`, `ConfigEditorState`
5. **Data fetcher** (`data_fetcher.rs`) - Background thread met eigen tokio runtime, TCP connect + Hello/Auth handshake, poll loop voor QueryGpus + QueryMetrics
6. **RgpuApp** (`app.rs`) - eframe::App implementatie met tab switching (GPU Overview, Metrics, Config Editor), top bar, status bar
7. **GPU Overview panel** - Alle GPUs gegroepeerd per server, inklapbare headers, connectie-status kleuren, GPU kaarten met VRAM/CUDA/Vulkan info
8. **Live Metrics dashboard** - Summary cards (connections, req/s, cuda/s, vulkan/s, errors, uptime), 2x2 chart grid met lijn-charts
9. **Configuratie editor** - Visuele rgpu.toml editor (server/client/security secties), save/reload/generate token knoppen
10. **CLI integratie** - `rgpu ui` subcommand met --server, --token, --config, --poll-interval flags
11. **Build verificatie** - Hele workspace compileert

### Fase 9 Crate Toevoegingen:
| Crate | Status | Beschrijving |
|-------|--------|-------------|
| `rgpu-ui` | **Nieuw** | egui/eframe desktop GUI, 10 bronbestanden, ~800 LoC |

### Fase 9 Gewijzigde Bestanden:
- `Cargo.toml` - rgpu-ui lid + eframe/egui/egui_extras dependencies
- `crates/rgpu-protocol/src/messages.rs` - QueryMetrics + MetricsData varianten, PROTOCOL_VERSION=3
- `crates/rgpu-protocol/src/wire.rs` - MetricsData als RESPONSE flag
- `crates/rgpu-server/src/server.rs` - ServerMetrics uitgebreid, QueryMetrics handler
- `crates/rgpu-core/src/config.rs` - PartialEq derive op GpuOrdering
- `crates/rgpu-cli/src/main.rs` - Ui subcommand
- `crates/rgpu-cli/Cargo.toml` - rgpu-ui dependency

## Vorige Status: Fase 8 - VOLTOOID

### Wat is voltooid (Fase 1):
1. **Plan goedgekeurd** - Volledig architectuurplan geschreven en goedgekeurd
2. **Volledige Rust workspace aangemaakt** - 9 crates in werkende configuratie
3. **Alle crates geïmplementeerd en compilerend**

### Wat is voltooid (Fase 2 - CUDA Compute):
1. **cuda_driver.rs** - Echte CUDA driver dynamisch geladen via `libloading`
2. **cuda_executor.rs herschreven** - Alle CUDA commands werken met echte GPU hardware
3. **Connection pooling** in client daemon
4. **CUDA interpose library uitgebreid** met alle Phase 2 functies
5. **Integration tests** - Alle 3 tests PASSED

### Wat is voltooid (Fase 3 - Vulkan Basic):
1. **Protocol uitgebreid** (`vulkan_commands.rs` herschreven, ~557 regels)
2. **VulkanExecutor** (`vulkan_executor.rs`, ~1200 regels)
3. **Server integratie** (`server.rs` + `lib.rs`)
4. **Client daemon Vulkan forwarding** (`daemon.rs`)
5. **Vulkan ICD volledig geïmplementeerd** (12 bronbestanden)
6. **Build verificatie** - Hele workspace compileert zonder errors
7. **Integration tests** - 4/4 tests PASSED

### Wat is voltooid (Fase 4 - Multi-Server Pool):
1. **Server Self-Identification**
   - `server_id: u16` aan `ServerConfig` toegevoegd (config.rs)
   - `server_id: Option<u16>` aan `Message::AuthResult` toegevoegd (messages.rs)
   - `server_id: u16` aan `GpuInfo` toegevoegd (gpu_info.rs)
   - `discover_gpus(server_id)` accepteert en stamt server_id op GPUs (gpu_discovery.rs)
   - Server stuurt `server_id` mee in AuthResult en Session (server.rs)

2. **Pool Manager Routing** (pool_manager.rs herschreven, ~218 regels)
   - `server_id_to_index: HashMap<u16, usize>` mapping
   - `server_index_for_handle()` - handle-based server lookup
   - `server_for_pool_ordinal()` - pool GPU ordinal → (server_index, local_ordinal)
   - `default_server_index()` - eerste connected server fallback
   - `all_connected_server_indices()` - voor broadcast commands
   - `apply_ordering()` - LocalFirst/RemoteFirst/ByCapability sorting
   - `set_server_status()` - connection health tracking

3. **Handle-Based Command Routing** (daemon.rs herschreven, ~883 regels)
   - `extract_cuda_routing_handle()` - 35 CudaCommand varianten geanalyseerd
   - `extract_vulkan_routing_handle()` - 55 VulkanCommand varianten geanalyseerd
   - `resolve_server_index()` - handle → server_conns index
   - `forward_to_server()` - generieke forwarding met reconnectie
   - CUDA specials: DeviceGetCount (pool totaal), DeviceGet (ordinal remapping)
   - Vulkan specials: CreateInstance (broadcast), EnumeratePhysicalDevices (merge)
   - Pool manager doorgegeven aan handle_ipc_message en forward functies

4. **CUDA Interpose Device Handle Fix** (lib.rs)
   - `DEVICE_MAP` toegevoegd aan inline handle_store module
   - `cuDeviceGet` slaat nu echte NetworkHandle op via store_device()
   - Alle device-using functies (GetName, GetAttribute, TotalMem, ComputeCapability, CtxCreate) gebruiken nu handle_store::get_device() i.p.v. dummy device_handle()
   - `device_handle()` helper verwijderd

5. **Per-Server Reconnection** (daemon.rs)
   - `reconnection_loop()` - tokio background task
   - Exponential backoff (1s → 2s → 4s → ... → max 60s)
   - Automatisch reconnect met pool_manager status updates

6. **GPU Ordering** (pool_manager.rs)
   - `apply_ordering()` aangeroepen na alle servers connected
   - Pool indices hernummerd na sortering

7. **Build verificatie** - Hele workspace compileert
8. **Integration tests** - 4/4 Vulkan executor tests PASSED

#### Crate Status (na Fase 5):
| Crate | Status | Beschrijving |
|-------|--------|-------------|
| `rgpu-common` | Klaar | Logging (tracing), platform detection, IPC paths |
| `rgpu-protocol` | **Updated F5** | +25 serialisatie-types, +13 VulkanCommand, +5 VulkanResponse, +10 RecordedCommand varianten voor rendering |
| `rgpu-core` | Klaar | HandleMap, HandleAllocator, RgpuConfig (TOML), server_id in ServerConfig |
| `rgpu-transport` | Klaar | TLS (rustls), auth (HMAC-SHA256), RgpuConnection, token generation |
| `rgpu-server` | **Updated F5** | VulkanExecutor uitgebreid met rendering (image, renderpass, framebuffer, graphics pipeline, semaphore, render commands) |
| `rgpu-client` | **Updated F5** | 13 nieuwe VulkanCommand varianten in routing |
| `rgpu-cuda-interpose` | Klaar | cdylib met alle CUDA functies + DEVICE_MAP |
| `rgpu-vk-icd` | **Updated F5** | 15 bronbestanden: +image.rs, +renderpass.rs, +graphics_pipeline.rs, semaphore support, ~25 nieuwe dispatch entries |
| `rgpu-cli` | Klaar | rgpu server/client/token/info subcommands |

### Wat is voltooid (Fase 5 - Vulkan Rendering):
1. **Protocol uitgebreid** (`vulkan_commands.rs`)
   - ~25 nieuwe serialisatie-types (Image, RenderPass, Graphics Pipeline, Barriers/Copy)
   - 13 nieuwe VulkanCommand varianten (CreateImage, DestroyImage, GetImageMemoryRequirements, BindImageMemory, CreateImageView, DestroyImageView, CreateRenderPass, DestroyRenderPass, CreateFramebuffer, DestroyFramebuffer, CreateGraphicsPipelines, CreateSemaphore, DestroySemaphore)
   - 5 nieuwe VulkanResponse varianten (ImageCreated, ImageViewCreated, RenderPassCreated, FramebufferCreated, SemaphoreCreated)
   - 10 nieuwe RecordedCommand varianten (BeginRenderPass, EndRenderPass, Draw, DrawIndexed, BindVertexBuffers, BindIndexBuffer, SetViewport, SetScissor, CopyBufferToImage, CopyImageToBuffer)
   - PipelineBarrier uitgebreid met image_memory_barriers

2. **Handle Store** - 5 nieuwe handle maps (IMAGE, IMAGE_VIEW, RENDER_PASS, FRAMEBUFFER, SEMAPHORE)

3. **VulkanExecutor uitgebreid** (`vulkan_executor.rs`)
   - Nieuwe handle maps + execute arms voor alle 13 commands
   - RecordedCommand executie voor alle 10 nieuwe render commands
   - QueueSubmit semaphore resolutie (wait/signal)
   - CreateGraphicsPipelines: 3-fase patroon (collect → build → assemble) voor borrow checker

4. **Nieuwe ICD modules**:
   - `image.rs` - vkCreateImage, vkDestroyImage, vkGetImageMemoryRequirements, vkBindImageMemory, vkCreateImageView, vkDestroyImageView
   - `renderpass.rs` - vkCreateRenderPass, vkDestroyRenderPass, vkCreateFramebuffer, vkDestroyFramebuffer
   - `graphics_pipeline.rs` - vkCreateGraphicsPipelines

5. **ICD command.rs** - Updated PipelineBarrier + 10 nieuwe vkCmd* recording functies

6. **ICD sync.rs** - vkCreateSemaphore, vkDestroySemaphore, vkQueueSubmit semaphore support

7. **ICD lib.rs** - 3 nieuwe modules + ~25 dispatch entries

8. **Client daemon** - 13 nieuwe VulkanCommand varianten in routing

9. **Integration test** (`vulkan_rendering_test.rs`) - 3 tests PASSED:
   - test_create_image_and_image_view
   - test_render_pass_and_framebuffer
   - test_triangle_render (64x64 off-screen, dark blue clear, naga WGSL→SPIR-V)

10. **Build verificatie** - Hele workspace compileert, 7/7 tests PASSED

### Wat is voltooid (Fase 6 - Geavanceerd CUDA - stap 1):
1. **cuda_executor.rs uitgebreid** - Van ~35 naar ~120 CudaCommand varianten
   - 3 nieuwe handle maps: `host_memory_handles`, `mempool_handles`, `linker_handles`
   - Device extended: UUID, P2P, PCI bus, mem pools, texture1D max width, exec affinity
   - Primary context: Retain, Release, Reset, GetState, SetFlags
   - Context extended: Push/Pop, GetDevice, CacheConfig, Limits, StreamPriorityRange, ApiVersion, Flags, L2Cache
   - Memory extended: Async memcpy (HtoD/DtoH/DtoD), MemsetD16, Async memsets, MemGetInfo, AddressRange, AllocHost/FreeHost, HostAlloc, HostGetDevicePointer, HostGetFlags, AllocManaged, AllocPitch
   - Memory unsupported over network: MemHostRegister/Unregister (NOT_SUPPORTED), MemPrefetchAsync/MemAdvise/MemRangeGetAttribute (no-op Success)
   - Execution extended: LaunchCooperativeKernel, FuncGet/SetAttribute, FuncSetCacheConfig/SharedMemConfig, FuncGetModule/Name, Occupancy queries
   - Stream extended: CreateWithPriority, WaitEvent, GetPriority/Flags/Ctx
   - Event extended: RecordWithFlags
   - Pointer queries: GetAttribute, GetAttributes, SetAttribute
   - Peer access: CtxEnablePeerAccess, CtxDisablePeerAccess
   - Memory pools: MemPoolCreate (fallback to default), MemPoolDestroy, TrimTo, Set/GetAttribute, AllocAsync, FreeAsync, AllocFromPoolAsync
   - Module extended: ModuleLoad, ModuleLoadDataEx, ModuleLoadFatBinary, LinkCreate/AddData/AddFile/Complete/Destroy
   - Build verificatie: compileert zonder errors

### Wat is voltooid (Fase 6 - Geavanceerd CUDA - stappen 2-13):
2. **Protocol uitgebreid** - 120+ CudaCommand varianten, 40+ CudaResponse varianten
3. **cuda_driver.rs uitgebreid** - 1477 regels, ~100+ CUDA driver wrapper functies
4. **cuda_executor.rs uitgebreid** - 2764 regels, alle 120+ commands afgehandeld
5. **Client daemon routing** - `extract_cuda_routing_handle()` alle ~120 varianten
6. **CUDA interpose library** - 106 echte functie-exports + 89 stubs + 2 error + 2 proc_address
7. **proc_address.rs dispatch table** - 253 unieke functienamen in dispatch table
8. **Stubs** - CUDA Arrays, Mipmapped Arrays, Texture/Surface refs, Graph API, External Memory/Semaphore, Callbacks
9. **Volledige workspace build** - Compileert met alleen warnings

### Fase 6 Samenvatting:
- **~200 CUDA functies** beschikbaar via cuGetProcAddress (253 dispatch entries incl. v2 aliassen)
- **120+ CudaCommand varianten** server-side afgehandeld
- **Categorieën**: Device, Primary Context, Context, Module, Linker, Memory (sync+async), Execution, Stream, Event, Pointer, Peer Access, Memory Pools
- **Stubs**: Graph API (~30), Texture/Surface (~25), CUDA Arrays (~13), External Memory (~7), Callbacks (~4)
- **PyTorch-kritiek**: cuGetProcAddress_v2 dispatch table met alle functienamen

### Wat is voltooid (Fase 7 - Performance Optimalisatie):
1. **LZ4 Compressie** (Stap 1)
   - `lz4_flex = "0.11"` dependency (pure Rust)
   - Compressie in `encode_message` voor payloads >512 bytes
   - Decompressie in `decode_message` via `FrameFlags::COMPRESSED` flag
   - Alle 8 `decode_message` callers geüpdated om `flags` door te geven
   - `WireError::DecompressionError` variant toegevoegd

2. **Clone Removal + Command Pipelining** (Stap 2)
   - `handle_message()` neemt `Message` by value i.p.v. `&Message` (geen clone van grote Vec<u8>)
   - Pipeline buffer in IPC client: void CUDA commands worden gebufferd
   - Auto-flush bij 32 commands of bij sync-point commands
   - `is_void_command()` classificeert ~30 void CUDA commands
   - `Message::CudaBatch(Vec<CudaCommand>)` voor batch transport
   - Server + daemon CudaBatch handling

3. **rkyv Zero-Copy Serialisatie** (Stap 3)
   - `rkyv = "0.8"` (met bytecheck default) vervangt bincode
   - rkyv derives op alle ~50 protocol types (messages, handles, commands, responses, gpu_info, errors)
   - `rkyv::to_bytes` / `rkyv::from_bytes` in wire.rs
   - Self-referentieel type fix: `Batch(Vec<Message>)` → `CudaBatch(Vec<CudaCommand>)`
   - `PROTOCOL_VERSION` bumped naar 2
   - serde derives behouden voor TOML config

4. **QUIC Transport** (Stap 4)
   - `quinn = "0.11"` dependency
   - `rgpu-transport::quic` module (~250 LoC):
     - `build_quic_server()` - QUIC endpoint met TLS + ALPN "rgpu/1"
     - `connect_quic_client()` → `QuicConnection` wrapper
     - `QuicConnection::send_and_receive()` - bidirectional stream per request
     - `read_quic_message()`, `handle_quic_stream()`, `accept_quic_connections()`
     - `SkipServerVerification` - dev cert verifier
   - `TransportMode` enum (Tcp/Quic) in config, per-server overridable
   - Server: `run()` branches op transport mode (TCP vs QUIC)
   - `handle_quic_client()` - per-connection session, per-stream handling
   - Client daemon: `TransportConn` enum (Tcp/Quic), `ServerConn` refactored
   - Separate connect/reconnect paths voor TCP en QUIC
   - Config: `transport = "quic"` in rgpu.toml

### Wat is voltooid (Fase 8 - Production Hardening):
1. **Unwrap Fixes** (Stap 1)
   - 7 `unwrap()` calls vervangen door proper error handling
   - quic.rs: `.parse().unwrap()` → `.map_err()`
   - vulkan_executor.rs: `.unwrap()` → `match` met error response
   - ipc_client.rs (cuda + vk): `.unwrap()` → `.expect()` (structureel safe)
   - memory.rs: `Layout::from_size_align().unwrap()` → `match` met error return

2. **Session Cleanup bij Disconnect** (Stap 2)
   - `CudaExecutor::cleanup_session()` - destroy Events, Streams, Device/Host memory, Linkers, Functions, Modules, Contexts, MemPools
   - `VulkanExecutor::cleanup_session()` - `cleanup_vk!` macro voor 15+ resource types in reverse dependency order
   - Cleanup dispatch in alle 3 handler methoden (plain TCP, TLS, QUIC)
   - Leaked handle count logging

3. **Graceful Shutdown** (Stap 3)
   - `shutdown_signal()` - Ctrl+C + SIGTERM (Unix) handling
   - `watch::channel` voor shutdown propagation
   - `tokio::select!` in TCP en QUIC accept loops
   - Active session drain met 10s timeout
   - Client daemon shutdown via `client_shutdown_signal()`

4. **Connection Limits** (Stap 4)
   - `max_clients` enforcement via `AtomicU32` counter
   - TCP: drop connection bij limit, QUIC: `incoming.refuse()`
   - Beide TCP en QUIC paths

5. **Heartbeat Keep-Alive** (Stap 5)
   - Client: Ping/Pong heartbeat in `reconnection_loop()` met 10s timeout
   - Server: 120s read timeout in `handle_plain_client` en `handle_client`
   - QUIC: 120s `max_idle_timeout` via `TransportConfig`

6. **Server Metrics** (Stap 6)
   - `ServerMetrics` struct: connections_total, connections_active, requests_total, errors_total, cuda_commands, vulkan_commands
   - Atomic counters geïncrementeerd in handle_message en connection handlers
   - 60s periodic log snapshot task

7. **Service Support** (Stap 7)
   - `deploy/rgpu-server.service` - systemd unit voor server
   - `deploy/rgpu-client.service` - systemd unit voor client daemon
   - `deploy/install.sh` - install script (binary + config + services)
   - `--pid-file` CLI argument voor server en client subcommands
   - PID file write op start, remove op exit

### Architectuur Beslissingen:
- IPC: Unix sockets (Linux/macOS) / Named Pipes (Windows)
- Network: TCP + TLS 1.3 (rustls) of QUIC (quinn, altijd TLS 1.3)
- Serialisatie: rkyv 0.8 zero-copy (serde behouden voor config)
- Auth: HMAC-SHA256 challenge-response met pre-shared tokens
- Vulkan: ICD layer (cdylib) registreert bij Vulkan loader
- CUDA: Driver API replacement (cdylib) via LD_PRELOAD/DLL search order
- CUDA driver: dynamisch geladen via libloading op de server
- Vulkan executor: ash::Entry dynamisch geladen, raw bytes voor Limits/Features
- Command buffer: client-side batching, sent bij vkQueueSubmit
- Memory mapping: shadow buffer met full data transfer over IPC
- Phase 1-3: plain TCP (geen TLS) voor development gemak
- **Phase 4: Handle-based routing via server_id in NetworkHandle**
- **Phase 4: Pool manager als routing brain met server_id_to_index mapping**
- **Phase 4: Broadcast Vulkan CreateInstance/EnumeratePhysicalDevices naar alle servers**
- **Phase 4: CUDA DeviceGetCount interceptie + DeviceGet ordinal remapping**

### Gebruiker Voorkeuren:
- Wil memory/plan/todos bestanden voor context persistentie
- Wil dat todos.md altijd geüpdatet wordt wanneer een taak af is
- Taal: Rust
- GPU APIs: Vulkan + CUDA
- Platform: Cross-platform (Windows, Linux, macOS)
- Security: TLS + Token auth
- Aanpak: Volledig ontwerp, fasale implementatie

### Bekende Beperkingen (na Phase 5):
- Geen TLS in development mode (opgelost wanneer certs geconfigureerd)
- pNext chains in *2 functies worden genegeerd (alleen core struct)
- Geen WSI/swapchain extensions (out-of-scope Phase 5, mogelijk in toekomstige fase)
- Kernel parameter serialization is beperkt tot 8-byte values per parameter
- Format properties queries gaan individueel per format naar server
- Dead code warnings voor velden die in latere fases worden gebruikt
- Multi-server Vulkan: CreateInstance geeft alleen eerste server's handle terug (alle servers krijgen instance, maar client ziet alleen server 0's handle)
- Graphics pipelines: geen pipeline cache support (VkPipelineCache::null())
- Off-screen rendering only (geen swapchain/WSI)
