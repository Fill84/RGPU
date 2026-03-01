//! NVIDIA NVAPI interposition library.
//!
//! This cdylib replaces nvapi64.dll in System32 on Windows. It intercepts
//! NVAPI GPU enumeration to add remote GPUs from the RGPU daemon, making
//! them visible in NVIDIA Control Panel, System Information, GPU-Z, and
//! any application that uses NVAPI for GPU enumeration.
//!
//! Architecture:
//!   NVIDIA Control Panel / GPU-Z / games
//!          |
//!     nvapi64.dll (this interpose)
//!       /              \
//!   nvapi64_real.dll    RGPU daemon (via IPC)
//!   (local GPUs)        (remote GPUs via QueryGpus)
//!
//! NVAPI uses a single export: `nvapi_QueryInterface(id) -> fn_ptr`.
//! All functions are obtained via this dispatch mechanism.

#![allow(non_camel_case_types, non_snake_case)]

mod ipc_client;

use std::ffi::{c_char, c_int, c_uint, c_void};
use std::sync::{Mutex, OnceLock};

use ipc_client::NvapiIpcClient;
use rgpu_protocol::gpu_info::GpuInfo;

// ── NVAPI Status Codes ──────────────────────────────────────────────────

type NvAPI_Status = c_int;

const NVAPI_OK: NvAPI_Status = 0;
#[allow(dead_code)]
const NVAPI_ERROR: NvAPI_Status = -1;
const NVAPI_NO_IMPLEMENTATION: NvAPI_Status = -3;
#[allow(dead_code)]
const NVAPI_API_NOT_INITIALIZED: NvAPI_Status = -4;
const NVAPI_INVALID_ARGUMENT: NvAPI_Status = -5;
const NVAPI_NVIDIA_DEVICE_NOT_FOUND: NvAPI_Status = -6;
#[allow(dead_code)]
const NVAPI_INVALID_HANDLE: NvAPI_Status = -8;

// ── NVAPI Types ─────────────────────────────────────────────────────────

type NvPhysicalGpuHandle = *mut c_void;
type NvLogicalGpuHandle = *mut c_void;

const NVAPI_MAX_PHYSICAL_GPUS: usize = 64;
const NVAPI_MAX_LOGICAL_GPUS: usize = 64;
const NVAPI_SHORT_STRING_MAX: usize = 64;

// ── NVAPI Function IDs (from NVIDIA open-source NVAPI SDK) ──────────────

const NVAPI_INITIALIZE: u32 = 0x0150E828;
const NVAPI_UNLOAD: u32 = 0xD22BDD7E;
const NVAPI_ENUM_PHYSICAL_GPUS: u32 = 0xE5AC921F;
const NVAPI_ENUM_LOGICAL_GPUS: u32 = 0x48B3EA59;
const NVAPI_GPU_GET_FULL_NAME: u32 = 0xCEEE8E9F;
const NVAPI_GPU_GET_PHYSICAL_FRAME_BUFFER_SIZE: u32 = 0x46FBEB03;
const NVAPI_GPU_GET_VIRTUAL_FRAME_BUFFER_SIZE: u32 = 0x5A04B644;
const NVAPI_GPU_GET_GPU_CORE_COUNT: u32 = 0xC7026A87;
const NVAPI_GPU_GET_ALL_CLOCK_FREQUENCIES: u32 = 0xDCB616C3;
const NVAPI_GPU_GET_THERMAL_SETTINGS: u32 = 0xE3640A56;
const NVAPI_GPU_GET_MEMORY_INFO: u32 = 0x774AA982;
const NVAPI_GPU_GET_BUS_TYPE: u32 = 0x1BB18724;
const NVAPI_GPU_GET_BUS_ID: u32 = 0x1BE0B8E5;
const NVAPI_GPU_GET_PCI_IDENTIFIERS: u32 = 0x2DDFB66E;
const NVAPI_GPU_GET_VBIOS_VERSION_STRING: u32 = 0xA561FD7D;
// System-level function — passes through to real NVAPI via default match arm
#[allow(dead_code)]
const NVAPI_SYS_GET_DRIVER_AND_BRANCH_VERSION: u32 = 0x2926AAAD;

// ── NVAPI Clock Domain Indices ──────────────────────────────────────────

const NVAPI_GPU_PUBLIC_CLOCK_GRAPHICS: usize = 0;
const NVAPI_GPU_PUBLIC_CLOCK_MEMORY: usize = 4;
const NVAPI_GPU_PUBLIC_CLOCK_PROCESSOR: usize = 7;
const NVAPI_MAX_GPU_PUBLIC_CLOCKS: usize = 32;

// ── NVAPI Structs ───────────────────────────────────────────────────────

#[repr(C)]
struct NvGpuClockDomain {
    flags: u32,         // bit 0 = bIsPresent
    frequency_khz: u32,
}

#[repr(C)]
struct NvGpuClockFrequencies {
    version: u32,
    clock_type_flags: u32,  // bits 0-1 = clock type (0=current, 1=base, 2=boost)
    domain: [NvGpuClockDomain; NVAPI_MAX_GPU_PUBLIC_CLOCKS],
}

#[repr(C)]
struct NvGpuThermalSensor {
    controller: c_int,
    default_min_temp: u32,
    default_max_temp: u32,
    current_temp: u32,
    target: c_int,
}

#[repr(C)]
struct NvGpuThermalSettings {
    version: u32,
    count: u32,
    sensor: [NvGpuThermalSensor; 3],
}

/// NV_DISPLAY_DRIVER_MEMORY_INFO — v3 layout (10 x u32 = 40 bytes)
#[repr(C)]
struct NvDisplayDriverMemoryInfo {
    version: u32,
    dedicated_video_memory: u32,            // KB
    available_dedicated_video_memory: u32,   // KB
    system_video_memory: u32,               // KB
    shared_system_memory: u32,              // KB
    cur_available_dedicated_video_memory: u32, // KB
    dedicated_video_memory_evictions_size: u32,
    dedicated_video_memory_eviction_count: u32,
    dedicated_video_memory_promotions_size: u32,
    dedicated_video_memory_promotion_count: u32,
}

// ── GPU Specs Lookup ────────────────────────────────────────────────────

#[allow(dead_code)]
struct GpuSpecs {
    cuda_cores: u32,
    base_clock_mhz: u32,
    boost_clock_mhz: u32,
    memory_clock_mhz: u32,
    memory_bus_width: u32, // bits (e.g. 256 for 256-bit)
}

fn lookup_gpu_specs(name: &str) -> GpuSpecs {
    let n = name.to_lowercase();

    // Blackwell (SM 10.x)
    if n.contains("rtx 5090") { return GpuSpecs { cuda_cores: 21760, base_clock_mhz: 2010, boost_clock_mhz: 2407, memory_clock_mhz: 14000, memory_bus_width: 512 }; }
    if n.contains("rtx 5080") { return GpuSpecs { cuda_cores: 10752, base_clock_mhz: 2300, boost_clock_mhz: 2617, memory_clock_mhz: 14000, memory_bus_width: 256 }; }
    if n.contains("rtx 5070 ti") { return GpuSpecs { cuda_cores: 8960, base_clock_mhz: 2162, boost_clock_mhz: 2452, memory_clock_mhz: 14000, memory_bus_width: 256 }; }
    if n.contains("rtx 5070") { return GpuSpecs { cuda_cores: 6144, base_clock_mhz: 2162, boost_clock_mhz: 2512, memory_clock_mhz: 14000, memory_bus_width: 192 }; }

    // Ada Lovelace (SM 8.9, 128 cores/SM)
    if n.contains("rtx 4090") { return GpuSpecs { cuda_cores: 16384, base_clock_mhz: 2235, boost_clock_mhz: 2520, memory_clock_mhz: 10501, memory_bus_width: 384 }; }
    if n.contains("rtx 4080 super") { return GpuSpecs { cuda_cores: 10240, base_clock_mhz: 2295, boost_clock_mhz: 2550, memory_clock_mhz: 11600, memory_bus_width: 256 }; }
    if n.contains("rtx 4080") { return GpuSpecs { cuda_cores: 9728, base_clock_mhz: 2205, boost_clock_mhz: 2505, memory_clock_mhz: 11200, memory_bus_width: 256 }; }
    if n.contains("rtx 4070 ti super") { return GpuSpecs { cuda_cores: 8448, base_clock_mhz: 2340, boost_clock_mhz: 2610, memory_clock_mhz: 10501, memory_bus_width: 256 }; }
    if n.contains("rtx 4070 ti") { return GpuSpecs { cuda_cores: 7680, base_clock_mhz: 2310, boost_clock_mhz: 2610, memory_clock_mhz: 10501, memory_bus_width: 192 }; }
    if n.contains("rtx 4070 super") { return GpuSpecs { cuda_cores: 7168, base_clock_mhz: 1980, boost_clock_mhz: 2475, memory_clock_mhz: 10501, memory_bus_width: 192 }; }
    if n.contains("rtx 4070") { return GpuSpecs { cuda_cores: 5888, base_clock_mhz: 1920, boost_clock_mhz: 2475, memory_clock_mhz: 10501, memory_bus_width: 192 }; }
    if n.contains("rtx 4060 ti") { return GpuSpecs { cuda_cores: 4352, base_clock_mhz: 2310, boost_clock_mhz: 2535, memory_clock_mhz: 9001, memory_bus_width: 128 }; }
    if n.contains("rtx 4060") { return GpuSpecs { cuda_cores: 3072, base_clock_mhz: 1830, boost_clock_mhz: 2460, memory_clock_mhz: 8501, memory_bus_width: 128 }; }

    // Ampere (SM 8.x, 128 cores/SM for consumer)
    if n.contains("rtx 3090 ti") { return GpuSpecs { cuda_cores: 10752, base_clock_mhz: 1560, boost_clock_mhz: 1860, memory_clock_mhz: 10751, memory_bus_width: 384 }; }
    if n.contains("rtx 3090") { return GpuSpecs { cuda_cores: 10496, base_clock_mhz: 1395, boost_clock_mhz: 1695, memory_clock_mhz: 9751, memory_bus_width: 384 }; }
    if n.contains("rtx 3080 ti") { return GpuSpecs { cuda_cores: 10240, base_clock_mhz: 1365, boost_clock_mhz: 1665, memory_clock_mhz: 9501, memory_bus_width: 384 }; }
    if n.contains("rtx 3080") { return GpuSpecs { cuda_cores: 8704, base_clock_mhz: 1440, boost_clock_mhz: 1710, memory_clock_mhz: 9501, memory_bus_width: 320 }; }
    if n.contains("rtx 3070 ti") { return GpuSpecs { cuda_cores: 6144, base_clock_mhz: 1580, boost_clock_mhz: 1770, memory_clock_mhz: 9501, memory_bus_width: 256 }; }
    if n.contains("rtx 3070") { return GpuSpecs { cuda_cores: 5888, base_clock_mhz: 1500, boost_clock_mhz: 1725, memory_clock_mhz: 7001, memory_bus_width: 256 }; }
    if n.contains("rtx 3060 ti") { return GpuSpecs { cuda_cores: 4864, base_clock_mhz: 1410, boost_clock_mhz: 1665, memory_clock_mhz: 7001, memory_bus_width: 256 }; }
    if n.contains("rtx 3060") { return GpuSpecs { cuda_cores: 3584, base_clock_mhz: 1320, boost_clock_mhz: 1777, memory_clock_mhz: 7501, memory_bus_width: 192 }; }
    if n.contains("rtx 3050") { return GpuSpecs { cuda_cores: 2560, base_clock_mhz: 1552, boost_clock_mhz: 1777, memory_clock_mhz: 7001, memory_bus_width: 128 }; }

    // Turing (SM 7.5, 64 cores/SM)
    if n.contains("rtx 2080 ti") { return GpuSpecs { cuda_cores: 4352, base_clock_mhz: 1350, boost_clock_mhz: 1545, memory_clock_mhz: 7001, memory_bus_width: 352 }; }
    if n.contains("rtx 2080 super") { return GpuSpecs { cuda_cores: 3072, base_clock_mhz: 1650, boost_clock_mhz: 1815, memory_clock_mhz: 7751, memory_bus_width: 256 }; }
    if n.contains("rtx 2080") { return GpuSpecs { cuda_cores: 2944, base_clock_mhz: 1515, boost_clock_mhz: 1710, memory_clock_mhz: 7001, memory_bus_width: 256 }; }
    if n.contains("rtx 2070 super") { return GpuSpecs { cuda_cores: 2560, base_clock_mhz: 1605, boost_clock_mhz: 1770, memory_clock_mhz: 7001, memory_bus_width: 256 }; }
    if n.contains("rtx 2070") { return GpuSpecs { cuda_cores: 2304, base_clock_mhz: 1410, boost_clock_mhz: 1620, memory_clock_mhz: 7001, memory_bus_width: 256 }; }
    if n.contains("rtx 2060 super") { return GpuSpecs { cuda_cores: 2176, base_clock_mhz: 1470, boost_clock_mhz: 1650, memory_clock_mhz: 7001, memory_bus_width: 256 }; }
    if n.contains("rtx 2060") { return GpuSpecs { cuda_cores: 1920, base_clock_mhz: 1365, boost_clock_mhz: 1680, memory_clock_mhz: 7001, memory_bus_width: 192 }; }

    // Pascal (SM 6.1, 128 cores/SM)
    if n.contains("gtx 1080 ti") { return GpuSpecs { cuda_cores: 3584, base_clock_mhz: 1480, boost_clock_mhz: 1582, memory_clock_mhz: 5505, memory_bus_width: 352 }; }
    if n.contains("gtx 1080") { return GpuSpecs { cuda_cores: 2560, base_clock_mhz: 1607, boost_clock_mhz: 1733, memory_clock_mhz: 5005, memory_bus_width: 256 }; }
    if n.contains("gtx 1070 ti") { return GpuSpecs { cuda_cores: 2432, base_clock_mhz: 1607, boost_clock_mhz: 1683, memory_clock_mhz: 4006, memory_bus_width: 256 }; }
    if n.contains("gtx 1070") { return GpuSpecs { cuda_cores: 1920, base_clock_mhz: 1506, boost_clock_mhz: 1683, memory_clock_mhz: 4006, memory_bus_width: 256 }; }
    if n.contains("gtx 1060") && n.contains("3gb") { return GpuSpecs { cuda_cores: 1152, base_clock_mhz: 1506, boost_clock_mhz: 1708, memory_clock_mhz: 4006, memory_bus_width: 192 }; }
    if n.contains("gtx 1060") { return GpuSpecs { cuda_cores: 1280, base_clock_mhz: 1506, boost_clock_mhz: 1708, memory_clock_mhz: 4006, memory_bus_width: 192 }; }
    if n.contains("gtx 1050 ti") { return GpuSpecs { cuda_cores: 768, base_clock_mhz: 1290, boost_clock_mhz: 1392, memory_clock_mhz: 3504, memory_bus_width: 128 }; }
    if n.contains("gtx 1050") { return GpuSpecs { cuda_cores: 640, base_clock_mhz: 1354, boost_clock_mhz: 1455, memory_clock_mhz: 3504, memory_bus_width: 128 }; }

    // Data center / workstation GPUs
    if n.contains("a100") { return GpuSpecs { cuda_cores: 6912, base_clock_mhz: 765, boost_clock_mhz: 1410, memory_clock_mhz: 1215, memory_bus_width: 5120 }; }
    if n.contains("a6000") || n.contains("rtx 6000") { return GpuSpecs { cuda_cores: 10752, base_clock_mhz: 1410, boost_clock_mhz: 1800, memory_clock_mhz: 8001, memory_bus_width: 384 }; }
    if n.contains("h100") { return GpuSpecs { cuda_cores: 14592, base_clock_mhz: 1095, boost_clock_mhz: 1755, memory_clock_mhz: 1593, memory_bus_width: 5120 }; }

    // Fallback: return zeros (unknown GPU)
    GpuSpecs { cuda_cores: 0, base_clock_mhz: 0, boost_clock_mhz: 0, memory_clock_mhz: 0, memory_bus_width: 0 }
}

// ── Real NVAPI ──────────────────────────────────────────────────────────

type QueryInterfaceFn = unsafe extern "C" fn(u32) -> *const c_void;

struct RealNvapi {
    _lib: libloading::Library,
    query_interface: QueryInterfaceFn,
}

unsafe impl Send for RealNvapi {}

impl RealNvapi {
    fn load() -> Result<Self, String> {
        #[cfg(target_os = "windows")]
        let lib_name = "nvapi64_real.dll";
        #[cfg(not(target_os = "windows"))]
        let lib_name = "nvapi64_real.so"; // won't exist on non-Windows

        let lib = unsafe {
            libloading::Library::new(lib_name)
                .map_err(|e| format!("failed to load real NVAPI ({}): {}", lib_name, e))?
        };

        let query_interface: QueryInterfaceFn = unsafe {
            let sym = lib
                .get::<QueryInterfaceFn>(b"nvapi_QueryInterface")
                .map_err(|e| format!("failed to find nvapi_QueryInterface: {}", e))?;
            *sym
        };

        Ok(Self { _lib: lib, query_interface })
    }

    /// Call a real NVAPI function through the dispatch table.
    unsafe fn call_fn(&self, id: u32) -> *const c_void {
        (self.query_interface)(id)
    }
}

// ── Global State ────────────────────────────────────────────────────────

const REMOTE_HANDLE_SENTINEL: usize = 0x0000_ABCD_0000_0000;

struct NvapiState {
    real_nvapi: Option<RealNvapi>,
    local_gpu_count: u32,
    local_gpu_handles: [*mut c_void; NVAPI_MAX_PHYSICAL_GPUS],
    remote_gpus: Vec<GpuInfo>,
    ipc_client: Option<NvapiIpcClient>,
    initialized: bool,
}

unsafe impl Send for NvapiState {}

static STATE: OnceLock<Mutex<NvapiState>> = OnceLock::new();

fn get_state() -> &'static Mutex<NvapiState> {
    STATE.get_or_init(|| {
        Mutex::new(NvapiState {
            real_nvapi: None,
            local_gpu_count: 0,
            local_gpu_handles: [std::ptr::null_mut(); NVAPI_MAX_PHYSICAL_GPUS],
            remote_gpus: Vec::new(),
            ipc_client: None,
            initialized: false,
        })
    })
}

fn is_remote_handle(handle: *mut c_void) -> bool {
    let val = handle as usize;
    (val & 0xFFFF_FFFF_0000_0000) == REMOTE_HANDLE_SENTINEL && val != 0
}

fn remote_index(handle: *mut c_void) -> usize {
    (handle as usize) & 0x0000_0000_FFFF_FFFF
}

fn make_remote_handle(index: usize) -> *mut c_void {
    (REMOTE_HANDLE_SENTINEL | index) as *mut c_void
}

fn write_c_string(buf: *mut c_char, buf_len: usize, s: &str) {
    if buf.is_null() || buf_len == 0 {
        return;
    }
    let max_copy = (buf_len).saturating_sub(1).min(s.len());
    unsafe {
        std::ptr::copy_nonoverlapping(s.as_ptr() as *const c_char, buf, max_copy);
        *buf.add(max_copy) = 0;
    }
}

/// Resolve IPC path from env or default.
fn resolve_ipc_path() -> String {
    if let Ok(addr) = std::env::var("RGPU_IPC_ADDRESS") {
        if !addr.is_empty() {
            return addr;
        }
    }
    rgpu_common::platform::default_ipc_path()
}

/// Check if a local GPU has the given name (to avoid double-counting).
fn has_local_gpu_with_name(state: &NvapiState, name: &str) -> bool {
    if let Some(ref real) = state.real_nvapi {
        for i in 0..state.local_gpu_count as usize {
            let handle = state.local_gpu_handles[i];
            if handle.is_null() { continue; }

            let fn_ptr = unsafe { real.call_fn(NVAPI_GPU_GET_FULL_NAME) };
            if fn_ptr.is_null() { continue; }

            type GetFullNameFn = unsafe extern "C" fn(*mut c_void, *mut c_char) -> NvAPI_Status;
            let get_full_name: GetFullNameFn = unsafe { std::mem::transmute(fn_ptr) };

            let mut buf = [0i8; NVAPI_SHORT_STRING_MAX];
            if unsafe { get_full_name(handle, buf.as_mut_ptr()) } == NVAPI_OK {
                let cstr = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) };
                if let Ok(s) = cstr.to_str() {
                    if s == name {
                        return true;
                    }
                }
            }
        }
    }
    false
}

// ── Marker for interpose detection ──────────────────────────────────────

#[no_mangle]
pub extern "C" fn rgpu_interpose_marker() -> c_int {
    1
}

// ── Main NVAPI Export ───────────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn nvapi_QueryInterface(id: u32) -> *const c_void {
    // Ensure real NVAPI is loaded (lazy, on first call)
    {
        if let Ok(mut state) = get_state().lock() {
            if state.real_nvapi.is_none() {
                match RealNvapi::load() {
                    Ok(real) => { state.real_nvapi = Some(real); }
                    Err(_e) => {
                        // Real NVAPI not available — we'll still provide remote GPUs
                    }
                }
            }
        }
    }

    match id {
        NVAPI_INITIALIZE => our_initialize as *const c_void,
        NVAPI_UNLOAD => our_unload as *const c_void,
        NVAPI_ENUM_PHYSICAL_GPUS => our_enum_physical_gpus as *const c_void,
        NVAPI_ENUM_LOGICAL_GPUS => our_enum_logical_gpus as *const c_void,
        NVAPI_GPU_GET_FULL_NAME => our_gpu_get_full_name as *const c_void,
        NVAPI_GPU_GET_PHYSICAL_FRAME_BUFFER_SIZE => our_gpu_get_physical_frame_buffer_size as *const c_void,
        NVAPI_GPU_GET_VIRTUAL_FRAME_BUFFER_SIZE => our_gpu_get_virtual_frame_buffer_size as *const c_void,
        NVAPI_GPU_GET_GPU_CORE_COUNT => our_gpu_get_core_count as *const c_void,
        NVAPI_GPU_GET_ALL_CLOCK_FREQUENCIES => our_gpu_get_all_clock_frequencies as *const c_void,
        NVAPI_GPU_GET_THERMAL_SETTINGS => our_gpu_get_thermal_settings as *const c_void,
        NVAPI_GPU_GET_MEMORY_INFO => our_gpu_get_memory_info as *const c_void,
        NVAPI_GPU_GET_BUS_TYPE => our_gpu_get_bus_type as *const c_void,
        NVAPI_GPU_GET_BUS_ID => our_gpu_get_bus_id as *const c_void,
        NVAPI_GPU_GET_PCI_IDENTIFIERS => our_gpu_get_pci_identifiers as *const c_void,
        NVAPI_GPU_GET_VBIOS_VERSION_STRING => our_gpu_get_vbios_version_string as *const c_void,
        _ => {
            // Pass through to real NVAPI
            let state = match get_state().lock() {
                Ok(s) => s,
                Err(_) => return std::ptr::null(),
            };
            if let Some(ref real) = state.real_nvapi {
                (real.query_interface)(id)
            } else {
                std::ptr::null()
            }
        }
    }
}

// ── NvAPI_Initialize ────────────────────────────────────────────────────

unsafe extern "C" fn our_initialize() -> NvAPI_Status {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .try_init();

    let mut state = match get_state().lock() {
        Ok(s) => s,
        Err(_) => return NVAPI_ERROR,
    };

    // Load real NVAPI if not already loaded
    if state.real_nvapi.is_none() {
        if let Ok(real) = RealNvapi::load() {
            state.real_nvapi = Some(real);
        }
    }

    // Call real NvAPI_Initialize
    if let Some(ref real) = state.real_nvapi {
        let fn_ptr = real.call_fn(NVAPI_INITIALIZE);
        if !fn_ptr.is_null() {
            type InitFn = unsafe extern "C" fn() -> NvAPI_Status;
            let real_init: InitFn = std::mem::transmute(fn_ptr);
            let ret = real_init();
            if ret != NVAPI_OK {
                tracing::warn!("real NvAPI_Initialize returned {}", ret);
            }
        }

        // Enumerate local GPUs
        let fn_ptr = real.call_fn(NVAPI_ENUM_PHYSICAL_GPUS);
        if !fn_ptr.is_null() {
            type EnumFn = unsafe extern "C" fn(*mut NvPhysicalGpuHandle, *mut u32) -> NvAPI_Status;
            let real_enum: EnumFn = std::mem::transmute(fn_ptr);

            let mut count: u32 = 0;
            let ret = real_enum(state.local_gpu_handles.as_mut_ptr(), &mut count);
            if ret == NVAPI_OK {
                state.local_gpu_count = count;
                tracing::info!("NVAPI interpose: {} local GPU(s) found", count);
            }
        }
    }

    // Connect to RGPU daemon for remote GPUs
    let ipc_path = resolve_ipc_path();
    let client = NvapiIpcClient::new(&ipc_path);
    match client.query_gpus() {
        Ok(gpus) => {
            let remote: Vec<GpuInfo> = gpus
                .into_iter()
                .filter(|g| g.server_id != 0 || !has_local_gpu_with_name(&state, &g.device_name))
                .collect();
            tracing::info!("NVAPI interpose: {} remote GPU(s) from daemon", remote.len());
            state.remote_gpus = remote;
        }
        Err(e) => {
            tracing::warn!("could not query RGPU daemon: {} -- only local GPUs visible", e);
        }
    }
    state.ipc_client = Some(client);
    state.initialized = true;

    NVAPI_OK
}

// ── NvAPI_Unload ────────────────────────────────────────────────────────

unsafe extern "C" fn our_unload() -> NvAPI_Status {
    let mut state = match get_state().lock() {
        Ok(s) => s,
        Err(_) => return NVAPI_ERROR,
    };

    if let Some(ref real) = state.real_nvapi {
        let fn_ptr = real.call_fn(NVAPI_UNLOAD);
        if !fn_ptr.is_null() {
            type UnloadFn = unsafe extern "C" fn() -> NvAPI_Status;
            let real_unload: UnloadFn = std::mem::transmute(fn_ptr);
            real_unload();
        }
    }

    state.remote_gpus.clear();
    state.local_gpu_count = 0;
    state.local_gpu_handles = [std::ptr::null_mut(); NVAPI_MAX_PHYSICAL_GPUS];
    state.ipc_client = None;
    state.initialized = false;

    NVAPI_OK
}

// ── NvAPI_EnumPhysicalGPUs ──────────────────────────────────────────────

unsafe extern "C" fn our_enum_physical_gpus(
    handles: *mut NvPhysicalGpuHandle,
    count: *mut u32,
) -> NvAPI_Status {
    if handles.is_null() || count.is_null() {
        return NVAPI_INVALID_ARGUMENT;
    }

    let state = match get_state().lock() {
        Ok(s) => s,
        Err(_) => return NVAPI_ERROR,
    };

    let mut idx = 0usize;

    // Local GPUs
    for i in 0..state.local_gpu_count as usize {
        if idx >= NVAPI_MAX_PHYSICAL_GPUS { break; }
        *handles.add(idx) = state.local_gpu_handles[i];
        idx += 1;
    }

    // Remote GPUs
    for i in 0..state.remote_gpus.len() {
        if idx >= NVAPI_MAX_PHYSICAL_GPUS { break; }
        *handles.add(idx) = make_remote_handle(i);
        idx += 1;
    }

    *count = idx as u32;
    NVAPI_OK
}

// ── NvAPI_EnumLogicalGPUs ───────────────────────────────────────────────

unsafe extern "C" fn our_enum_logical_gpus(
    handles: *mut NvLogicalGpuHandle,
    count: *mut u32,
) -> NvAPI_Status {
    if handles.is_null() || count.is_null() {
        return NVAPI_INVALID_ARGUMENT;
    }

    let state = match get_state().lock() {
        Ok(s) => s,
        Err(_) => return NVAPI_ERROR,
    };

    let mut idx = 0usize;

    // Get real logical GPUs
    if let Some(ref real) = state.real_nvapi {
        let fn_ptr = real.call_fn(NVAPI_ENUM_LOGICAL_GPUS);
        if !fn_ptr.is_null() {
            type EnumLogicalFn = unsafe extern "C" fn(*mut NvLogicalGpuHandle, *mut u32) -> NvAPI_Status;
            let real_enum: EnumLogicalFn = std::mem::transmute(fn_ptr);
            let mut local_count: u32 = 0;
            if real_enum(handles, &mut local_count) == NVAPI_OK {
                idx = local_count as usize;
            }
        }
    }

    // Add remote GPUs as logical GPUs too
    for i in 0..state.remote_gpus.len() {
        if idx >= NVAPI_MAX_LOGICAL_GPUS { break; }
        *handles.add(idx) = make_remote_handle(i);
        idx += 1;
    }

    *count = idx as u32;
    NVAPI_OK
}

// ── NvAPI_GPU_GetFullName ───────────────────────────────────────────────

unsafe extern "C" fn our_gpu_get_full_name(
    handle: NvPhysicalGpuHandle,
    name: *mut c_char,
) -> NvAPI_Status {
    if name.is_null() {
        return NVAPI_INVALID_ARGUMENT;
    }

    if is_remote_handle(handle) {
        let state = match get_state().lock() {
            Ok(s) => s,
            Err(_) => return NVAPI_ERROR,
        };
        let idx = remote_index(handle);
        if let Some(gpu) = state.remote_gpus.get(idx) {
            let display_name = format!("{} (Remote - RGPU)", gpu.device_name);
            write_c_string(name, NVAPI_SHORT_STRING_MAX, &display_name);
            return NVAPI_OK;
        }
        return NVAPI_NVIDIA_DEVICE_NOT_FOUND;
    }

    // Pass through to real NVAPI
    let state = match get_state().lock() {
        Ok(s) => s,
        Err(_) => return NVAPI_ERROR,
    };
    if let Some(ref real) = state.real_nvapi {
        let fn_ptr = real.call_fn(NVAPI_GPU_GET_FULL_NAME);
        if !fn_ptr.is_null() {
            type Fn = unsafe extern "C" fn(NvPhysicalGpuHandle, *mut c_char) -> NvAPI_Status;
            let real_fn: Fn = std::mem::transmute(fn_ptr);
            return real_fn(handle, name);
        }
    }
    NVAPI_NO_IMPLEMENTATION
}

// ── NvAPI_GPU_GetPhysicalFrameBufferSize ────────────────────────────────

unsafe extern "C" fn our_gpu_get_physical_frame_buffer_size(
    handle: NvPhysicalGpuHandle,
    size_kb: *mut u32,
) -> NvAPI_Status {
    if size_kb.is_null() {
        return NVAPI_INVALID_ARGUMENT;
    }

    if is_remote_handle(handle) {
        let state = match get_state().lock() {
            Ok(s) => s,
            Err(_) => return NVAPI_ERROR,
        };
        let idx = remote_index(handle);
        if let Some(gpu) = state.remote_gpus.get(idx) {
            *size_kb = (gpu.total_memory / 1024) as u32;
            return NVAPI_OK;
        }
        return NVAPI_NVIDIA_DEVICE_NOT_FOUND;
    }

    let state = match get_state().lock() {
        Ok(s) => s,
        Err(_) => return NVAPI_ERROR,
    };
    if let Some(ref real) = state.real_nvapi {
        let fn_ptr = real.call_fn(NVAPI_GPU_GET_PHYSICAL_FRAME_BUFFER_SIZE);
        if !fn_ptr.is_null() {
            type Fn = unsafe extern "C" fn(NvPhysicalGpuHandle, *mut u32) -> NvAPI_Status;
            let real_fn: Fn = std::mem::transmute(fn_ptr);
            return real_fn(handle, size_kb);
        }
    }
    NVAPI_NO_IMPLEMENTATION
}

// ── NvAPI_GPU_GetVirtualFrameBufferSize ─────────────────────────────────

unsafe extern "C" fn our_gpu_get_virtual_frame_buffer_size(
    handle: NvPhysicalGpuHandle,
    size_kb: *mut u32,
) -> NvAPI_Status {
    if size_kb.is_null() {
        return NVAPI_INVALID_ARGUMENT;
    }

    if is_remote_handle(handle) {
        let state = match get_state().lock() {
            Ok(s) => s,
            Err(_) => return NVAPI_ERROR,
        };
        let idx = remote_index(handle);
        if let Some(gpu) = state.remote_gpus.get(idx) {
            *size_kb = (gpu.total_memory / 1024) as u32;
            return NVAPI_OK;
        }
        return NVAPI_NVIDIA_DEVICE_NOT_FOUND;
    }

    let state = match get_state().lock() {
        Ok(s) => s,
        Err(_) => return NVAPI_ERROR,
    };
    if let Some(ref real) = state.real_nvapi {
        let fn_ptr = real.call_fn(NVAPI_GPU_GET_VIRTUAL_FRAME_BUFFER_SIZE);
        if !fn_ptr.is_null() {
            type Fn = unsafe extern "C" fn(NvPhysicalGpuHandle, *mut u32) -> NvAPI_Status;
            let real_fn: Fn = std::mem::transmute(fn_ptr);
            return real_fn(handle, size_kb);
        }
    }
    NVAPI_NO_IMPLEMENTATION
}

// ── NvAPI_GPU_GetGpuCoreCount ───────────────────────────────────────────

unsafe extern "C" fn our_gpu_get_core_count(
    handle: NvPhysicalGpuHandle,
    count: *mut u32,
) -> NvAPI_Status {
    if count.is_null() {
        return NVAPI_INVALID_ARGUMENT;
    }

    if is_remote_handle(handle) {
        let state = match get_state().lock() {
            Ok(s) => s,
            Err(_) => return NVAPI_ERROR,
        };
        let idx = remote_index(handle);
        if let Some(gpu) = state.remote_gpus.get(idx) {
            let specs = lookup_gpu_specs(&gpu.device_name);
            *count = specs.cuda_cores;
            return NVAPI_OK;
        }
        return NVAPI_NVIDIA_DEVICE_NOT_FOUND;
    }

    let state = match get_state().lock() {
        Ok(s) => s,
        Err(_) => return NVAPI_ERROR,
    };
    if let Some(ref real) = state.real_nvapi {
        let fn_ptr = real.call_fn(NVAPI_GPU_GET_GPU_CORE_COUNT);
        if !fn_ptr.is_null() {
            type Fn = unsafe extern "C" fn(NvPhysicalGpuHandle, *mut u32) -> NvAPI_Status;
            let real_fn: Fn = std::mem::transmute(fn_ptr);
            return real_fn(handle, count);
        }
    }
    NVAPI_NO_IMPLEMENTATION
}

// ── NvAPI_GPU_GetAllClockFrequencies ────────────────────────────────────

unsafe extern "C" fn our_gpu_get_all_clock_frequencies(
    handle: NvPhysicalGpuHandle,
    clk_freqs: *mut NvGpuClockFrequencies,
) -> NvAPI_Status {
    if clk_freqs.is_null() {
        return NVAPI_INVALID_ARGUMENT;
    }

    if is_remote_handle(handle) {
        let state = match get_state().lock() {
            Ok(s) => s,
            Err(_) => return NVAPI_ERROR,
        };
        let idx = remote_index(handle);
        if let Some(gpu) = state.remote_gpus.get(idx) {
            let specs = lookup_gpu_specs(&gpu.device_name);
            let clock_type = (*clk_freqs).clock_type_flags & 0x3;

            let (graphics_mhz, memory_mhz) = match clock_type {
                1 => (specs.base_clock_mhz, specs.memory_clock_mhz),  // base
                2 => (specs.boost_clock_mhz, specs.memory_clock_mhz), // boost
                _ => (specs.boost_clock_mhz, specs.memory_clock_mhz), // current (use boost)
            };

            // Zero all domains first
            for i in 0..NVAPI_MAX_GPU_PUBLIC_CLOCKS {
                (*clk_freqs).domain[i].flags = 0;
                (*clk_freqs).domain[i].frequency_khz = 0;
            }

            // Graphics clock
            if graphics_mhz > 0 {
                (*clk_freqs).domain[NVAPI_GPU_PUBLIC_CLOCK_GRAPHICS].flags = 1; // bIsPresent
                (*clk_freqs).domain[NVAPI_GPU_PUBLIC_CLOCK_GRAPHICS].frequency_khz = graphics_mhz * 1000;
            }

            // Memory clock
            if memory_mhz > 0 {
                (*clk_freqs).domain[NVAPI_GPU_PUBLIC_CLOCK_MEMORY].flags = 1;
                (*clk_freqs).domain[NVAPI_GPU_PUBLIC_CLOCK_MEMORY].frequency_khz = memory_mhz * 1000;
            }

            // Processor clock (same as graphics for NVIDIA GPUs)
            if graphics_mhz > 0 {
                (*clk_freqs).domain[NVAPI_GPU_PUBLIC_CLOCK_PROCESSOR].flags = 1;
                (*clk_freqs).domain[NVAPI_GPU_PUBLIC_CLOCK_PROCESSOR].frequency_khz = graphics_mhz * 1000;
            }

            return NVAPI_OK;
        }
        return NVAPI_NVIDIA_DEVICE_NOT_FOUND;
    }

    let state = match get_state().lock() {
        Ok(s) => s,
        Err(_) => return NVAPI_ERROR,
    };
    if let Some(ref real) = state.real_nvapi {
        let fn_ptr = real.call_fn(NVAPI_GPU_GET_ALL_CLOCK_FREQUENCIES);
        if !fn_ptr.is_null() {
            type Fn = unsafe extern "C" fn(NvPhysicalGpuHandle, *mut NvGpuClockFrequencies) -> NvAPI_Status;
            let real_fn: Fn = std::mem::transmute(fn_ptr);
            return real_fn(handle, clk_freqs);
        }
    }
    NVAPI_NO_IMPLEMENTATION
}

// ── NvAPI_GPU_GetThermalSettings ────────────────────────────────────────

unsafe extern "C" fn our_gpu_get_thermal_settings(
    handle: NvPhysicalGpuHandle,
    sensor_index: c_uint,
    thermal_settings: *mut NvGpuThermalSettings,
) -> NvAPI_Status {
    if thermal_settings.is_null() {
        return NVAPI_INVALID_ARGUMENT;
    }

    if is_remote_handle(handle) {
        // Return empty thermal data for remote GPUs
        (*thermal_settings).count = 0;
        return NVAPI_OK;
    }

    let state = match get_state().lock() {
        Ok(s) => s,
        Err(_) => return NVAPI_ERROR,
    };
    if let Some(ref real) = state.real_nvapi {
        let fn_ptr = real.call_fn(NVAPI_GPU_GET_THERMAL_SETTINGS);
        if !fn_ptr.is_null() {
            type Fn = unsafe extern "C" fn(NvPhysicalGpuHandle, c_uint, *mut NvGpuThermalSettings) -> NvAPI_Status;
            let real_fn: Fn = std::mem::transmute(fn_ptr);
            return real_fn(handle, sensor_index, thermal_settings);
        }
    }
    NVAPI_NO_IMPLEMENTATION
}

// ── NvAPI_GPU_GetMemoryInfo ─────────────────────────────────────────────

unsafe extern "C" fn our_gpu_get_memory_info(
    handle: NvPhysicalGpuHandle,
    mem_info: *mut NvDisplayDriverMemoryInfo,
) -> NvAPI_Status {
    if mem_info.is_null() {
        return NVAPI_INVALID_ARGUMENT;
    }

    if is_remote_handle(handle) {
        let state = match get_state().lock() {
            Ok(s) => s,
            Err(_) => return NVAPI_ERROR,
        };
        let idx = remote_index(handle);
        if let Some(gpu) = state.remote_gpus.get(idx) {
            let vram_kb = (gpu.total_memory / 1024) as u32;
            // Preserve the version field set by the caller
            (*mem_info).dedicated_video_memory = vram_kb;
            (*mem_info).available_dedicated_video_memory = vram_kb;
            (*mem_info).system_video_memory = 0;
            (*mem_info).shared_system_memory = 0;
            (*mem_info).cur_available_dedicated_video_memory = vram_kb;
            (*mem_info).dedicated_video_memory_evictions_size = 0;
            (*mem_info).dedicated_video_memory_eviction_count = 0;
            (*mem_info).dedicated_video_memory_promotions_size = 0;
            (*mem_info).dedicated_video_memory_promotion_count = 0;
            return NVAPI_OK;
        }
        return NVAPI_NVIDIA_DEVICE_NOT_FOUND;
    }

    let state = match get_state().lock() {
        Ok(s) => s,
        Err(_) => return NVAPI_ERROR,
    };
    if let Some(ref real) = state.real_nvapi {
        let fn_ptr = real.call_fn(NVAPI_GPU_GET_MEMORY_INFO);
        if !fn_ptr.is_null() {
            type Fn = unsafe extern "C" fn(NvPhysicalGpuHandle, *mut NvDisplayDriverMemoryInfo) -> NvAPI_Status;
            let real_fn: Fn = std::mem::transmute(fn_ptr);
            return real_fn(handle, mem_info);
        }
    }
    NVAPI_NO_IMPLEMENTATION
}

// ── NvAPI_GPU_GetBusType ────────────────────────────────────────────────

unsafe extern "C" fn our_gpu_get_bus_type(
    handle: NvPhysicalGpuHandle,
    bus_type: *mut u32,
) -> NvAPI_Status {
    if bus_type.is_null() {
        return NVAPI_INVALID_ARGUMENT;
    }

    if is_remote_handle(handle) {
        // 3 = NVAPI_GPU_BUS_TYPE_PCI_EXPRESS (report as PCIe)
        *bus_type = 3;
        return NVAPI_OK;
    }

    let state = match get_state().lock() {
        Ok(s) => s,
        Err(_) => return NVAPI_ERROR,
    };
    if let Some(ref real) = state.real_nvapi {
        let fn_ptr = real.call_fn(NVAPI_GPU_GET_BUS_TYPE);
        if !fn_ptr.is_null() {
            type Fn = unsafe extern "C" fn(NvPhysicalGpuHandle, *mut u32) -> NvAPI_Status;
            let real_fn: Fn = std::mem::transmute(fn_ptr);
            return real_fn(handle, bus_type);
        }
    }
    NVAPI_NO_IMPLEMENTATION
}

// ── NvAPI_GPU_GetBusId ──────────────────────────────────────────────────

unsafe extern "C" fn our_gpu_get_bus_id(
    handle: NvPhysicalGpuHandle,
    bus_id: *mut u32,
) -> NvAPI_Status {
    if bus_id.is_null() {
        return NVAPI_INVALID_ARGUMENT;
    }

    if is_remote_handle(handle) {
        let idx = remote_index(handle);
        // Use a fake bus ID derived from index
        *bus_id = 0xAB00 + idx as u32;
        return NVAPI_OK;
    }

    let state = match get_state().lock() {
        Ok(s) => s,
        Err(_) => return NVAPI_ERROR,
    };
    if let Some(ref real) = state.real_nvapi {
        let fn_ptr = real.call_fn(NVAPI_GPU_GET_BUS_ID);
        if !fn_ptr.is_null() {
            type Fn = unsafe extern "C" fn(NvPhysicalGpuHandle, *mut u32) -> NvAPI_Status;
            let real_fn: Fn = std::mem::transmute(fn_ptr);
            return real_fn(handle, bus_id);
        }
    }
    NVAPI_NO_IMPLEMENTATION
}

// ── NvAPI_GPU_GetPCIIdentifiers ─────────────────────────────────────────

unsafe extern "C" fn our_gpu_get_pci_identifiers(
    handle: NvPhysicalGpuHandle,
    device_id: *mut u32,
    sub_system_id: *mut u32,
    revision_id: *mut u32,
    ext_device_id: *mut u32,
) -> NvAPI_Status {
    if device_id.is_null() || sub_system_id.is_null() || revision_id.is_null() || ext_device_id.is_null() {
        return NVAPI_INVALID_ARGUMENT;
    }

    if is_remote_handle(handle) {
        let state = match get_state().lock() {
            Ok(s) => s,
            Err(_) => return NVAPI_ERROR,
        };
        let idx = remote_index(handle);
        if let Some(gpu) = state.remote_gpus.get(idx) {
            *device_id = (gpu.device_id << 16) | gpu.vendor_id;
            *sub_system_id = 0;
            *revision_id = 0;
            *ext_device_id = gpu.device_id;
            return NVAPI_OK;
        }
        return NVAPI_NVIDIA_DEVICE_NOT_FOUND;
    }

    let state = match get_state().lock() {
        Ok(s) => s,
        Err(_) => return NVAPI_ERROR,
    };
    if let Some(ref real) = state.real_nvapi {
        let fn_ptr = real.call_fn(NVAPI_GPU_GET_PCI_IDENTIFIERS);
        if !fn_ptr.is_null() {
            type Fn = unsafe extern "C" fn(NvPhysicalGpuHandle, *mut u32, *mut u32, *mut u32, *mut u32) -> NvAPI_Status;
            let real_fn: Fn = std::mem::transmute(fn_ptr);
            return real_fn(handle, device_id, sub_system_id, revision_id, ext_device_id);
        }
    }
    NVAPI_NO_IMPLEMENTATION
}

// ── NvAPI_GPU_GetVbiosVersionString ─────────────────────────────────────

unsafe extern "C" fn our_gpu_get_vbios_version_string(
    handle: NvPhysicalGpuHandle,
    bios_revision: *mut c_char,
) -> NvAPI_Status {
    if bios_revision.is_null() {
        return NVAPI_INVALID_ARGUMENT;
    }

    if is_remote_handle(handle) {
        write_c_string(bios_revision, NVAPI_SHORT_STRING_MAX, "RGPU Virtual BIOS");
        return NVAPI_OK;
    }

    let state = match get_state().lock() {
        Ok(s) => s,
        Err(_) => return NVAPI_ERROR,
    };
    if let Some(ref real) = state.real_nvapi {
        let fn_ptr = real.call_fn(NVAPI_GPU_GET_VBIOS_VERSION_STRING);
        if !fn_ptr.is_null() {
            type Fn = unsafe extern "C" fn(NvPhysicalGpuHandle, *mut c_char) -> NvAPI_Status;
            let real_fn: Fn = std::mem::transmute(fn_ptr);
            return real_fn(handle, bios_revision);
        }
    }
    NVAPI_NO_IMPLEMENTATION
}
