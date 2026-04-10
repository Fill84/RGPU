//! NVIDIA Management Library (NVML) interposition library.
//!
//! This cdylib replaces the standard NVML library (nvml.dll on Windows,
//! libnvidia-ml.so.1 on Linux). It intercepts NVML device enumeration calls
//! and adds remote GPUs from the RGPU daemon to the list of visible devices.
//!
//! This makes `nvidia-smi` and `nvidia-container-toolkit` see remote GPUs
//! alongside local GPUs.
//!
//! Architecture:
//!   nvidia-smi / nvidia-container-toolkit
//!          |
//!     nvml.dll (this interpose)
//!       /              \
//!   nvml_real.dll    RGPU daemon (via IPC)
//!   (local GPUs)     (remote GPUs via QueryGpus)

#![allow(non_camel_case_types, non_snake_case)]

mod ipc_client;

use std::ffi::{c_char, c_int, c_uint, c_void};
use std::sync::{Mutex, OnceLock};

use ipc_client::NvmlIpcClient;
use rgpu_protocol::gpu_info::GpuInfo;

// ── NVML Return Types ──────────────────────────────────────────────────

type nvmlReturn_t = c_int;

const NVML_SUCCESS: nvmlReturn_t = 0;
const NVML_ERROR_UNINITIALIZED: nvmlReturn_t = 1;
const NVML_ERROR_INVALID_ARGUMENT: nvmlReturn_t = 2;
const NVML_ERROR_NOT_SUPPORTED: nvmlReturn_t = 3;
const NVML_ERROR_NOT_FOUND: nvmlReturn_t = 6;
const NVML_ERROR_UNKNOWN: nvmlReturn_t = 999;

// NVML device handle — opaque pointer
type nvmlDevice_t = *mut c_void;

// ── NVML Structs ───────────────────────────────────────────────────────

#[repr(C)]
pub struct nvmlMemory_t {
    pub total: u64,
    pub free: u64,
    pub used: u64,
}

#[repr(C)]
pub struct nvmlUtilization_t {
    pub gpu: c_uint,
    pub memory: c_uint,
}

#[repr(C)]
pub struct nvmlPciInfo_t {
    pub busIdLegacy: [c_char; 16],   // NVML_DEVICE_PCI_BUS_ID_BUFFER_V2_SIZE
    pub domain: c_uint,
    pub bus: c_uint,
    pub device: c_uint,
    pub pciDeviceId: c_uint,
    pub pciSubSystemId: c_uint,
    pub busId: [c_char; 32],         // NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE
}

// ── Real NVML Function Table ───────────────────────────────────────────

struct RealNvml {
    _lib: libloading::Library,
    init: unsafe extern "C" fn() -> nvmlReturn_t,
    shutdown: unsafe extern "C" fn() -> nvmlReturn_t,
    device_get_count: unsafe extern "C" fn(*mut c_uint) -> nvmlReturn_t,
    device_get_handle_by_index: unsafe extern "C" fn(c_uint, *mut nvmlDevice_t) -> nvmlReturn_t,
    device_get_name: unsafe extern "C" fn(nvmlDevice_t, *mut c_char, c_uint) -> nvmlReturn_t,
    device_get_uuid: unsafe extern "C" fn(nvmlDevice_t, *mut c_char, c_uint) -> nvmlReturn_t,
    device_get_memory_info: unsafe extern "C" fn(nvmlDevice_t, *mut nvmlMemory_t) -> nvmlReturn_t,
    device_get_temperature:
        unsafe extern "C" fn(nvmlDevice_t, c_int, *mut c_uint) -> nvmlReturn_t,
    device_get_power_usage: unsafe extern "C" fn(nvmlDevice_t, *mut c_uint) -> nvmlReturn_t,
    device_get_utilization_rates:
        unsafe extern "C" fn(nvmlDevice_t, *mut nvmlUtilization_t) -> nvmlReturn_t,
    device_get_pci_info: unsafe extern "C" fn(nvmlDevice_t, *mut nvmlPciInfo_t) -> nvmlReturn_t,
    device_get_cuda_compute_capability:
        unsafe extern "C" fn(nvmlDevice_t, *mut c_int, *mut c_int) -> nvmlReturn_t,
    device_get_minor_number: unsafe extern "C" fn(nvmlDevice_t, *mut c_uint) -> nvmlReturn_t,
    device_get_index: unsafe extern "C" fn(nvmlDevice_t, *mut c_uint) -> nvmlReturn_t,
    system_get_driver_version: unsafe extern "C" fn(*mut c_char, c_uint) -> nvmlReturn_t,
    system_get_nvml_version: unsafe extern "C" fn(*mut c_char, c_uint) -> nvmlReturn_t,
    error_string: unsafe extern "C" fn(nvmlReturn_t) -> *const c_char,
}

// Safety: RealNvml contains function pointers and a Library handle which are
// inherently thread-safe (read-only after initialization). The Library is kept
// alive for the process lifetime.
unsafe impl Send for RealNvml {}

impl RealNvml {
    fn load() -> Result<Self, String> {
        #[cfg(target_os = "windows")]
        let lib_name = "nvml_real.dll";
        #[cfg(target_os = "linux")]
        let lib_name = "libnvidia-ml_real.so.1";
        #[cfg(target_os = "macos")]
        let lib_name = "libnvidia-ml_real.dylib";

        let lib = unsafe {
            libloading::Library::new(lib_name)
                .map_err(|e| format!("failed to load real NVML ({}): {}", lib_name, e))?
        };

        unsafe {
            // Helper: load a function pointer from the library by name.
            // Safety: caller must ensure T matches the actual function signature.
            unsafe fn load_sym<T: Copy>(
                lib: &libloading::Library,
                name: &[u8],
            ) -> Result<T, String> {
                let sym = lib
                    .get::<T>(name)
                    .map_err(|e| {
                        format!(
                            "failed to load {}: {}",
                            String::from_utf8_lossy(name),
                            e
                        )
                    })?;
                Ok(*sym)
            }

            let init = load_sym(&lib, b"nvmlInit_v2")?;
            let shutdown = load_sym(&lib, b"nvmlShutdown")?;
            let device_get_count = load_sym(&lib, b"nvmlDeviceGetCount_v2")?;
            let device_get_handle_by_index = load_sym(&lib, b"nvmlDeviceGetHandleByIndex_v2")?;
            let device_get_name = load_sym(&lib, b"nvmlDeviceGetName")?;
            let device_get_uuid = load_sym(&lib, b"nvmlDeviceGetUUID")?;
            let device_get_memory_info = load_sym(&lib, b"nvmlDeviceGetMemoryInfo")?;
            let device_get_temperature = load_sym(&lib, b"nvmlDeviceGetTemperature")?;
            let device_get_power_usage = load_sym(&lib, b"nvmlDeviceGetPowerUsage")?;
            let device_get_utilization_rates = load_sym(&lib, b"nvmlDeviceGetUtilizationRates")?;
            let device_get_pci_info = load_sym(&lib, b"nvmlDeviceGetPciInfo_v3")?;
            let device_get_cuda_compute_capability =
                load_sym(&lib, b"nvmlDeviceGetCudaComputeCapability")?;
            let device_get_minor_number = load_sym(&lib, b"nvmlDeviceGetMinorNumber")?;
            let device_get_index = load_sym(&lib, b"nvmlDeviceGetIndex")?;
            let system_get_driver_version = load_sym(&lib, b"nvmlSystemGetDriverVersion")?;
            let system_get_nvml_version = load_sym(&lib, b"nvmlSystemGetNVMLVersion")?;
            let error_string = load_sym(&lib, b"nvmlErrorString")?;

            Ok(Self {
                _lib: lib,
                init,
                shutdown,
                device_get_count,
                device_get_handle_by_index,
                device_get_name,
                device_get_uuid,
                device_get_memory_info,
                device_get_temperature,
                device_get_power_usage,
                device_get_utilization_rates,
                device_get_pci_info,
                device_get_cuda_compute_capability,
                device_get_minor_number,
                device_get_index,
                system_get_driver_version,
                system_get_nvml_version,
                error_string,
            })
        }
    }
}

// ── Global State ───────────────────────────────────────────────────────

/// Sentinel value used for remote GPU device handles.
/// Remote GPU handles are: REMOTE_HANDLE_SENTINEL | remote_index
const REMOTE_HANDLE_SENTINEL: u64 = 0xABCD_0000_0000_0000;

struct NvmlState {
    real_nvml: Option<RealNvml>,
    local_gpu_count: u32,
    remote_gpus: Vec<GpuInfo>,
    /// Maps real NVML handles to their index (stored as u64 for Send safety)
    real_handles: Vec<u64>,
    ipc_client: Option<NvmlIpcClient>,
    gpu_map: GpuMap,
}

static STATE: OnceLock<Mutex<NvmlState>> = OnceLock::new();

fn get_state() -> &'static Mutex<NvmlState> {
    STATE.get_or_init(|| {
        Mutex::new(NvmlState {
            real_nvml: None,
            local_gpu_count: 0,
            remote_gpus: Vec::new(),
            real_handles: Vec::new(),
            ipc_client: None,
            gpu_map: GpuMap::default(),
        })
    })
}

/// Check if a device handle refers to a remote GPU.
fn is_remote_handle(device: nvmlDevice_t) -> bool {
    let val = device as u64;
    (val & 0xFFFF_0000_0000_0000) == REMOTE_HANDLE_SENTINEL
}

/// Get the remote GPU index from a remote device handle.
fn remote_index(device: nvmlDevice_t) -> usize {
    (device as u64 & 0x0000_FFFF_FFFF_FFFF) as usize
}

/// Create a remote device handle from an index.
fn make_remote_handle(index: usize) -> nvmlDevice_t {
    (REMOTE_HANDLE_SENTINEL | index as u64) as nvmlDevice_t
}

/// Write a C string into a buffer, null-terminating and truncating if needed.
fn write_c_string(buf: *mut c_char, buf_len: c_uint, s: &str) {
    if buf.is_null() || buf_len == 0 {
        return;
    }
    let max_copy = (buf_len as usize).saturating_sub(1).min(s.len());
    unsafe {
        std::ptr::copy_nonoverlapping(s.as_ptr() as *const c_char, buf, max_copy);
        *buf.add(max_copy) = 0; // null terminator
    }
}

// ── GPU Map (read from device manager) ────────────────────────────────

#[derive(Default)]
struct GpuMap {
    entries: Vec<GpuMapEntry>,
}

struct GpuMapEntry {
    minor_number: u32,
    #[allow(dead_code)]
    server_id: u16,
    #[allow(dead_code)]
    device_index: u32,
    #[allow(dead_code)]
    device_name: String,
    #[allow(dead_code)]
    total_memory: u64,
}

fn read_gpu_map() -> GpuMap {
    #[cfg(not(unix))]
    { return GpuMap::default(); }

    #[cfg(unix)]
    {
        let path = "/run/rgpu/gpu_map.json";
        let json = match std::fs::read_to_string(path) {
            Ok(j) => j,
            Err(_) => return GpuMap::default(),
        };

        // Parse manually — we can't use serde in a cdylib interpose easily
        // Look for "minor_number": N patterns
        let mut entries = Vec::new();

        // Simple JSON parsing for our known format
        for section in json.split('{').skip(2) { // skip outer { and "gpus": [
            let minor = extract_json_u32(section, "minor_number").unwrap_or(0);
            let server_id = extract_json_u32(section, "server_id").unwrap_or(0) as u16;
            let device_index = extract_json_u32(section, "device_index").unwrap_or(0);
            let device_name = extract_json_string(section, "device_name")
                .unwrap_or_default();
            let total_memory = extract_json_u64(section, "total_memory").unwrap_or(0);

            if !device_name.is_empty() {
                entries.push(GpuMapEntry {
                    minor_number: minor,
                    server_id,
                    device_index,
                    device_name,
                    total_memory,
                });
            }
        }

        GpuMap { entries }
    }
}

fn extract_json_u32(s: &str, key: &str) -> Option<u32> {
    let pattern = format!("\"{}\":", key);
    let start = s.find(&pattern)? + pattern.len();
    let rest = s[start..].trim_start();
    let end = rest.find(|c: char| !c.is_ascii_digit())?;
    rest[..end].parse().ok()
}

fn extract_json_u64(s: &str, key: &str) -> Option<u64> {
    let pattern = format!("\"{}\":", key);
    let start = s.find(&pattern)? + pattern.len();
    let rest = s[start..].trim_start();
    let end = rest.find(|c: char| !c.is_ascii_digit())?;
    rest[..end].parse().ok()
}

fn extract_json_string(s: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{}\":", key);
    let start = s.find(&pattern)? + pattern.len();
    let rest = s[start..].trim_start();
    if !rest.starts_with('"') { return None; }
    let rest = &rest[1..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

// ── NVML Exported Functions ────────────────────────────────────────────

/// Marker function for RGPU interpose DLL detection.
#[no_mangle]
pub extern "C" fn rgpu_interpose_marker() -> c_int {
    1
}

#[allow(non_snake_case)]
unsafe fn nvmlInit_v2_impl() -> nvmlReturn_t {
    // Initialize tracing (once)
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .try_init();

    let mut state = match get_state().lock() {
        Ok(s) => s,
        Err(_) => return NVML_ERROR_UNKNOWN,
    };

    // Load and init real NVML
    match RealNvml::load() {
        Ok(real) => {
            let ret = (real.init)();
            if ret != NVML_SUCCESS {
                tracing::warn!("real nvmlInit_v2 returned {}", ret);
                // Continue anyway — remote GPUs might still work
            }

            // Get local GPU count
            let mut count: c_uint = 0;
            let ret = (real.device_get_count)(&mut count);
            if ret == NVML_SUCCESS {
                state.local_gpu_count = count;
                tracing::info!("NVML interpose: {} local GPUs found", count);

                // Cache real handles (as u64 for Send safety)
                state.real_handles.clear();
                for i in 0..count {
                    let mut handle: nvmlDevice_t = std::ptr::null_mut();
                    if (real.device_get_handle_by_index)(i, &mut handle) == NVML_SUCCESS {
                        state.real_handles.push(handle as u64);
                    }
                }
            }

            state.real_nvml = Some(real);
        }
        Err(e) => {
            tracing::warn!("could not load real NVML: {} — only remote GPUs will be visible", e);
            state.local_gpu_count = 0;
        }
    }

    // Connect to RGPU daemon and get remote GPU list
    let ipc_path = resolve_ipc_path();
    let client = NvmlIpcClient::new(&ipc_path);
    match client.query_gpus() {
        Ok(gpus) => {
            // Filter to only remote GPUs (server_id != 0 means remote, or we check
            // if it's not the local server)
            let remote: Vec<GpuInfo> = gpus
                .into_iter()
                .filter(|g| g.server_id != 0 || !has_local_gpu_with_name(&state, &g.device_name))
                .collect();
            tracing::info!("NVML interpose: {} remote GPUs from daemon", remote.len());
            state.remote_gpus = remote;
        }
        Err(e) => {
            tracing::warn!("could not query RGPU daemon: {} — only local GPUs visible", e);
        }
    }
    state.ipc_client = Some(client);

    // Load GPU map written by device manager
    state.gpu_map = read_gpu_map();

    NVML_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn nvmlInit_v2() -> nvmlReturn_t {
    rgpu_common::ffi::catch_panic(NVML_ERROR_UNKNOWN, || nvmlInit_v2_impl())
}

/// Check if a local GPU with the given name exists (to avoid double-counting).
fn has_local_gpu_with_name(state: &NvmlState, name: &str) -> bool {
    if let Some(ref real) = state.real_nvml {
        for i in 0..state.local_gpu_count {
            let mut handle: nvmlDevice_t = std::ptr::null_mut();
            if unsafe { (real.device_get_handle_by_index)(i, &mut handle) } == NVML_SUCCESS {
                let mut buf = [0i8; 256];
                if unsafe { (real.device_get_name)(handle, buf.as_mut_ptr(), 256) } == NVML_SUCCESS
                {
                    let cstr = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) };
                    if let Ok(s) = cstr.to_str() {
                        if s == name {
                            return true;
                        }
                    }
                }
            }
        }
    }
    false
}

/// Resolve IPC path: check RGPU_IPC_ADDRESS env var first, then default.
fn resolve_ipc_path() -> String {
    if let Ok(addr) = std::env::var("RGPU_IPC_ADDRESS") {
        if !addr.is_empty() {
            return addr;
        }
    }
    rgpu_common::platform::default_ipc_path()
}

#[allow(non_snake_case)]
unsafe fn nvmlInit_impl() -> nvmlReturn_t {
    nvmlInit_v2()
}

#[no_mangle]
pub unsafe extern "C" fn nvmlInit() -> nvmlReturn_t {
    rgpu_common::ffi::catch_panic(NVML_ERROR_UNKNOWN, || nvmlInit_impl())
}

/// nvmlInitWithFlags — used by nvidia-smi and newer NVML consumers.
/// The flags parameter is ignored; we always do a full init.
#[allow(non_snake_case)]
unsafe fn nvmlInitWithFlags_impl(_flags: c_uint) -> nvmlReturn_t {
    nvmlInit_v2_impl()
}

#[no_mangle]
pub unsafe extern "C" fn nvmlInitWithFlags(flags: c_uint) -> nvmlReturn_t {
    rgpu_common::ffi::catch_panic(NVML_ERROR_UNKNOWN, || nvmlInitWithFlags_impl(flags))
}

/// Internal NVIDIA export table function used by nvidia-smi to load the API.
/// We return NOT_SUPPORTED since we don't have an internal table — nvidia-smi
/// will fall back to standard NVML function calls.
#[no_mangle]
pub unsafe extern "C" fn nvmlInternalGetExportTable(
    _table: *mut *const std::ffi::c_void,
    _guid: *const std::ffi::c_void,
) -> nvmlReturn_t {
    NVML_ERROR_NOT_SUPPORTED
}

#[allow(non_snake_case)]
unsafe fn nvmlShutdown_impl() -> nvmlReturn_t {
    let mut state = match get_state().lock() {
        Ok(s) => s,
        Err(_) => return NVML_ERROR_UNKNOWN,
    };

    let ret = if let Some(ref real) = state.real_nvml {
        (real.shutdown)()
    } else {
        NVML_SUCCESS
    };

    state.remote_gpus.clear();
    state.real_handles.clear();
    state.ipc_client = None;

    ret
}

#[no_mangle]
pub unsafe extern "C" fn nvmlShutdown() -> nvmlReturn_t {
    rgpu_common::ffi::catch_panic(NVML_ERROR_UNKNOWN, || nvmlShutdown_impl())
}

#[allow(non_snake_case)]
unsafe fn nvmlDeviceGetCount_v2_impl(device_count: *mut c_uint) -> nvmlReturn_t {
    if device_count.is_null() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }

    let state = match get_state().lock() {
        Ok(s) => s,
        Err(_) => return NVML_ERROR_UNKNOWN,
    };

    let total = state.local_gpu_count + state.remote_gpus.len() as u32;
    *device_count = total;
    NVML_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn nvmlDeviceGetCount_v2(device_count: *mut c_uint) -> nvmlReturn_t {
    rgpu_common::ffi::catch_panic(NVML_ERROR_UNKNOWN, || nvmlDeviceGetCount_v2_impl(device_count))
}

#[allow(non_snake_case)]
unsafe fn nvmlDeviceGetCount_impl(device_count: *mut c_uint) -> nvmlReturn_t {
    nvmlDeviceGetCount_v2(device_count)
}

#[no_mangle]
pub unsafe extern "C" fn nvmlDeviceGetCount(device_count: *mut c_uint) -> nvmlReturn_t {
    rgpu_common::ffi::catch_panic(NVML_ERROR_UNKNOWN, || nvmlDeviceGetCount_impl(device_count))
}

#[allow(non_snake_case)]
unsafe fn nvmlDeviceGetHandleByIndex_v2_impl(
    index: c_uint,
    device: *mut nvmlDevice_t,
) -> nvmlReturn_t {
    if device.is_null() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }

    let state = match get_state().lock() {
        Ok(s) => s,
        Err(_) => return NVML_ERROR_UNKNOWN,
    };

    if index < state.local_gpu_count {
        // Local GPU — use real NVML handle
        if let Some(ref real) = state.real_nvml {
            return (real.device_get_handle_by_index)(index, device);
        }
        return NVML_ERROR_NOT_FOUND;
    }

    let remote_idx = (index - state.local_gpu_count) as usize;
    if remote_idx < state.remote_gpus.len() {
        *device = make_remote_handle(remote_idx);
        return NVML_SUCCESS;
    }

    NVML_ERROR_NOT_FOUND
}

#[no_mangle]
pub unsafe extern "C" fn nvmlDeviceGetHandleByIndex_v2(
    index: c_uint,
    device: *mut nvmlDevice_t,
) -> nvmlReturn_t {
    rgpu_common::ffi::catch_panic(NVML_ERROR_UNKNOWN, || nvmlDeviceGetHandleByIndex_v2_impl(index, device))
}

#[allow(non_snake_case)]
unsafe fn nvmlDeviceGetHandleByIndex_impl(
    index: c_uint,
    device: *mut nvmlDevice_t,
) -> nvmlReturn_t {
    nvmlDeviceGetHandleByIndex_v2(index, device)
}

#[no_mangle]
pub unsafe extern "C" fn nvmlDeviceGetHandleByIndex(
    index: c_uint,
    device: *mut nvmlDevice_t,
) -> nvmlReturn_t {
    rgpu_common::ffi::catch_panic(NVML_ERROR_UNKNOWN, || nvmlDeviceGetHandleByIndex_impl(index, device))
}

#[allow(non_snake_case)]
unsafe fn nvmlDeviceGetName_impl(
    device: nvmlDevice_t,
    name: *mut c_char,
    length: c_uint,
) -> nvmlReturn_t {
    if is_remote_handle(device) {
        let state = match get_state().lock() {
            Ok(s) => s,
            Err(_) => return NVML_ERROR_UNKNOWN,
        };
        let idx = remote_index(device);
        if let Some(gpu) = state.remote_gpus.get(idx) {
            let display_name = &gpu.device_name;
            write_c_string(name, length, display_name);
            return NVML_SUCCESS;
        }
        return NVML_ERROR_NOT_FOUND;
    }

    let state = match get_state().lock() {
        Ok(s) => s,
        Err(_) => return NVML_ERROR_UNKNOWN,
    };
    if let Some(ref real) = state.real_nvml {
        return (real.device_get_name)(device, name, length);
    }
    NVML_ERROR_UNINITIALIZED
}

#[no_mangle]
pub unsafe extern "C" fn nvmlDeviceGetName(
    device: nvmlDevice_t,
    name: *mut c_char,
    length: c_uint,
) -> nvmlReturn_t {
    rgpu_common::ffi::catch_panic(NVML_ERROR_UNKNOWN, || nvmlDeviceGetName_impl(device, name, length))
}

#[allow(non_snake_case)]
unsafe fn nvmlDeviceGetUUID_impl(
    device: nvmlDevice_t,
    uuid: *mut c_char,
    length: c_uint,
) -> nvmlReturn_t {
    if is_remote_handle(device) {
        let state = match get_state().lock() {
            Ok(s) => s,
            Err(_) => return NVML_ERROR_UNKNOWN,
        };
        let idx = remote_index(device);
        if let Some(gpu) = state.remote_gpus.get(idx) {
            // Generate a deterministic UUID using FNV-1a hash (NVIDIA standard format)
            let hash_input = format!("RGPU-{}-{}", gpu.server_id, gpu.server_device_index);
            let mut hash: u64 = 0xcbf29ce484222325u64; // FNV-1a offset basis
            for byte in hash_input.as_bytes() {
                hash ^= *byte as u64;
                hash = hash.wrapping_mul(0x100000001b3u64); // FNV-1a prime
            }
            let hash2 = hash.wrapping_mul(0x517cc1b727220a95u64);
            let uuid_str = format!(
                "GPU-{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
                (hash >> 32) as u32,
                (hash >> 16) as u16,
                hash as u16,
                (hash2 >> 48) as u16,
                hash2 & 0xFFFFFFFFFFFFu64,
            );
            write_c_string(uuid, length, &uuid_str);
            return NVML_SUCCESS;
        }
        return NVML_ERROR_NOT_FOUND;
    }

    let state = match get_state().lock() {
        Ok(s) => s,
        Err(_) => return NVML_ERROR_UNKNOWN,
    };
    if let Some(ref real) = state.real_nvml {
        return (real.device_get_uuid)(device, uuid, length);
    }
    NVML_ERROR_UNINITIALIZED
}

#[no_mangle]
pub unsafe extern "C" fn nvmlDeviceGetUUID(
    device: nvmlDevice_t,
    uuid: *mut c_char,
    length: c_uint,
) -> nvmlReturn_t {
    rgpu_common::ffi::catch_panic(NVML_ERROR_UNKNOWN, || nvmlDeviceGetUUID_impl(device, uuid, length))
}

#[allow(non_snake_case)]
unsafe fn nvmlDeviceGetMemoryInfo_impl(
    device: nvmlDevice_t,
    memory: *mut nvmlMemory_t,
) -> nvmlReturn_t {
    if memory.is_null() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }

    if is_remote_handle(device) {
        let state = match get_state().lock() {
            Ok(s) => s,
            Err(_) => return NVML_ERROR_UNKNOWN,
        };
        let idx = remote_index(device);
        if let Some(gpu) = state.remote_gpus.get(idx) {
            (*memory).total = gpu.total_memory;
            (*memory).free = gpu.total_memory; // assume full availability for remote
            (*memory).used = 0;
            return NVML_SUCCESS;
        }
        return NVML_ERROR_NOT_FOUND;
    }

    let state = match get_state().lock() {
        Ok(s) => s,
        Err(_) => return NVML_ERROR_UNKNOWN,
    };
    if let Some(ref real) = state.real_nvml {
        return (real.device_get_memory_info)(device, memory);
    }
    NVML_ERROR_UNINITIALIZED
}

#[no_mangle]
pub unsafe extern "C" fn nvmlDeviceGetMemoryInfo(
    device: nvmlDevice_t,
    memory: *mut nvmlMemory_t,
) -> nvmlReturn_t {
    rgpu_common::ffi::catch_panic(NVML_ERROR_UNKNOWN, || nvmlDeviceGetMemoryInfo_impl(device, memory))
}

#[allow(non_snake_case)]
unsafe fn nvmlDeviceGetTemperature_impl(
    device: nvmlDevice_t,
    sensor_type: c_int,
    temp: *mut c_uint,
) -> nvmlReturn_t {
    if temp.is_null() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }

    if is_remote_handle(device) {
        *temp = 0; // Temperature not available for remote GPUs
        return NVML_SUCCESS;
    }

    let state = match get_state().lock() {
        Ok(s) => s,
        Err(_) => return NVML_ERROR_UNKNOWN,
    };
    if let Some(ref real) = state.real_nvml {
        return (real.device_get_temperature)(device, sensor_type, temp);
    }
    NVML_ERROR_UNINITIALIZED
}

#[no_mangle]
pub unsafe extern "C" fn nvmlDeviceGetTemperature(
    device: nvmlDevice_t,
    sensor_type: c_int,
    temp: *mut c_uint,
) -> nvmlReturn_t {
    rgpu_common::ffi::catch_panic(NVML_ERROR_UNKNOWN, || nvmlDeviceGetTemperature_impl(device, sensor_type, temp))
}

#[allow(non_snake_case)]
unsafe fn nvmlDeviceGetPowerUsage_impl(
    device: nvmlDevice_t,
    power: *mut c_uint,
) -> nvmlReturn_t {
    if power.is_null() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }

    if is_remote_handle(device) {
        *power = 0;
        return NVML_SUCCESS;
    }

    let state = match get_state().lock() {
        Ok(s) => s,
        Err(_) => return NVML_ERROR_UNKNOWN,
    };
    if let Some(ref real) = state.real_nvml {
        return (real.device_get_power_usage)(device, power);
    }
    NVML_ERROR_UNINITIALIZED
}

#[no_mangle]
pub unsafe extern "C" fn nvmlDeviceGetPowerUsage(
    device: nvmlDevice_t,
    power: *mut c_uint,
) -> nvmlReturn_t {
    rgpu_common::ffi::catch_panic(NVML_ERROR_UNKNOWN, || nvmlDeviceGetPowerUsage_impl(device, power))
}

#[allow(non_snake_case)]
unsafe fn nvmlDeviceGetUtilizationRates_impl(
    device: nvmlDevice_t,
    utilization: *mut nvmlUtilization_t,
) -> nvmlReturn_t {
    if utilization.is_null() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }

    if is_remote_handle(device) {
        (*utilization).gpu = 0;
        (*utilization).memory = 0;
        return NVML_SUCCESS;
    }

    let state = match get_state().lock() {
        Ok(s) => s,
        Err(_) => return NVML_ERROR_UNKNOWN,
    };
    if let Some(ref real) = state.real_nvml {
        return (real.device_get_utilization_rates)(device, utilization);
    }
    NVML_ERROR_UNINITIALIZED
}

#[no_mangle]
pub unsafe extern "C" fn nvmlDeviceGetUtilizationRates(
    device: nvmlDevice_t,
    utilization: *mut nvmlUtilization_t,
) -> nvmlReturn_t {
    rgpu_common::ffi::catch_panic(NVML_ERROR_UNKNOWN, || nvmlDeviceGetUtilizationRates_impl(device, utilization))
}

#[allow(non_snake_case)]
unsafe fn nvmlDeviceGetPciInfo_v3_impl(
    device: nvmlDevice_t,
    pci: *mut nvmlPciInfo_t,
) -> nvmlReturn_t {
    if pci.is_null() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }

    if is_remote_handle(device) {
        let state = match get_state().lock() {
            Ok(s) => s,
            Err(_) => return NVML_ERROR_UNKNOWN,
        };
        let idx = remote_index(device);
        if let Some(gpu) = state.remote_gpus.get(idx) {
            // Generate realistic PCI info for remote GPU
            std::ptr::write_bytes(pci, 0, 1);
            let bus = 0x80u32 + gpu.server_id as u32;
            let dev_slot = gpu.server_device_index;
            (*pci).domain = 0;
            (*pci).bus = bus;
            (*pci).device = dev_slot;
            (*pci).pciDeviceId = gpu.device_id;
            (*pci).pciSubSystemId = gpu.vendor_id;

            let bus_id_str = format!("00000000:{:02X}:{:02X}.0", bus, dev_slot);
            write_c_string((*pci).busId.as_mut_ptr(), 32, &bus_id_str);
            write_c_string((*pci).busIdLegacy.as_mut_ptr(), 16, &bus_id_str);

            return NVML_SUCCESS;
        }
        return NVML_ERROR_NOT_FOUND;
    }

    let state = match get_state().lock() {
        Ok(s) => s,
        Err(_) => return NVML_ERROR_UNKNOWN,
    };
    if let Some(ref real) = state.real_nvml {
        return (real.device_get_pci_info)(device, pci);
    }
    NVML_ERROR_UNINITIALIZED
}

#[no_mangle]
pub unsafe extern "C" fn nvmlDeviceGetPciInfo_v3(
    device: nvmlDevice_t,
    pci: *mut nvmlPciInfo_t,
) -> nvmlReturn_t {
    rgpu_common::ffi::catch_panic(NVML_ERROR_UNKNOWN, || nvmlDeviceGetPciInfo_v3_impl(device, pci))
}

#[allow(non_snake_case)]
unsafe fn nvmlDeviceGetPciInfo_impl(
    device: nvmlDevice_t,
    pci: *mut nvmlPciInfo_t,
) -> nvmlReturn_t {
    nvmlDeviceGetPciInfo_v3(device, pci)
}

#[no_mangle]
pub unsafe extern "C" fn nvmlDeviceGetPciInfo(
    device: nvmlDevice_t,
    pci: *mut nvmlPciInfo_t,
) -> nvmlReturn_t {
    rgpu_common::ffi::catch_panic(NVML_ERROR_UNKNOWN, || nvmlDeviceGetPciInfo_impl(device, pci))
}

#[allow(non_snake_case)]
unsafe fn nvmlDeviceGetCudaComputeCapability_impl(
    device: nvmlDevice_t,
    major: *mut c_int,
    minor: *mut c_int,
) -> nvmlReturn_t {
    if major.is_null() || minor.is_null() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }

    if is_remote_handle(device) {
        let state = match get_state().lock() {
            Ok(s) => s,
            Err(_) => return NVML_ERROR_UNKNOWN,
        };
        let idx = remote_index(device);
        if let Some(gpu) = state.remote_gpus.get(idx) {
            if let Some((maj, min)) = gpu.cuda_compute_capability {
                *major = maj;
                *minor = min;
                return NVML_SUCCESS;
            }
            return NVML_ERROR_NOT_SUPPORTED;
        }
        return NVML_ERROR_NOT_FOUND;
    }

    let state = match get_state().lock() {
        Ok(s) => s,
        Err(_) => return NVML_ERROR_UNKNOWN,
    };
    if let Some(ref real) = state.real_nvml {
        return (real.device_get_cuda_compute_capability)(device, major, minor);
    }
    NVML_ERROR_UNINITIALIZED
}

#[no_mangle]
pub unsafe extern "C" fn nvmlDeviceGetCudaComputeCapability(
    device: nvmlDevice_t,
    major: *mut c_int,
    minor: *mut c_int,
) -> nvmlReturn_t {
    rgpu_common::ffi::catch_panic(NVML_ERROR_UNKNOWN, || nvmlDeviceGetCudaComputeCapability_impl(device, major, minor))
}

#[allow(non_snake_case)]
unsafe fn nvmlDeviceGetMinorNumber_impl(
    device: nvmlDevice_t,
    minor_number: *mut c_uint,
) -> nvmlReturn_t {
    if minor_number.is_null() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }

    if is_remote_handle(device) {
        let state = match get_state().lock() {
            Ok(s) => s,
            Err(_) => return NVML_ERROR_UNKNOWN,
        };
        let idx = remote_index(device);
        // Look up from GPU map first
        if idx < state.gpu_map.entries.len() {
            *minor_number = state.gpu_map.entries[idx].minor_number;
            return NVML_SUCCESS;
        }
        // Fallback: local_count + idx
        *minor_number = state.local_gpu_count + idx as u32;
        return NVML_SUCCESS;
    }

    let state = match get_state().lock() {
        Ok(s) => s,
        Err(_) => return NVML_ERROR_UNKNOWN,
    };
    if let Some(ref real) = state.real_nvml {
        return (real.device_get_minor_number)(device, minor_number);
    }
    NVML_ERROR_UNINITIALIZED
}

#[no_mangle]
pub unsafe extern "C" fn nvmlDeviceGetMinorNumber(
    device: nvmlDevice_t,
    minor_number: *mut c_uint,
) -> nvmlReturn_t {
    rgpu_common::ffi::catch_panic(NVML_ERROR_UNKNOWN, || nvmlDeviceGetMinorNumber_impl(device, minor_number))
}

#[allow(non_snake_case)]
unsafe fn nvmlDeviceGetIndex_impl(
    device: nvmlDevice_t,
    index: *mut c_uint,
) -> nvmlReturn_t {
    if index.is_null() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }

    if is_remote_handle(device) {
        let state = match get_state().lock() {
            Ok(s) => s,
            Err(_) => return NVML_ERROR_UNKNOWN,
        };
        let idx = remote_index(device);
        *index = state.local_gpu_count + idx as u32;
        return NVML_SUCCESS;
    }

    let state = match get_state().lock() {
        Ok(s) => s,
        Err(_) => return NVML_ERROR_UNKNOWN,
    };
    if let Some(ref real) = state.real_nvml {
        return (real.device_get_index)(device, index);
    }
    NVML_ERROR_UNINITIALIZED
}

#[no_mangle]
pub unsafe extern "C" fn nvmlDeviceGetIndex(
    device: nvmlDevice_t,
    index: *mut c_uint,
) -> nvmlReturn_t {
    rgpu_common::ffi::catch_panic(NVML_ERROR_UNKNOWN, || nvmlDeviceGetIndex_impl(device, index))
}

#[allow(non_snake_case)]
unsafe fn nvmlDeviceGetHandleByUUID_impl(
    uuid: *const c_char,
    device: *mut nvmlDevice_t,
) -> nvmlReturn_t {
    if uuid.is_null() || device.is_null() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }

    let uuid_str = match std::ffi::CStr::from_ptr(uuid).to_str() {
        Ok(s) => s,
        Err(_) => return NVML_ERROR_INVALID_ARGUMENT,
    };

    // Check if it's one of our remote GPU UUIDs by regenerating and comparing
    if uuid_str.starts_with("GPU-") {
        let state = match get_state().lock() {
            Ok(s) => s,
            Err(_) => return NVML_ERROR_UNKNOWN,
        };
        for (i, gpu) in state.remote_gpus.iter().enumerate() {
            let hash_input = format!("RGPU-{}-{}", gpu.server_id, gpu.server_device_index);
            let mut hash: u64 = 0xcbf29ce484222325u64;
            for byte in hash_input.as_bytes() {
                hash ^= *byte as u64;
                hash = hash.wrapping_mul(0x100000001b3u64);
            }
            let hash2 = hash.wrapping_mul(0x517cc1b727220a95u64);
            let candidate = format!(
                "GPU-{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
                (hash >> 32) as u32,
                (hash >> 16) as u16,
                hash as u16,
                (hash2 >> 48) as u16,
                hash2 & 0xFFFFFFFFFFFFu64,
            );
            if uuid_str == candidate {
                *device = make_remote_handle(i);
                return NVML_SUCCESS;
            }
        }
        // Fall through to check real NVML
    }

    // Try real NVML
    let state = match get_state().lock() {
        Ok(s) => s,
        Err(_) => return NVML_ERROR_UNKNOWN,
    };
    // Real NVML doesn't have a direct "by UUID" in older versions, but newer ones do
    // For now, search by iterating
    if let Some(ref real) = state.real_nvml {
        for i in 0..state.local_gpu_count {
            let mut handle: nvmlDevice_t = std::ptr::null_mut();
            if (real.device_get_handle_by_index)(i, &mut handle) == NVML_SUCCESS {
                let mut buf = [0i8; 96];
                if (real.device_get_uuid)(handle, buf.as_mut_ptr(), 96) == NVML_SUCCESS {
                    let cstr = std::ffi::CStr::from_ptr(buf.as_ptr());
                    if let Ok(s) = cstr.to_str() {
                        if s == uuid_str {
                            *device = handle;
                            return NVML_SUCCESS;
                        }
                    }
                }
            }
        }
    }
    NVML_ERROR_NOT_FOUND
}

#[no_mangle]
pub unsafe extern "C" fn nvmlDeviceGetHandleByUUID(
    uuid: *const c_char,
    device: *mut nvmlDevice_t,
) -> nvmlReturn_t {
    rgpu_common::ffi::catch_panic(NVML_ERROR_UNKNOWN, || nvmlDeviceGetHandleByUUID_impl(uuid, device))
}

#[allow(non_snake_case)]
unsafe fn nvmlDeviceGetHandleByPciBusId_v2_impl(
    pci_bus_id: *const c_char,
    device: *mut nvmlDevice_t,
) -> nvmlReturn_t {
    if pci_bus_id.is_null() || device.is_null() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }

    let bus_id_str = match std::ffi::CStr::from_ptr(pci_bus_id).to_str() {
        Ok(s) => s,
        Err(_) => return NVML_ERROR_INVALID_ARGUMENT,
    };

    // Check if it matches one of our remote GPU PCI bus IDs (00000000:8X:XX.0)
    // Remote buses start at 0x80, so they begin with "00000000:8" or higher
    {
        let state = match get_state().lock() {
            Ok(s) => s,
            Err(_) => return NVML_ERROR_UNKNOWN,
        };
        for (i, gpu) in state.remote_gpus.iter().enumerate() {
            let bus = 0x80u32 + gpu.server_id as u32;
            let candidate_bus_id = format!(
                "00000000:{:02X}:{:02X}.0",
                bus, gpu.server_device_index
            );
            if bus_id_str == candidate_bus_id {
                *device = make_remote_handle(i);
                return NVML_SUCCESS;
            }
        }
        // No remote match — fall through to real NVML
    }

    // Try real NVML — search by iterating
    let state = match get_state().lock() {
        Ok(s) => s,
        Err(_) => return NVML_ERROR_UNKNOWN,
    };
    if let Some(ref real) = state.real_nvml {
        for i in 0..state.local_gpu_count {
            let mut handle: nvmlDevice_t = std::ptr::null_mut();
            if (real.device_get_handle_by_index)(i, &mut handle) == NVML_SUCCESS {
                let mut pci_info: nvmlPciInfo_t = std::mem::zeroed();
                if (real.device_get_pci_info)(handle, &mut pci_info) == NVML_SUCCESS {
                    let cstr = std::ffi::CStr::from_ptr(pci_info.busId.as_ptr());
                    if let Ok(s) = cstr.to_str() {
                        if s == bus_id_str {
                            *device = handle;
                            return NVML_SUCCESS;
                        }
                    }
                }
            }
        }
    }
    NVML_ERROR_NOT_FOUND
}

#[no_mangle]
pub unsafe extern "C" fn nvmlDeviceGetHandleByPciBusId_v2(
    pci_bus_id: *const c_char,
    device: *mut nvmlDevice_t,
) -> nvmlReturn_t {
    rgpu_common::ffi::catch_panic(NVML_ERROR_UNKNOWN, || nvmlDeviceGetHandleByPciBusId_v2_impl(pci_bus_id, device))
}

#[allow(non_snake_case)]
unsafe fn nvmlDeviceGetHandleByPciBusId_impl(
    pci_bus_id: *const c_char,
    device: *mut nvmlDevice_t,
) -> nvmlReturn_t {
    nvmlDeviceGetHandleByPciBusId_v2(pci_bus_id, device)
}

#[no_mangle]
pub unsafe extern "C" fn nvmlDeviceGetHandleByPciBusId(
    pci_bus_id: *const c_char,
    device: *mut nvmlDevice_t,
) -> nvmlReturn_t {
    rgpu_common::ffi::catch_panic(NVML_ERROR_UNKNOWN, || nvmlDeviceGetHandleByPciBusId_impl(pci_bus_id, device))
}

#[allow(non_snake_case)]
unsafe fn nvmlSystemGetDriverVersion_impl(
    version: *mut c_char,
    length: c_uint,
) -> nvmlReturn_t {
    let state = match get_state().lock() {
        Ok(s) => s,
        Err(_) => return NVML_ERROR_UNKNOWN,
    };
    if let Some(ref real) = state.real_nvml {
        return (real.system_get_driver_version)(version, length);
    }
    // No real NVML — return our own version string
    write_c_string(version, length, "RGPU Virtual Driver");
    NVML_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn nvmlSystemGetDriverVersion(
    version: *mut c_char,
    length: c_uint,
) -> nvmlReturn_t {
    rgpu_common::ffi::catch_panic(NVML_ERROR_UNKNOWN, || nvmlSystemGetDriverVersion_impl(version, length))
}

#[allow(non_snake_case)]
unsafe fn nvmlSystemGetNVMLVersion_impl(
    version: *mut c_char,
    length: c_uint,
) -> nvmlReturn_t {
    let state = match get_state().lock() {
        Ok(s) => s,
        Err(_) => return NVML_ERROR_UNKNOWN,
    };
    if let Some(ref real) = state.real_nvml {
        return (real.system_get_nvml_version)(version, length);
    }
    write_c_string(version, length, "12.0");
    NVML_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn nvmlSystemGetNVMLVersion(
    version: *mut c_char,
    length: c_uint,
) -> nvmlReturn_t {
    rgpu_common::ffi::catch_panic(NVML_ERROR_UNKNOWN, || nvmlSystemGetNVMLVersion_impl(version, length))
}

#[allow(non_snake_case)]
unsafe fn nvmlErrorString_impl(result: nvmlReturn_t) -> *const c_char {
    let state = match get_state().lock() {
        Ok(s) => s,
        Err(_) => return b"Unknown Error\0".as_ptr() as *const c_char,
    };
    if let Some(ref real) = state.real_nvml {
        return (real.error_string)(result);
    }

    match result {
        0 => b"Success\0".as_ptr() as *const c_char,
        1 => b"Uninitialized\0".as_ptr() as *const c_char,
        2 => b"Invalid Argument\0".as_ptr() as *const c_char,
        3 => b"Not Supported\0".as_ptr() as *const c_char,
        6 => b"Not Found\0".as_ptr() as *const c_char,
        _ => b"Unknown Error\0".as_ptr() as *const c_char,
    }
}

#[no_mangle]
pub unsafe extern "C" fn nvmlErrorString(result: nvmlReturn_t) -> *const c_char {
    rgpu_common::ffi::catch_panic(b"Unknown Error\0".as_ptr() as *const c_char, || nvmlErrorString_impl(result))
}

// ── Process queries (stubs for remote GPUs) ────────────────────────────

#[repr(C)]
pub struct nvmlProcessInfo_t {
    pub pid: c_uint,
    pub usedGpuMemory: u64,
    pub gpuInstanceId: c_uint,
    pub computeInstanceId: c_uint,
}

#[allow(non_snake_case)]
unsafe fn nvmlDeviceGetComputeRunningProcesses_v3_impl(
    device: nvmlDevice_t,
    info_count: *mut c_uint,
    _infos: *mut nvmlProcessInfo_t,
) -> nvmlReturn_t {
    if info_count.is_null() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }

    if is_remote_handle(device) {
        *info_count = 0;
        return NVML_SUCCESS;
    }

    // For local GPUs, we'd need the real NVML function
    // For now, return empty
    *info_count = 0;
    NVML_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn nvmlDeviceGetComputeRunningProcesses_v3(
    device: nvmlDevice_t,
    info_count: *mut c_uint,
    _infos: *mut nvmlProcessInfo_t,
) -> nvmlReturn_t {
    rgpu_common::ffi::catch_panic(NVML_ERROR_UNKNOWN, || nvmlDeviceGetComputeRunningProcesses_v3_impl(device, info_count, _infos))
}

#[allow(non_snake_case)]
unsafe fn nvmlDeviceGetGraphicsRunningProcesses_v3_impl(
    device: nvmlDevice_t,
    info_count: *mut c_uint,
    _infos: *mut nvmlProcessInfo_t,
) -> nvmlReturn_t {
    if info_count.is_null() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }

    if is_remote_handle(device) {
        *info_count = 0;
        return NVML_SUCCESS;
    }

    *info_count = 0;
    NVML_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn nvmlDeviceGetGraphicsRunningProcesses_v3(
    device: nvmlDevice_t,
    info_count: *mut c_uint,
    _infos: *mut nvmlProcessInfo_t,
) -> nvmlReturn_t {
    rgpu_common::ffi::catch_panic(NVML_ERROR_UNKNOWN, || nvmlDeviceGetGraphicsRunningProcesses_v3_impl(device, info_count, _infos))
}

// ── Additional NVML functions commonly queried ─────────────────────────

#[allow(non_snake_case)]
unsafe fn nvmlDeviceGetClockInfo_impl(
    device: nvmlDevice_t,
    _clock_type: c_int,
    clock: *mut c_uint,
) -> nvmlReturn_t {
    if clock.is_null() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    if is_remote_handle(device) {
        *clock = 0;
        return NVML_SUCCESS;
    }
    NVML_ERROR_NOT_SUPPORTED
}

#[no_mangle]
pub unsafe extern "C" fn nvmlDeviceGetClockInfo(
    device: nvmlDevice_t,
    _clock_type: c_int,
    clock: *mut c_uint,
) -> nvmlReturn_t {
    rgpu_common::ffi::catch_panic(NVML_ERROR_UNKNOWN, || nvmlDeviceGetClockInfo_impl(device, _clock_type, clock))
}

#[allow(non_snake_case)]
unsafe fn nvmlDeviceGetMaxClockInfo_impl(
    device: nvmlDevice_t,
    _clock_type: c_int,
    clock: *mut c_uint,
) -> nvmlReturn_t {
    if clock.is_null() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    if is_remote_handle(device) {
        *clock = 0;
        return NVML_SUCCESS;
    }
    NVML_ERROR_NOT_SUPPORTED
}

#[no_mangle]
pub unsafe extern "C" fn nvmlDeviceGetMaxClockInfo(
    device: nvmlDevice_t,
    _clock_type: c_int,
    clock: *mut c_uint,
) -> nvmlReturn_t {
    rgpu_common::ffi::catch_panic(NVML_ERROR_UNKNOWN, || nvmlDeviceGetMaxClockInfo_impl(device, _clock_type, clock))
}

#[allow(non_snake_case)]
unsafe fn nvmlDeviceGetFanSpeed_impl(
    device: nvmlDevice_t,
    speed: *mut c_uint,
) -> nvmlReturn_t {
    if speed.is_null() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    if is_remote_handle(device) {
        *speed = 0;
        return NVML_SUCCESS;
    }
    NVML_ERROR_NOT_SUPPORTED
}

#[no_mangle]
pub unsafe extern "C" fn nvmlDeviceGetFanSpeed(
    device: nvmlDevice_t,
    speed: *mut c_uint,
) -> nvmlReturn_t {
    rgpu_common::ffi::catch_panic(NVML_ERROR_UNKNOWN, || nvmlDeviceGetFanSpeed_impl(device, speed))
}

#[allow(non_snake_case)]
unsafe fn nvmlDeviceGetDisplayActive_impl(
    device: nvmlDevice_t,
    is_active: *mut c_uint,
) -> nvmlReturn_t {
    if is_active.is_null() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    if is_remote_handle(device) {
        *is_active = 0; // no display on remote
        return NVML_SUCCESS;
    }
    NVML_ERROR_NOT_SUPPORTED
}

#[no_mangle]
pub unsafe extern "C" fn nvmlDeviceGetDisplayActive(
    device: nvmlDevice_t,
    is_active: *mut c_uint,
) -> nvmlReturn_t {
    rgpu_common::ffi::catch_panic(NVML_ERROR_UNKNOWN, || nvmlDeviceGetDisplayActive_impl(device, is_active))
}

#[allow(non_snake_case)]
unsafe fn nvmlDeviceGetDisplayMode_impl(
    device: nvmlDevice_t,
    display: *mut c_uint,
) -> nvmlReturn_t {
    if display.is_null() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    if is_remote_handle(device) {
        *display = 0;
        return NVML_SUCCESS;
    }
    NVML_ERROR_NOT_SUPPORTED
}

#[no_mangle]
pub unsafe extern "C" fn nvmlDeviceGetDisplayMode(
    device: nvmlDevice_t,
    display: *mut c_uint,
) -> nvmlReturn_t {
    rgpu_common::ffi::catch_panic(NVML_ERROR_UNKNOWN, || nvmlDeviceGetDisplayMode_impl(device, display))
}

#[allow(non_snake_case)]
unsafe fn nvmlDeviceGetPersistenceMode_impl(
    device: nvmlDevice_t,
    mode: *mut c_uint,
) -> nvmlReturn_t {
    if mode.is_null() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    if is_remote_handle(device) {
        *mode = 0;
        return NVML_SUCCESS;
    }
    NVML_ERROR_NOT_SUPPORTED
}

#[no_mangle]
pub unsafe extern "C" fn nvmlDeviceGetPersistenceMode(
    device: nvmlDevice_t,
    mode: *mut c_uint,
) -> nvmlReturn_t {
    rgpu_common::ffi::catch_panic(NVML_ERROR_UNKNOWN, || nvmlDeviceGetPersistenceMode_impl(device, mode))
}

#[allow(non_snake_case)]
unsafe fn nvmlDeviceGetPerformanceState_impl(
    device: nvmlDevice_t,
    p_state: *mut c_int,
) -> nvmlReturn_t {
    if p_state.is_null() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    if is_remote_handle(device) {
        *p_state = 0; // P0 (max performance)
        return NVML_SUCCESS;
    }
    NVML_ERROR_NOT_SUPPORTED
}

#[no_mangle]
pub unsafe extern "C" fn nvmlDeviceGetPerformanceState(
    device: nvmlDevice_t,
    p_state: *mut c_int,
) -> nvmlReturn_t {
    rgpu_common::ffi::catch_panic(NVML_ERROR_UNKNOWN, || nvmlDeviceGetPerformanceState_impl(device, p_state))
}

#[allow(non_snake_case)]
unsafe fn nvmlDeviceGetEncoderUtilization_impl(
    device: nvmlDevice_t,
    utilization: *mut c_uint,
    sampling_period: *mut c_uint,
) -> nvmlReturn_t {
    if is_remote_handle(device) {
        if !utilization.is_null() {
            *utilization = 0;
        }
        if !sampling_period.is_null() {
            *sampling_period = 0;
        }
        return NVML_SUCCESS;
    }
    NVML_ERROR_NOT_SUPPORTED
}

#[no_mangle]
pub unsafe extern "C" fn nvmlDeviceGetEncoderUtilization(
    device: nvmlDevice_t,
    utilization: *mut c_uint,
    sampling_period: *mut c_uint,
) -> nvmlReturn_t {
    rgpu_common::ffi::catch_panic(NVML_ERROR_UNKNOWN, || nvmlDeviceGetEncoderUtilization_impl(device, utilization, sampling_period))
}

#[allow(non_snake_case)]
unsafe fn nvmlDeviceGetDecoderUtilization_impl(
    device: nvmlDevice_t,
    utilization: *mut c_uint,
    sampling_period: *mut c_uint,
) -> nvmlReturn_t {
    if is_remote_handle(device) {
        if !utilization.is_null() {
            *utilization = 0;
        }
        if !sampling_period.is_null() {
            *sampling_period = 0;
        }
        return NVML_SUCCESS;
    }
    NVML_ERROR_NOT_SUPPORTED
}

#[no_mangle]
pub unsafe extern "C" fn nvmlDeviceGetDecoderUtilization(
    device: nvmlDevice_t,
    utilization: *mut c_uint,
    sampling_period: *mut c_uint,
) -> nvmlReturn_t {
    rgpu_common::ffi::catch_panic(NVML_ERROR_UNKNOWN, || nvmlDeviceGetDecoderUtilization_impl(device, utilization, sampling_period))
}
