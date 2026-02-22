//! CUDA Driver API interception library.
//!
//! This cdylib replaces the standard CUDA driver library (libcuda.so / nvcuda.dll).
//! It intercepts CUDA driver API calls and forwards them to the RGPU client daemon
//! via IPC, which in turn sends them to a remote RGPU server.
//!
//! Usage:
//! - Linux: LD_PRELOAD=librgpu_cuda_interpose.so <application>
//! - Windows: Place as nvcuda.dll in the application's directory

mod ipc_client;
pub mod handle_store;
pub mod error;
pub mod proc_address;
pub mod stubs;

use std::ffi::{c_char, c_int, c_uint, c_void};
use std::sync::OnceLock;

use tracing::{debug, error, info};

use rgpu_protocol::cuda_commands::{CudaCommand, CudaResponse, KernelParam};
use rgpu_protocol::handle::{NetworkHandle, ResourceType};

use ipc_client::IpcClient;

// CUDA types
type CUresult = c_int;
type CUdevice = c_int;
type CUcontext = *mut c_void;
type CUmodule = *mut c_void;
type CUfunction = *mut c_void;
type CUdeviceptr = u64;
type CUstream = *mut c_void;
type CUevent = *mut c_void;
type CUlinkState = *mut c_void;
type CUmemoryPool = *mut c_void;

const CUDA_SUCCESS: CUresult = 0;
const CUDA_ERROR_INVALID_VALUE: CUresult = 1;
const _CUDA_ERROR_NOT_INITIALIZED: CUresult = 3;
const CUDA_ERROR_NOT_READY: CUresult = 600;
const _CUDA_ERROR_NOT_SUPPORTED: CUresult = 801;
const CUDA_ERROR_UNKNOWN: CUresult = 999;

static IPC_CLIENT: OnceLock<IpcClient> = OnceLock::new();

fn get_client() -> &'static IpcClient {
    IPC_CLIENT.get_or_init(|| {
        let path = rgpu_common::platform::default_ipc_path();
        IpcClient::new(&path)
    })
}

fn send_cuda_command(cmd: CudaCommand) -> CudaResponse {
    let client = get_client();
    match client.send_command(cmd) {
        Ok(resp) => resp,
        Err(e) => {
            error!("IPC error: {}", e);
            CudaResponse::Error {
                code: CUDA_ERROR_UNKNOWN,
                message: e.to_string(),
            }
        }
    }
}

/// Create a null/default NetworkHandle for stream references (NULL stream = default).
fn null_stream_handle() -> NetworkHandle {
    NetworkHandle {
        server_id: 0,
        session_id: 0,
        resource_id: 0,
        resource_type: ResourceType::CuStream,
    }
}

// ── Marker function for RGPU interpose DLL detection ────────────────
// Used by the installer/daemon to verify that nvcuda.dll in System32
// is the RGPU interpose library and not the real NVIDIA driver.
#[no_mangle]
pub extern "C" fn rgpu_interpose_marker() -> c_int {
    1
}

// ── Exported CUDA Driver API Functions ──────────────────────────────

// ── Initialization ──────────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn cuInit(flags: c_uint) -> CUresult {
    // Initialize logging on first call
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("RGPU_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .try_init();

    info!("cuInit(flags={})", flags);

    match send_cuda_command(CudaCommand::Init {
        flags: flags as u32,
    }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuDriverGetVersion(version: *mut c_int) -> CUresult {
    if version.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }

    match send_cuda_command(CudaCommand::DriverGetVersion) {
        CudaResponse::DriverVersion(v) => {
            *version = v;
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

// ── Device Management ───────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn cuDeviceGetCount(count: *mut c_int) -> CUresult {
    if count.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }

    match send_cuda_command(CudaCommand::DeviceGetCount) {
        CudaResponse::DeviceCount(n) => {
            debug!("cuDeviceGetCount -> {}", n);
            *count = n;
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult {
    if device.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }

    match send_cuda_command(CudaCommand::DeviceGet { ordinal }) {
        CudaResponse::Device(handle) => {
            let local_id = handle_store::store_device(handle);
            *device = local_id as CUdevice;
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuDeviceGetName(
    name: *mut c_char,
    len: c_int,
    device: CUdevice,
) -> CUresult {
    if name.is_null() || len <= 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }

    let dev_handle = match handle_store::get_device(device as u64) {
        Some(h) => h,
        None => return CUDA_ERROR_INVALID_VALUE,
    };

    match send_cuda_command(CudaCommand::DeviceGetName {
        device: dev_handle,
    }) {
        CudaResponse::DeviceName(dev_name) => {
            let bytes = dev_name.as_bytes();
            let copy_len = std::cmp::min(bytes.len(), (len - 1) as usize);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), name as *mut u8, copy_len);
            *name.add(copy_len) = 0;
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuDeviceGetAttribute(
    pi: *mut c_int,
    attrib: c_int,
    device: CUdevice,
) -> CUresult {
    if pi.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }

    let dev_handle = match handle_store::get_device(device as u64) {
        Some(h) => h,
        None => return CUDA_ERROR_INVALID_VALUE,
    };

    match send_cuda_command(CudaCommand::DeviceGetAttribute {
        attrib,
        device: dev_handle,
    }) {
        CudaResponse::DeviceAttribute(val) => {
            *pi = val;
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuDeviceTotalMem_v2(bytes: *mut u64, device: CUdevice) -> CUresult {
    if bytes.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }

    let dev_handle = match handle_store::get_device(device as u64) {
        Some(h) => h,
        None => return CUDA_ERROR_INVALID_VALUE,
    };

    match send_cuda_command(CudaCommand::DeviceTotalMem {
        device: dev_handle,
    }) {
        CudaResponse::DeviceTotalMem(total) => {
            *bytes = total;
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuDeviceComputeCapability(
    major: *mut c_int,
    minor: *mut c_int,
    device: CUdevice,
) -> CUresult {
    if major.is_null() || minor.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }

    let dev_handle = match handle_store::get_device(device as u64) {
        Some(h) => h,
        None => return CUDA_ERROR_INVALID_VALUE,
    };

    match send_cuda_command(CudaCommand::DeviceComputeCapability {
        device: dev_handle,
    }) {
        CudaResponse::ComputeCapability {
            major: maj,
            minor: min,
        } => {
            *major = maj;
            *minor = min;
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

// ── Context Management ──────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn cuCtxCreate_v2(
    pctx: *mut CUcontext,
    flags: c_uint,
    dev: CUdevice,
) -> CUresult {
    if pctx.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }

    debug!("cuCtxCreate_v2(flags={}, dev={})", flags, dev);

    let dev_handle = match handle_store::get_device(dev as u64) {
        Some(h) => h,
        None => return CUDA_ERROR_INVALID_VALUE,
    };

    match send_cuda_command(CudaCommand::CtxCreate {
        flags: flags as u32,
        device: dev_handle,
    }) {
        CudaResponse::Context(handle) => {
            let local_id = handle_store::store_ctx(handle);
            *pctx = local_id as CUcontext;
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuCtxDestroy_v2(ctx: CUcontext) -> CUresult {
    let local_id = ctx as u64;
    let net_handle = match handle_store::get_ctx(local_id) {
        Some(h) => h,
        None => {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    match send_cuda_command(CudaCommand::CtxDestroy { ctx: net_handle }) {
        CudaResponse::Success => {
            handle_store::remove_ctx(local_id);
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult {
    let local_id = ctx as u64;
    let net_handle = match handle_store::get_ctx(local_id) {
        Some(h) => h,
        None => {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    match send_cuda_command(CudaCommand::CtxSetCurrent { ctx: net_handle }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuCtxGetCurrent(pctx: *mut CUcontext) -> CUresult {
    if pctx.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }

    match send_cuda_command(CudaCommand::CtxGetCurrent) {
        CudaResponse::Context(handle) => {
            let local_id = handle_store::store_ctx(handle);
            *pctx = local_id as CUcontext;
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuCtxSynchronize() -> CUresult {
    match send_cuda_command(CudaCommand::CtxSynchronize) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

// ── Module Management ───────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn cuModuleLoadData(
    module: *mut CUmodule,
    image: *const c_void,
) -> CUresult {
    if module.is_null() || image.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }

    let image_data = detect_and_read_module_image(image);

    debug!("cuModuleLoadData({} bytes)", image_data.len());

    match send_cuda_command(CudaCommand::ModuleLoadData { image: image_data }) {
        CudaResponse::Module(handle) => {
            let local_id = handle_store::store_mod(handle);
            *module = local_id as CUmodule;
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuModuleUnload(hmod: CUmodule) -> CUresult {
    let local_id = hmod as u64;
    let net_handle = match handle_store::get_mod(local_id) {
        Some(h) => h,
        None => {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    match send_cuda_command(CudaCommand::ModuleUnload { module: net_handle }) {
        CudaResponse::Success => {
            handle_store::remove_mod(local_id);
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuModuleGetFunction(
    hfunc: *mut CUfunction,
    hmod: CUmodule,
    name: *const c_char,
) -> CUresult {
    if hfunc.is_null() || name.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }

    let local_mod_id = hmod as u64;
    let net_module = match handle_store::get_mod(local_mod_id) {
        Some(h) => h,
        None => {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    let func_name = std::ffi::CStr::from_ptr(name).to_string_lossy().into_owned();
    debug!("cuModuleGetFunction('{}')", func_name);

    match send_cuda_command(CudaCommand::ModuleGetFunction {
        module: net_module,
        name: func_name,
    }) {
        CudaResponse::Function(handle) => {
            let local_id = handle_store::store_func(handle);
            *hfunc = local_id as CUfunction;
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

// ── Memory Management ───────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult {
    if dptr.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }

    debug!("cuMemAlloc_v2({} bytes)", bytesize);

    match send_cuda_command(CudaCommand::MemAlloc {
        byte_size: bytesize as u64,
    }) {
        CudaResponse::MemAllocated(handle) => {
            let local_id = handle_store::store_mem(handle);
            *dptr = local_id;
            debug!("cuMemAlloc_v2 -> local_id=0x{:x}", local_id);
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult {
    let net_handle = match handle_store::get_mem_by_ptr(dptr) {
        Some(h) => h,
        None => {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    match send_cuda_command(CudaCommand::MemFree { dptr: net_handle }) {
        CudaResponse::Success => {
            handle_store::remove_mem(dptr);
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemcpyHtoD_v2(
    dst_device: CUdeviceptr,
    src_host: *const c_void,
    byte_count: usize,
) -> CUresult {
    if src_host.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }

    let net_dst = match handle_store::get_mem_by_ptr(dst_device) {
        Some(h) => h,
        None => {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    let src_data = std::slice::from_raw_parts(src_host as *const u8, byte_count).to_vec();

    debug!("cuMemcpyHtoD_v2({} bytes)", byte_count);

    match send_cuda_command(CudaCommand::MemcpyHtoD {
        dst: net_dst,
        src_data,
        byte_count: byte_count as u64,
    }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemcpyDtoH_v2(
    dst_host: *mut c_void,
    src_device: CUdeviceptr,
    byte_count: usize,
) -> CUresult {
    if dst_host.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }

    let net_src = match handle_store::get_mem_by_ptr(src_device) {
        Some(h) => h,
        None => {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    debug!("cuMemcpyDtoH_v2({} bytes)", byte_count);

    match send_cuda_command(CudaCommand::MemcpyDtoH {
        src: net_src,
        byte_count: byte_count as u64,
    }) {
        CudaResponse::MemoryData(data) => {
            let copy_len = std::cmp::min(data.len(), byte_count);
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst_host as *mut u8, copy_len);
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

// ── Execution Control ───────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn cuLaunchKernel(
    f: CUfunction,
    grid_dim_x: c_uint,
    grid_dim_y: c_uint,
    grid_dim_z: c_uint,
    block_dim_x: c_uint,
    block_dim_y: c_uint,
    block_dim_z: c_uint,
    shared_mem_bytes: c_uint,
    hstream: CUstream,
    kernel_params: *mut *mut c_void,
    _extra: *mut *mut c_void,
) -> CUresult {
    let local_func_id = f as u64;
    let net_func = match handle_store::get_func(local_func_id) {
        Some(h) => h,
        None => {
            error!("cuLaunchKernel: invalid function handle");
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    let stream_id = hstream as u64;
    let net_stream = if stream_id == 0 {
        null_stream_handle()
    } else {
        match handle_store::get_stream(stream_id) {
            Some(h) => h,
            None => null_stream_handle(),
        }
    };

    let mut params = Vec::new();
    if !kernel_params.is_null() {
        let mut i = 0;
        loop {
            let param_ptr = *kernel_params.add(i);
            if param_ptr.is_null() {
                break;
            }
            let data =
                std::slice::from_raw_parts(param_ptr as *const u8, std::mem::size_of::<u64>())
                    .to_vec();
            params.push(KernelParam { data });
            i += 1;
            if i >= 256 {
                break;
            }
        }
    }

    debug!(
        "cuLaunchKernel(grid=[{}x{}x{}], block=[{}x{}x{}], shared={}, params={})",
        grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y, block_dim_z,
        shared_mem_bytes, params.len()
    );

    match send_cuda_command(CudaCommand::LaunchKernel {
        func: net_func,
        grid_dim: [grid_dim_x, grid_dim_y, grid_dim_z],
        block_dim: [block_dim_x, block_dim_y, block_dim_z],
        shared_mem_bytes,
        stream: net_stream,
        kernel_params: params,
    }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

// ── Stream Management ───────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn cuStreamCreate(phstream: *mut CUstream, flags: c_uint) -> CUresult {
    if phstream.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }

    match send_cuda_command(CudaCommand::StreamCreate {
        flags: flags as u32,
    }) {
        CudaResponse::Stream(handle) => {
            let local_id = handle_store::store_stream(handle);
            *phstream = local_id as CUstream;
            debug!("cuStreamCreate -> local_id=0x{:x}", local_id);
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuStreamDestroy_v2(hstream: CUstream) -> CUresult {
    let local_id = hstream as u64;
    let net_handle = match handle_store::get_stream(local_id) {
        Some(h) => h,
        None => {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    match send_cuda_command(CudaCommand::StreamDestroy {
        stream: net_handle,
    }) {
        CudaResponse::Success => {
            handle_store::remove_stream(local_id);
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuStreamSynchronize(hstream: CUstream) -> CUresult {
    let local_id = hstream as u64;

    let net_handle = if local_id == 0 {
        null_stream_handle()
    } else {
        match handle_store::get_stream(local_id) {
            Some(h) => h,
            None => {
                return CUDA_ERROR_INVALID_VALUE;
            }
        }
    };

    match send_cuda_command(CudaCommand::StreamSynchronize {
        stream: net_handle,
    }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuStreamQuery(hstream: CUstream) -> CUresult {
    let local_id = hstream as u64;
    let net_handle = if local_id == 0 {
        null_stream_handle()
    } else {
        match handle_store::get_stream(local_id) {
            Some(h) => h,
            None => {
                return CUDA_ERROR_INVALID_VALUE;
            }
        }
    };

    match send_cuda_command(CudaCommand::StreamQuery {
        stream: net_handle,
    }) {
        CudaResponse::StreamStatus(true) => CUDA_SUCCESS,
        CudaResponse::StreamStatus(false) => CUDA_ERROR_NOT_READY,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

// ── Event Management ────────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn cuEventCreate(phevent: *mut CUevent, flags: c_uint) -> CUresult {
    if phevent.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }

    match send_cuda_command(CudaCommand::EventCreate {
        flags: flags as u32,
    }) {
        CudaResponse::Event(handle) => {
            let local_id = handle_store::store_event(handle);
            *phevent = local_id as CUevent;
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuEventDestroy_v2(hevent: CUevent) -> CUresult {
    let local_id = hevent as u64;
    let net_handle = match handle_store::get_event(local_id) {
        Some(h) => h,
        None => return CUDA_ERROR_INVALID_VALUE,
    };

    match send_cuda_command(CudaCommand::EventDestroy { event: net_handle }) {
        CudaResponse::Success => {
            handle_store::remove_event(local_id);
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuEventRecord(hevent: CUevent, hstream: CUstream) -> CUresult {
    let local_event_id = hevent as u64;
    let net_event = match handle_store::get_event(local_event_id) {
        Some(h) => h,
        None => return CUDA_ERROR_INVALID_VALUE,
    };

    let stream_id = hstream as u64;
    let net_stream = if stream_id == 0 {
        null_stream_handle()
    } else {
        match handle_store::get_stream(stream_id) {
            Some(h) => h,
            None => null_stream_handle(),
        }
    };

    match send_cuda_command(CudaCommand::EventRecord {
        event: net_event,
        stream: net_stream,
    }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuEventSynchronize(hevent: CUevent) -> CUresult {
    let local_id = hevent as u64;
    let net_handle = match handle_store::get_event(local_id) {
        Some(h) => h,
        None => return CUDA_ERROR_INVALID_VALUE,
    };

    match send_cuda_command(CudaCommand::EventSynchronize { event: net_handle }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuEventQuery(hevent: CUevent) -> CUresult {
    let local_id = hevent as u64;
    let net_handle = match handle_store::get_event(local_id) {
        Some(h) => h,
        None => return CUDA_ERROR_INVALID_VALUE,
    };

    match send_cuda_command(CudaCommand::EventQuery { event: net_handle }) {
        CudaResponse::EventStatus(true) => CUDA_SUCCESS,
        CudaResponse::EventStatus(false) => CUDA_ERROR_NOT_READY,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuEventElapsedTime(
    ms: *mut f32,
    hstart: CUevent,
    hend: CUevent,
) -> CUresult {
    if ms.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }

    let net_start = match handle_store::get_event(hstart as u64) {
        Some(h) => h,
        None => return CUDA_ERROR_INVALID_VALUE,
    };
    let net_end = match handle_store::get_event(hend as u64) {
        Some(h) => h,
        None => return CUDA_ERROR_INVALID_VALUE,
    };

    match send_cuda_command(CudaCommand::EventElapsedTime {
        start: net_start,
        end: net_end,
    }) {
        CudaResponse::ElapsedTime(elapsed) => {
            *ms = elapsed;
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

// ── Device Management Extended ───────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn cuDeviceGetUuid(uuid: *mut [u8; 16], dev: CUdevice) -> CUresult {
    if uuid.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let dev_handle = match handle_store::get_device(dev as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::DeviceGetUuid { device: dev_handle }) {
        CudaResponse::DeviceUuid(data) => {
            let dst = &mut *uuid;
            let len = std::cmp::min(data.len(), 16);
            dst[..len].copy_from_slice(&data[..len]);
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuDeviceGetP2PAttribute(value: *mut c_int, attrib: c_int, src: CUdevice, dst: CUdevice) -> CUresult {
    if value.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let src_h = match handle_store::get_device(src as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    let dst_h = match handle_store::get_device(dst as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::DeviceGetP2PAttribute { attrib, src_device: src_h, dst_device: dst_h }) {
        CudaResponse::P2PAttribute(v) => { *value = v; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuDeviceCanAccessPeer(can_access: *mut c_int, dev: CUdevice, peer: CUdevice) -> CUresult {
    if can_access.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let dev_h = match handle_store::get_device(dev as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    let peer_h = match handle_store::get_device(peer as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::DeviceCanAccessPeer { device: dev_h, peer_device: peer_h }) {
        CudaResponse::BoolResult(b) => { *can_access = if b { 1 } else { 0 }; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuDeviceGetByPCIBusId(dev: *mut CUdevice, pci_bus_id: *const c_char) -> CUresult {
    if dev.is_null() || pci_bus_id.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let id_str = std::ffi::CStr::from_ptr(pci_bus_id).to_string_lossy().into_owned();
    match send_cuda_command(CudaCommand::DeviceGetByPCIBusId { pci_bus_id: id_str }) {
        CudaResponse::Device(handle) => {
            let local_id = handle_store::store_device(handle);
            *dev = local_id as CUdevice;
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuDeviceGetPCIBusId(pci_bus_id: *mut c_char, len: c_int, dev: CUdevice) -> CUresult {
    if pci_bus_id.is_null() || len <= 0 { return CUDA_ERROR_INVALID_VALUE; }
    let dev_h = match handle_store::get_device(dev as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::DeviceGetPCIBusId { device: dev_h }) {
        CudaResponse::DevicePCIBusId(id) => {
            let bytes = id.as_bytes();
            let copy_len = std::cmp::min(bytes.len(), (len - 1) as usize);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), pci_bus_id as *mut u8, copy_len);
            *pci_bus_id.add(copy_len) = 0;
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuDeviceGetDefaultMemPool(pool: *mut CUmemoryPool, dev: CUdevice) -> CUresult {
    if pool.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let dev_h = match handle_store::get_device(dev as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::DeviceGetDefaultMemPool { device: dev_h }) {
        CudaResponse::MemPool(handle) => { let id = handle_store::store_mempool(handle); *pool = id as CUmemoryPool; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuDeviceGetMemPool(pool: *mut CUmemoryPool, dev: CUdevice) -> CUresult {
    if pool.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let dev_h = match handle_store::get_device(dev as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::DeviceGetMemPool { device: dev_h }) {
        CudaResponse::MemPool(handle) => { let id = handle_store::store_mempool(handle); *pool = id as CUmemoryPool; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuDeviceSetMemPool(dev: CUdevice, pool: CUmemoryPool) -> CUresult {
    let dev_h = match handle_store::get_device(dev as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    let pool_h = match handle_store::get_mempool(pool as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::DeviceSetMemPool { device: dev_h, mem_pool: pool_h }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

// ── Primary Context ─────────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn cuDevicePrimaryCtxRetain(pctx: *mut CUcontext, dev: CUdevice) -> CUresult {
    if pctx.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let dev_h = match handle_store::get_device(dev as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::DevicePrimaryCtxRetain { device: dev_h }) {
        CudaResponse::Context(handle) => { let id = handle_store::store_ctx(handle); *pctx = id as CUcontext; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuDevicePrimaryCtxRelease_v2(dev: CUdevice) -> CUresult {
    let dev_h = match handle_store::get_device(dev as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::DevicePrimaryCtxRelease { device: dev_h }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuDevicePrimaryCtxReset_v2(dev: CUdevice) -> CUresult {
    let dev_h = match handle_store::get_device(dev as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::DevicePrimaryCtxReset { device: dev_h }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuDevicePrimaryCtxGetState(dev: CUdevice, flags: *mut c_uint, active: *mut c_int) -> CUresult {
    if flags.is_null() || active.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let dev_h = match handle_store::get_device(dev as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::DevicePrimaryCtxGetState { device: dev_h }) {
        CudaResponse::PrimaryCtxState { flags: f, active: a } => { *flags = f; *active = if a { 1 } else { 0 }; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuDevicePrimaryCtxSetFlags_v2(dev: CUdevice, flags: c_uint) -> CUresult {
    let dev_h = match handle_store::get_device(dev as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::DevicePrimaryCtxSetFlags { device: dev_h, flags: flags as u32 }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

// ── Context Management Extended ─────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn cuCtxPushCurrent_v2(ctx: CUcontext) -> CUresult {
    let net_h = match handle_store::get_ctx(ctx as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::CtxPushCurrent { ctx: net_h }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuCtxPopCurrent_v2(pctx: *mut CUcontext) -> CUresult {
    match send_cuda_command(CudaCommand::CtxPopCurrent) {
        CudaResponse::Context(handle) => {
            let id = handle_store::store_ctx(handle);
            if !pctx.is_null() { *pctx = id as CUcontext; }
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuCtxGetDevice(device: *mut CUdevice) -> CUresult {
    if device.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    match send_cuda_command(CudaCommand::CtxGetDevice) {
        CudaResponse::ContextDevice(handle) => {
            let id = handle_store::store_device(handle);
            *device = id as CUdevice;
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuCtxSetCacheConfig(config: c_int) -> CUresult {
    match send_cuda_command(CudaCommand::CtxSetCacheConfig { config }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuCtxGetCacheConfig(config: *mut c_int) -> CUresult {
    if config.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    match send_cuda_command(CudaCommand::CtxGetCacheConfig) {
        CudaResponse::CacheConfig(c) => { *config = c; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuCtxSetLimit(limit: c_int, value: usize) -> CUresult {
    match send_cuda_command(CudaCommand::CtxSetLimit { limit, value: value as u64 }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuCtxGetLimit(pvalue: *mut usize, limit: c_int) -> CUresult {
    if pvalue.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    match send_cuda_command(CudaCommand::CtxGetLimit { limit }) {
        CudaResponse::ContextLimit(v) => { *pvalue = v as usize; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuCtxGetStreamPriorityRange(least: *mut c_int, greatest: *mut c_int) -> CUresult {
    match send_cuda_command(CudaCommand::CtxGetStreamPriorityRange) {
        CudaResponse::StreamPriorityRange { least: l, greatest: g } => {
            if !least.is_null() { *least = l; }
            if !greatest.is_null() { *greatest = g; }
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuCtxGetApiVersion(ctx: CUcontext, version: *mut c_uint) -> CUresult {
    if version.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let net_h = match handle_store::get_ctx(ctx as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::CtxGetApiVersion { ctx: net_h }) {
        CudaResponse::ContextApiVersion(v) => { *version = v; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuCtxGetFlags(flags: *mut c_uint) -> CUresult {
    if flags.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    match send_cuda_command(CudaCommand::CtxGetFlags) {
        CudaResponse::ContextFlags(f) => { *flags = f; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuCtxSetFlags(flags: c_uint) -> CUresult {
    match send_cuda_command(CudaCommand::CtxSetFlags { flags: flags as u32 }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuCtxResetPersistingL2Cache() -> CUresult {
    match send_cuda_command(CudaCommand::CtxResetPersistingL2Cache) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

// ── Peer Access ─────────────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn cuCtxEnablePeerAccess(peer_ctx: CUcontext, flags: c_uint) -> CUresult {
    let net_h = match handle_store::get_ctx(peer_ctx as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::CtxEnablePeerAccess { peer_ctx: net_h, flags: flags as u32 }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuCtxDisablePeerAccess(peer_ctx: CUcontext) -> CUresult {
    let net_h = match handle_store::get_ctx(peer_ctx as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::CtxDisablePeerAccess { peer_ctx: net_h }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

// ── Module Management Extended ──────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn cuModuleLoad(module: *mut CUmodule, fname: *const c_char) -> CUresult {
    if module.is_null() || fname.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let name = std::ffi::CStr::from_ptr(fname).to_string_lossy().into_owned();
    match send_cuda_command(CudaCommand::ModuleLoad { fname: name }) {
        CudaResponse::Module(handle) => { let id = handle_store::store_mod(handle); *module = id as CUmodule; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuModuleLoadDataEx(
    module: *mut CUmodule, image: *const c_void,
    _num_options: c_uint, _options: *mut c_int, _option_values: *mut *mut c_void,
) -> CUresult {
    if module.is_null() || image.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let image_data = detect_and_read_module_image(image);
    match send_cuda_command(CudaCommand::ModuleLoadDataEx { image: image_data, num_options: 0, options: vec![], option_values: vec![] }) {
        CudaResponse::Module(handle) => { let id = handle_store::store_mod(handle); *module = id as CUmodule; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuModuleLoadFatBinary(module: *mut CUmodule, fat_cubin: *const c_void) -> CUresult {
    if module.is_null() || fat_cubin.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let image_data = detect_and_read_module_image(fat_cubin);
    match send_cuda_command(CudaCommand::ModuleLoadFatBinary { fat_cubin: image_data }) {
        CudaResponse::Module(handle) => { let id = handle_store::store_mod(handle); *module = id as CUmodule; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuModuleGetGlobal_v2(dptr: *mut CUdeviceptr, bytes: *mut usize, hmod: CUmodule, name: *const c_char) -> CUresult {
    if name.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let net_mod = match handle_store::get_mod(hmod as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    let func_name = std::ffi::CStr::from_ptr(name).to_string_lossy().into_owned();
    match send_cuda_command(CudaCommand::ModuleGetGlobal { module: net_mod, name: func_name }) {
        CudaResponse::GlobalPtr { ptr, size } => {
            if !dptr.is_null() { let id = handle_store::store_mem(ptr); *dptr = id; }
            if !bytes.is_null() { *bytes = size as usize; }
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

// ── Linker ──────────────────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn cuLinkCreate_v2(
    _num_options: c_uint, _options: *mut c_int, _option_values: *mut *mut c_void,
    state: *mut CUlinkState,
) -> CUresult {
    if state.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    match send_cuda_command(CudaCommand::LinkCreate { num_options: 0, options: vec![], option_values: vec![] }) {
        CudaResponse::Linker(handle) => { let id = handle_store::store_linker(handle); *state = id as CUlinkState; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuLinkAddData_v2(
    state: CUlinkState, jit_type: c_int, data: *mut c_void, size: usize,
    name: *const c_char, _num_options: c_uint, _options: *mut c_int, _option_values: *mut *mut c_void,
) -> CUresult {
    let net_link = match handle_store::get_linker(state as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    let data_vec = if !data.is_null() && size > 0 {
        std::slice::from_raw_parts(data as *const u8, size).to_vec()
    } else { vec![] };
    let name_str = if !name.is_null() { std::ffi::CStr::from_ptr(name).to_string_lossy().into_owned() } else { String::new() };
    match send_cuda_command(CudaCommand::LinkAddData { link: net_link, jit_type, data: data_vec, name: name_str, num_options: 0, options: vec![], option_values: vec![] }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuLinkAddFile_v2(
    state: CUlinkState, jit_type: c_int, path: *const c_char,
    _num_options: c_uint, _options: *mut c_int, _option_values: *mut *mut c_void,
) -> CUresult {
    if path.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let net_link = match handle_store::get_linker(state as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    let path_str = std::ffi::CStr::from_ptr(path).to_string_lossy().into_owned();
    match send_cuda_command(CudaCommand::LinkAddFile { link: net_link, jit_type, path: path_str, num_options: 0, options: vec![], option_values: vec![] }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuLinkComplete(state: CUlinkState, cubin_out: *mut *mut c_void, size_out: *mut usize) -> CUresult {
    let net_link = match handle_store::get_linker(state as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::LinkComplete { link: net_link }) {
        CudaResponse::LinkCompleted { cubin_data } => {
            // Leak the data so the pointer stays valid
            let boxed = cubin_data.into_boxed_slice();
            let len = boxed.len();
            let ptr = Box::into_raw(boxed) as *mut c_void;
            if !cubin_out.is_null() { *cubin_out = ptr; }
            if !size_out.is_null() { *size_out = len; }
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuLinkDestroy(state: CUlinkState) -> CUresult {
    let net_link = match handle_store::get_linker(state as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::LinkDestroy { link: net_link }) {
        CudaResponse::Success => { handle_store::remove_linker(state as u64); CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

// ── Memory Management Extended ──────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn cuMemcpyDtoD_v2(dst: CUdeviceptr, src: CUdeviceptr, byte_count: usize) -> CUresult {
    let net_dst = match handle_store::get_mem_by_ptr(dst) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    let net_src = match handle_store::get_mem_by_ptr(src) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::MemcpyDtoD { dst: net_dst, src: net_src, byte_count: byte_count as u64 }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemcpyHtoDAsync_v2(dst: CUdeviceptr, src: *const c_void, byte_count: usize, hstream: CUstream) -> CUresult {
    if src.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let net_dst = match handle_store::get_mem_by_ptr(dst) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    let net_stream = if (hstream as u64) == 0 { null_stream_handle() } else { handle_store::get_stream(hstream as u64).unwrap_or_else(null_stream_handle) };
    let src_data = std::slice::from_raw_parts(src as *const u8, byte_count).to_vec();
    match send_cuda_command(CudaCommand::MemcpyHtoDAsync { dst: net_dst, src_data, byte_count: byte_count as u64, stream: net_stream }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemcpyDtoHAsync_v2(dst: *mut c_void, src: CUdeviceptr, byte_count: usize, hstream: CUstream) -> CUresult {
    if dst.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let net_src = match handle_store::get_mem_by_ptr(src) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    let net_stream = if (hstream as u64) == 0 { null_stream_handle() } else { handle_store::get_stream(hstream as u64).unwrap_or_else(null_stream_handle) };
    match send_cuda_command(CudaCommand::MemcpyDtoHAsync { src: net_src, byte_count: byte_count as u64, stream: net_stream }) {
        CudaResponse::MemoryData(data) => {
            let copy_len = std::cmp::min(data.len(), byte_count);
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst as *mut u8, copy_len);
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemcpyDtoDAsync_v2(dst: CUdeviceptr, src: CUdeviceptr, byte_count: usize, hstream: CUstream) -> CUresult {
    let net_dst = match handle_store::get_mem_by_ptr(dst) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    let net_src = match handle_store::get_mem_by_ptr(src) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    let net_stream = if (hstream as u64) == 0 { null_stream_handle() } else { handle_store::get_stream(hstream as u64).unwrap_or_else(null_stream_handle) };
    match send_cuda_command(CudaCommand::MemcpyDtoDAsync { dst: net_dst, src: net_src, byte_count: byte_count as u64, stream: net_stream }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemsetD8_v2(dst: CUdeviceptr, value: u8, count: usize) -> CUresult {
    let net_dst = match handle_store::get_mem_by_ptr(dst) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::MemsetD8 { dst: net_dst, value, count: count as u64 }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemsetD16_v2(dst: CUdeviceptr, value: u16, count: usize) -> CUresult {
    let net_dst = match handle_store::get_mem_by_ptr(dst) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::MemsetD16 { dst: net_dst, value, count: count as u64 }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemsetD32_v2(dst: CUdeviceptr, value: u32, count: usize) -> CUresult {
    let net_dst = match handle_store::get_mem_by_ptr(dst) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::MemsetD32 { dst: net_dst, value, count: count as u64 }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> CUresult {
    match send_cuda_command(CudaCommand::MemGetInfo) {
        CudaResponse::MemInfo { free: f, total: t } => {
            if !free.is_null() { *free = f as usize; }
            if !total.is_null() { *total = t as usize; }
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemGetAddressRange_v2(pbase: *mut CUdeviceptr, psize: *mut usize, dptr: CUdeviceptr) -> CUresult {
    let net_ptr = match handle_store::get_mem_by_ptr(dptr) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::MemGetAddressRange { dptr: net_ptr }) {
        CudaResponse::MemAddressRange { base, size } => {
            if !pbase.is_null() { let id = handle_store::store_mem(base); *pbase = id; }
            if !psize.is_null() { *psize = size as usize; }
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemAllocHost_v2(pp: *mut *mut c_void, bytesize: usize) -> CUresult {
    if pp.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    match send_cuda_command(CudaCommand::MemAllocHost { byte_size: bytesize as u64 }) {
        CudaResponse::HostPtr(handle) => {
            let id = handle_store::store_host_mem(handle);
            *pp = id as *mut c_void;
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemFreeHost(p: *mut c_void) -> CUresult {
    let net_h = match handle_store::get_host_mem(p as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::MemFreeHost { ptr: net_h }) {
        CudaResponse::Success => { handle_store::remove_host_mem(p as u64); CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemHostAlloc(pp: *mut *mut c_void, bytesize: usize, flags: c_uint) -> CUresult {
    if pp.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    match send_cuda_command(CudaCommand::MemHostAlloc { byte_size: bytesize as u64, flags: flags as u32 }) {
        CudaResponse::HostPtr(handle) => {
            let id = handle_store::store_host_mem(handle);
            *pp = id as *mut c_void;
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemHostGetDevicePointer_v2(pdptr: *mut CUdeviceptr, p: *mut c_void, flags: c_uint) -> CUresult {
    if pdptr.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let net_h = match handle_store::get_host_mem(p as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::MemHostGetDevicePointer { host_ptr: net_h, flags: flags as u32 }) {
        CudaResponse::HostDevicePtr(handle) => { let id = handle_store::store_mem(handle); *pdptr = id; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemHostGetFlags(pflags: *mut c_uint, p: *mut c_void) -> CUresult {
    if pflags.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let net_h = match handle_store::get_host_mem(p as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::MemHostGetFlags { host_ptr: net_h }) {
        CudaResponse::HostFlags(f) => { *pflags = f; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemAllocManaged(dptr: *mut CUdeviceptr, bytesize: usize, flags: c_uint) -> CUresult {
    if dptr.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    match send_cuda_command(CudaCommand::MemAllocManaged { byte_size: bytesize as u64, flags: flags as u32 }) {
        CudaResponse::MemAllocated(handle) => { let id = handle_store::store_mem(handle); *dptr = id; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemAllocPitch_v2(dptr: *mut CUdeviceptr, ppitch: *mut usize, width: usize, height: usize, element_size: c_uint) -> CUresult {
    if dptr.is_null() || ppitch.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    match send_cuda_command(CudaCommand::MemAllocPitch { width: width as u64, height: height as u64, element_size }) {
        CudaResponse::MemAllocPitch { dptr: handle, pitch } => {
            let id = handle_store::store_mem(handle);
            *dptr = id;
            *ppitch = pitch as usize;
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

// ── Execution Control Extended ──────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn cuLaunchCooperativeKernel(
    f: CUfunction,
    grid_dim_x: c_uint, grid_dim_y: c_uint, grid_dim_z: c_uint,
    block_dim_x: c_uint, block_dim_y: c_uint, block_dim_z: c_uint,
    shared_mem_bytes: c_uint, hstream: CUstream,
    kernel_params: *mut *mut c_void,
) -> CUresult {
    let net_func = match handle_store::get_func(f as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    let net_stream = if (hstream as u64) == 0 { null_stream_handle() } else { handle_store::get_stream(hstream as u64).unwrap_or_else(null_stream_handle) };

    let mut params = Vec::new();
    if !kernel_params.is_null() {
        let mut i = 0;
        loop {
            let param_ptr = *kernel_params.add(i);
            if param_ptr.is_null() { break; }
            let data = std::slice::from_raw_parts(param_ptr as *const u8, std::mem::size_of::<u64>()).to_vec();
            params.push(KernelParam { data });
            i += 1;
            if i >= 256 { break; }
        }
    }

    match send_cuda_command(CudaCommand::LaunchCooperativeKernel {
        func: net_func, grid_dim: [grid_dim_x, grid_dim_y, grid_dim_z],
        block_dim: [block_dim_x, block_dim_y, block_dim_z],
        shared_mem_bytes, stream: net_stream, kernel_params: params,
    }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuFuncGetAttribute(pi: *mut c_int, attrib: c_int, hfunc: CUfunction) -> CUresult {
    if pi.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let net_func = match handle_store::get_func(hfunc as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::FuncGetAttribute { attrib, func: net_func }) {
        CudaResponse::FuncAttribute(v) => { *pi = v; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuFuncSetAttribute(hfunc: CUfunction, attrib: c_int, value: c_int) -> CUresult {
    let net_func = match handle_store::get_func(hfunc as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::FuncSetAttribute { attrib, func: net_func, value }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuFuncSetCacheConfig(hfunc: CUfunction, config: c_int) -> CUresult {
    let net_func = match handle_store::get_func(hfunc as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::FuncSetCacheConfig { func: net_func, config }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuFuncSetSharedMemConfig(hfunc: CUfunction, config: c_int) -> CUresult {
    let net_func = match handle_store::get_func(hfunc as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::FuncSetSharedMemConfig { func: net_func, config }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuFuncGetModule(hmod: *mut CUmodule, hfunc: CUfunction) -> CUresult {
    if hmod.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let net_func = match handle_store::get_func(hfunc as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::FuncGetModule { func: net_func }) {
        CudaResponse::FuncModule(handle) => { let id = handle_store::store_mod(handle); *hmod = id as CUmodule; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuFuncGetName(name: *mut *const c_char, hfunc: CUfunction) -> CUresult {
    if name.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let net_func = match handle_store::get_func(hfunc as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::FuncGetName { func: net_func }) {
        CudaResponse::FuncName(n) => {
            let c_str = std::ffi::CString::new(n).unwrap_or_default();
            *name = c_str.into_raw();
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuOccupancyMaxActiveBlocksPerMultiprocessor(num_blocks: *mut c_int, func: CUfunction, block_size: c_int, dynamic_smem_size: usize) -> CUresult {
    if num_blocks.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let net_func = match handle_store::get_func(func as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::OccupancyMaxActiveBlocksPerMultiprocessor { func: net_func, block_size, dynamic_smem_size: dynamic_smem_size as u64 }) {
        CudaResponse::OccupancyBlocks(b) => { *num_blocks = b; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(num_blocks: *mut c_int, func: CUfunction, block_size: c_int, dynamic_smem_size: usize, flags: c_uint) -> CUresult {
    if num_blocks.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let net_func = match handle_store::get_func(func as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::OccupancyMaxActiveBlocksPerMultiprocessorWithFlags { func: net_func, block_size, dynamic_smem_size: dynamic_smem_size as u64, flags: flags as u32 }) {
        CudaResponse::OccupancyBlocks(b) => { *num_blocks = b; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuOccupancyAvailableDynamicSMemPerBlock(dynamic_smem_size: *mut usize, func: CUfunction, num_blocks: c_int, block_size: c_int) -> CUresult {
    if dynamic_smem_size.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let net_func = match handle_store::get_func(func as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::OccupancyAvailableDynamicSMemPerBlock { func: net_func, num_blocks, block_size }) {
        CudaResponse::OccupancyDynamicSmem(s) => { *dynamic_smem_size = s as usize; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

// ── Stream Management Extended ──────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn cuStreamCreateWithPriority(phstream: *mut CUstream, flags: c_uint, priority: c_int) -> CUresult {
    if phstream.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    match send_cuda_command(CudaCommand::StreamCreateWithPriority { flags: flags as u32, priority }) {
        CudaResponse::Stream(handle) => { let id = handle_store::store_stream(handle); *phstream = id as CUstream; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuStreamWaitEvent(hstream: CUstream, hevent: CUevent, flags: c_uint) -> CUresult {
    let net_stream = if (hstream as u64) == 0 { null_stream_handle() } else { match handle_store::get_stream(hstream as u64) { Some(h) => h, None => null_stream_handle() } };
    let net_event = match handle_store::get_event(hevent as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::StreamWaitEvent { stream: net_stream, event: net_event, flags: flags as u32 }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuStreamGetPriority(hstream: CUstream, priority: *mut c_int) -> CUresult {
    if priority.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let net_stream = match handle_store::get_stream(hstream as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::StreamGetPriority { stream: net_stream }) {
        CudaResponse::StreamPriority(p) => { *priority = p; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuStreamGetFlags(hstream: CUstream, flags: *mut c_uint) -> CUresult {
    if flags.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let net_stream = match handle_store::get_stream(hstream as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::StreamGetFlags { stream: net_stream }) {
        CudaResponse::StreamFlags(f) => { *flags = f; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuStreamGetCtx_v2(hstream: CUstream, pctx: *mut CUcontext) -> CUresult {
    if pctx.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let net_stream = match handle_store::get_stream(hstream as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::StreamGetCtx { stream: net_stream }) {
        CudaResponse::StreamCtx(handle) => { let id = handle_store::store_ctx(handle); *pctx = id as CUcontext; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

// ── Event Management Extended ───────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn cuEventRecordWithFlags(hevent: CUevent, hstream: CUstream, flags: c_uint) -> CUresult {
    let net_event = match handle_store::get_event(hevent as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    let net_stream = if (hstream as u64) == 0 { null_stream_handle() } else { handle_store::get_stream(hstream as u64).unwrap_or_else(null_stream_handle) };
    match send_cuda_command(CudaCommand::EventRecordWithFlags { event: net_event, stream: net_stream, flags: flags as u32 }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

// ── Pointer Queries ─────────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn cuPointerGetAttribute(data: *mut c_void, attribute: c_int, ptr: CUdeviceptr) -> CUresult {
    if data.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let net_ptr = match handle_store::get_mem_by_ptr(ptr) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::PointerGetAttribute { attribute, ptr: net_ptr }) {
        CudaResponse::PointerAttribute(v) => { *(data as *mut u64) = v; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuPointerSetAttribute(value: *const c_void, attribute: c_int, ptr: CUdeviceptr) -> CUresult {
    let net_ptr = match handle_store::get_mem_by_ptr(ptr) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    let val = if !value.is_null() { *(value as *const u64) } else { 0 };
    match send_cuda_command(CudaCommand::PointerSetAttribute { attribute, ptr: net_ptr, value: val }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

// ── Memory Pools ────────────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn cuMemPoolDestroy(pool: CUmemoryPool) -> CUresult {
    let net_pool = match handle_store::get_mempool(pool as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::MemPoolDestroy { pool: net_pool }) {
        CudaResponse::Success => { handle_store::remove_mempool(pool as u64); CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemPoolTrimTo(pool: CUmemoryPool, min_bytes_to_keep: usize) -> CUresult {
    let net_pool = match handle_store::get_mempool(pool as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::MemPoolTrimTo { pool: net_pool, min_bytes_to_keep: min_bytes_to_keep as u64 }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemAllocAsync(dptr: *mut CUdeviceptr, bytesize: usize, hstream: CUstream) -> CUresult {
    if dptr.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let net_stream = if (hstream as u64) == 0 { null_stream_handle() } else { handle_store::get_stream(hstream as u64).unwrap_or_else(null_stream_handle) };
    match send_cuda_command(CudaCommand::MemAllocAsync { byte_size: bytesize as u64, stream: net_stream }) {
        CudaResponse::MemAllocated(handle) => { let id = handle_store::store_mem(handle); *dptr = id; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemFreeAsync(dptr: CUdeviceptr, hstream: CUstream) -> CUresult {
    let net_ptr = match handle_store::get_mem_by_ptr(dptr) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    let net_stream = if (hstream as u64) == 0 { null_stream_handle() } else { handle_store::get_stream(hstream as u64).unwrap_or_else(null_stream_handle) };
    match send_cuda_command(CudaCommand::MemFreeAsync { dptr: net_ptr, stream: net_stream }) {
        CudaResponse::Success => { handle_store::remove_mem(dptr); CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemAllocFromPoolAsync(dptr: *mut CUdeviceptr, bytesize: usize, pool: CUmemoryPool, hstream: CUstream) -> CUresult {
    if dptr.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let net_pool = match handle_store::get_mempool(pool as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    let net_stream = if (hstream as u64) == 0 { null_stream_handle() } else { handle_store::get_stream(hstream as u64).unwrap_or_else(null_stream_handle) };
    match send_cuda_command(CudaCommand::MemAllocFromPoolAsync { byte_size: bytesize as u64, pool: net_pool, stream: net_stream }) {
        CudaResponse::MemAllocated(handle) => { let id = handle_store::store_mem(handle); *dptr = id; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

// ── Helper Functions ────────────────────────────────────────────────

/// Detect whether a module image is PTX (text) or cubin (binary) and return the data.
unsafe fn detect_and_read_module_image(image: *const c_void) -> Vec<u8> {
    let ptr = image as *const u8;

    // Check for ELF magic (cubin files)
    let elf_magic = [0x7f, b'E', b'L', b'F'];
    let first_four = std::slice::from_raw_parts(ptr, 4);

    if first_four == elf_magic {
        let max_size = 64 * 1024 * 1024; // 64 MB max
        if let Some(size) = parse_elf_size(ptr) {
            return std::slice::from_raw_parts(ptr, size).to_vec();
        }
        return std::slice::from_raw_parts(ptr, max_size).to_vec();
    }

    // Assume PTX (null-terminated string)
    let c_str = std::ffi::CStr::from_ptr(image as *const c_char);
    c_str.to_bytes_with_nul().to_vec()
}

/// Try to parse the total size of an ELF binary.
unsafe fn parse_elf_size(ptr: *const u8) -> Option<usize> {
    if *ptr.add(4) != 2 {
        return None;
    }

    let e_shoff = *(ptr.add(40) as *const u64);
    let e_shentsize = *(ptr.add(58) as *const u16) as u64;
    let e_shnum = *(ptr.add(60) as *const u16) as u64;

    let total = e_shoff + (e_shentsize * e_shnum);
    Some(total as usize)
}
