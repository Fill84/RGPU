//! CUDA Device Management API functions.

use std::ffi::{c_char, c_int, c_uint};
use tracing::{debug, info};
use rgpu_protocol::cuda_commands::{CudaCommand, CudaResponse};

use crate::{
    CUresult, CUdevice, CUmemoryPool,
    CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_UNKNOWN,
    send_cuda_command, handle_store,
};

// ── Initialization ──────────────────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn cuInit_impl(flags: c_uint) -> CUresult {
    // Initialize logging on first call
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("RGPU_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .try_init();

    info!("cuInit(flags={})", flags);

    // Also initialize the real CUDA driver (for hybrid passthrough functions)
    if let Some(real_init) = crate::real_cuda_proc_address("cuInit") {
        let real_init: unsafe extern "C" fn(u32) -> i32 = std::mem::transmute(real_init);
        let ret = real_init(flags);
        if ret != 0 {
            tracing::warn!("real cuInit returned {}", ret);
        } else {
            debug!("real CUDA driver initialized successfully");
        }
    }

    // Retry with backoff to handle bootstrap race: the daemon's IPC listener
    // may not be ready yet when applications call cuInit early on startup.
    let max_retries = 10u32;
    let mut delay_ms = 100u64;

    for attempt in 0..max_retries {
        match send_cuda_command(CudaCommand::Init {
            flags: flags as u32,
        }) {
            CudaResponse::Success => return CUDA_SUCCESS,
            CudaResponse::Error { code, .. } if attempt < max_retries - 1 => {
                debug!("cuInit attempt {} failed (code {}), retrying in {}ms", attempt + 1, code, delay_ms);
                std::thread::sleep(std::time::Duration::from_millis(delay_ms));
                delay_ms = (delay_ms * 2).min(2000); // exponential backoff, max 2s
            }
            CudaResponse::Error { code, .. } => return code,
            _ => return CUDA_ERROR_UNKNOWN,
        }
    }
    CUDA_ERROR_UNKNOWN
}

#[no_mangle]
pub unsafe extern "C" fn cuInit(flags: c_uint) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuInit_impl(flags))
}

#[allow(non_snake_case)]
unsafe fn cuDriverGetVersion_impl(version: *mut c_int) -> CUresult {
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

#[no_mangle]
pub unsafe extern "C" fn cuDriverGetVersion(version: *mut c_int) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuDriverGetVersion_impl(version))
}

// ── Device Management ───────────────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn cuDeviceGetCount_impl(count: *mut c_int) -> CUresult {
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
pub unsafe extern "C" fn cuDeviceGetCount(count: *mut c_int) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuDeviceGetCount_impl(count))
}

#[allow(non_snake_case)]
unsafe fn cuDeviceGet_impl(device: *mut CUdevice, ordinal: c_int) -> CUresult {
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
pub unsafe extern "C" fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuDeviceGet_impl(device, ordinal))
}

#[allow(non_snake_case)]
unsafe fn cuDeviceGetName_impl(
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
pub unsafe extern "C" fn cuDeviceGetName(
    name: *mut c_char,
    len: c_int,
    device: CUdevice,
) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuDeviceGetName_impl(name, len, device))
}

#[allow(non_snake_case)]
unsafe fn cuDeviceGetAttribute_impl(
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
pub unsafe extern "C" fn cuDeviceGetAttribute(
    pi: *mut c_int,
    attrib: c_int,
    device: CUdevice,
) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuDeviceGetAttribute_impl(pi, attrib, device))
}

#[allow(non_snake_case)]
unsafe fn cuDeviceTotalMem_v2_impl(bytes: *mut u64, device: CUdevice) -> CUresult {
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
pub unsafe extern "C" fn cuDeviceTotalMem_v2(bytes: *mut u64, device: CUdevice) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuDeviceTotalMem_v2_impl(bytes, device))
}

#[allow(non_snake_case)]
unsafe fn cuDeviceComputeCapability_impl(
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

#[no_mangle]
pub unsafe extern "C" fn cuDeviceComputeCapability(
    major: *mut c_int,
    minor: *mut c_int,
    device: CUdevice,
) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuDeviceComputeCapability_impl(major, minor, device))
}

// ── Device Management Extended ───────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn cuDeviceGetUuid_impl(uuid: *mut [u8; 16], dev: CUdevice) -> CUresult {
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
pub unsafe extern "C" fn cuDeviceGetUuid(uuid: *mut [u8; 16], dev: CUdevice) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuDeviceGetUuid_impl(uuid, dev))
}

#[allow(non_snake_case)]
unsafe fn cuDeviceGetP2PAttribute_impl(value: *mut c_int, attrib: c_int, src: CUdevice, dst: CUdevice) -> CUresult {
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
pub unsafe extern "C" fn cuDeviceGetP2PAttribute(value: *mut c_int, attrib: c_int, src: CUdevice, dst: CUdevice) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuDeviceGetP2PAttribute_impl(value, attrib, src, dst))
}

#[allow(non_snake_case)]
unsafe fn cuDeviceCanAccessPeer_impl(can_access: *mut c_int, dev: CUdevice, peer: CUdevice) -> CUresult {
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
pub unsafe extern "C" fn cuDeviceCanAccessPeer(can_access: *mut c_int, dev: CUdevice, peer: CUdevice) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuDeviceCanAccessPeer_impl(can_access, dev, peer))
}

#[allow(non_snake_case)]
unsafe fn cuDeviceGetByPCIBusId_impl(dev: *mut CUdevice, pci_bus_id: *const c_char) -> CUresult {
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
pub unsafe extern "C" fn cuDeviceGetByPCIBusId(dev: *mut CUdevice, pci_bus_id: *const c_char) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuDeviceGetByPCIBusId_impl(dev, pci_bus_id))
}

#[allow(non_snake_case)]
unsafe fn cuDeviceGetPCIBusId_impl(pci_bus_id: *mut c_char, len: c_int, dev: CUdevice) -> CUresult {
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
pub unsafe extern "C" fn cuDeviceGetPCIBusId(pci_bus_id: *mut c_char, len: c_int, dev: CUdevice) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuDeviceGetPCIBusId_impl(pci_bus_id, len, dev))
}

#[allow(non_snake_case)]
unsafe fn cuDeviceGetDefaultMemPool_impl(pool: *mut CUmemoryPool, dev: CUdevice) -> CUresult {
    if pool.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let dev_h = match handle_store::get_device(dev as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::DeviceGetDefaultMemPool { device: dev_h }) {
        CudaResponse::MemPool(handle) => { let id = handle_store::store_mempool(handle); *pool = id as CUmemoryPool; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuDeviceGetDefaultMemPool(pool: *mut CUmemoryPool, dev: CUdevice) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuDeviceGetDefaultMemPool_impl(pool, dev))
}

#[allow(non_snake_case)]
unsafe fn cuDeviceGetMemPool_impl(pool: *mut CUmemoryPool, dev: CUdevice) -> CUresult {
    if pool.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let dev_h = match handle_store::get_device(dev as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::DeviceGetMemPool { device: dev_h }) {
        CudaResponse::MemPool(handle) => { let id = handle_store::store_mempool(handle); *pool = id as CUmemoryPool; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuDeviceGetMemPool(pool: *mut CUmemoryPool, dev: CUdevice) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuDeviceGetMemPool_impl(pool, dev))
}

#[allow(non_snake_case)]
unsafe fn cuDeviceSetMemPool_impl(dev: CUdevice, pool: CUmemoryPool) -> CUresult {
    let dev_h = match handle_store::get_device(dev as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    let pool_h = match handle_store::get_mempool(pool as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::DeviceSetMemPool { device: dev_h, mem_pool: pool_h }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuDeviceSetMemPool(dev: CUdevice, pool: CUmemoryPool) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuDeviceSetMemPool_impl(dev, pool))
}

// ── Unversioned Export Alias ───────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn cuDeviceTotalMem_impl(bytes: *mut u64, device: CUdevice) -> CUresult {
    cuDeviceTotalMem_v2(bytes, device)
}

#[no_mangle]
pub unsafe extern "C" fn cuDeviceTotalMem(bytes: *mut u64, device: CUdevice) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuDeviceTotalMem_impl(bytes, device))
}
