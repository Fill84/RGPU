//! Dynamic loading of the real CUDA driver library.
//!
//! Uses `libloading` to load `nvcuda.dll` (Windows) or `libcuda.so.1` (Linux)
//! and provides safe Rust wrappers around the raw CUDA driver API functions.

use std::ffi::{c_char, c_int, c_uint, c_void, CStr};
use std::sync::Arc;

use libloading::{Library, Symbol};
use tracing::{debug, info};

/// CUDA result type (CUresult).
pub type CUresult = c_int;

/// CUDA device type.
pub type CUdevice = c_int;

/// Opaque CUDA types (represented as pointers).
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUdeviceptr = u64;
pub type CUstream = *mut c_void;
pub type CUevent = *mut c_void;
pub type CUlinkState = *mut c_void;
pub type CUmemoryPool = *mut c_void;

pub const CUDA_SUCCESS: CUresult = 0;
pub const CUDA_ERROR_NOT_SUPPORTED: CUresult = 801;

/// UUID structure (16 bytes).
#[repr(C)]
pub struct CUuuid {
    pub bytes: [u8; 16],
}

/// Function pointer type definitions for the CUDA driver API.
type FnCuInit = unsafe extern "C" fn(flags: c_uint) -> CUresult;
type FnCuDriverGetVersion = unsafe extern "C" fn(version: *mut c_int) -> CUresult;
type FnCuDeviceGetCount = unsafe extern "C" fn(count: *mut c_int) -> CUresult;
type FnCuDeviceGet = unsafe extern "C" fn(device: *mut CUdevice, ordinal: c_int) -> CUresult;
type FnCuDeviceGetName =
    unsafe extern "C" fn(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult;
type FnCuDeviceGetAttribute =
    unsafe extern "C" fn(pi: *mut c_int, attrib: c_int, dev: CUdevice) -> CUresult;
type FnCuDeviceTotalMem =
    unsafe extern "C" fn(bytes: *mut usize, dev: CUdevice) -> CUresult;
type FnCuDeviceComputeCapability =
    unsafe extern "C" fn(major: *mut c_int, minor: *mut c_int, dev: CUdevice) -> CUresult;
type FnCuDeviceGetUuid = unsafe extern "C" fn(uuid: *mut CUuuid, dev: CUdevice) -> CUresult;
type FnCuDeviceGetPCIBusId = unsafe extern "C" fn(pci_bus_id: *mut c_char, len: c_int, dev: CUdevice) -> CUresult;
type FnCuDeviceGetByPCIBusId = unsafe extern "C" fn(dev: *mut CUdevice, pci_bus_id: *const c_char) -> CUresult;
type FnCuDeviceCanAccessPeer = unsafe extern "C" fn(can_access: *mut c_int, dev: CUdevice, peer_dev: CUdevice) -> CUresult;
type FnCuDeviceGetP2PAttribute = unsafe extern "C" fn(value: *mut c_int, attrib: c_int, src: CUdevice, dst: CUdevice) -> CUresult;

// Primary context
type FnCuDevicePrimaryCtxRetain = unsafe extern "C" fn(pctx: *mut CUcontext, dev: CUdevice) -> CUresult;
type FnCuDevicePrimaryCtxRelease = unsafe extern "C" fn(dev: CUdevice) -> CUresult;
type FnCuDevicePrimaryCtxReset = unsafe extern "C" fn(dev: CUdevice) -> CUresult;
type FnCuDevicePrimaryCtxGetState = unsafe extern "C" fn(dev: CUdevice, flags: *mut c_uint, active: *mut c_int) -> CUresult;
type FnCuDevicePrimaryCtxSetFlags = unsafe extern "C" fn(dev: CUdevice, flags: c_uint) -> CUresult;

// Context management
type FnCuCtxCreate =
    unsafe extern "C" fn(pctx: *mut CUcontext, flags: c_uint, dev: CUdevice) -> CUresult;
type FnCuCtxDestroy = unsafe extern "C" fn(ctx: CUcontext) -> CUresult;
type FnCuCtxSetCurrent = unsafe extern "C" fn(ctx: CUcontext) -> CUresult;
type FnCuCtxGetCurrent = unsafe extern "C" fn(pctx: *mut CUcontext) -> CUresult;
type FnCuCtxSynchronize = unsafe extern "C" fn() -> CUresult;
type FnCuCtxPushCurrent = unsafe extern "C" fn(ctx: CUcontext) -> CUresult;
type FnCuCtxPopCurrent = unsafe extern "C" fn(pctx: *mut CUcontext) -> CUresult;
type FnCuCtxGetDevice = unsafe extern "C" fn(device: *mut CUdevice) -> CUresult;
type FnCuCtxSetCacheConfig = unsafe extern "C" fn(config: c_int) -> CUresult;
type FnCuCtxGetCacheConfig = unsafe extern "C" fn(config: *mut c_int) -> CUresult;
type FnCuCtxSetLimit = unsafe extern "C" fn(limit: c_int, value: usize) -> CUresult;
type FnCuCtxGetLimit = unsafe extern "C" fn(pvalue: *mut usize, limit: c_int) -> CUresult;
type FnCuCtxGetStreamPriorityRange = unsafe extern "C" fn(least: *mut c_int, greatest: *mut c_int) -> CUresult;
type FnCuCtxGetApiVersion = unsafe extern "C" fn(ctx: CUcontext, version: *mut c_uint) -> CUresult;
type FnCuCtxGetFlags = unsafe extern "C" fn(flags: *mut c_uint) -> CUresult;
type FnCuCtxSetFlags = unsafe extern "C" fn(flags: c_uint) -> CUresult;
type FnCuCtxResetPersistingL2Cache = unsafe extern "C" fn() -> CUresult;

// Module management
type FnCuModuleLoadData =
    unsafe extern "C" fn(module: *mut CUmodule, image: *const c_void) -> CUresult;
type FnCuModuleUnload = unsafe extern "C" fn(hmod: CUmodule) -> CUresult;
type FnCuModuleGetFunction = unsafe extern "C" fn(
    hfunc: *mut CUfunction,
    hmod: CUmodule,
    name: *const c_char,
) -> CUresult;
type FnCuModuleGetGlobal = unsafe extern "C" fn(
    dptr: *mut CUdeviceptr,
    bytes: *mut usize,
    hmod: CUmodule,
    name: *const c_char,
) -> CUresult;
type FnCuModuleLoad = unsafe extern "C" fn(module: *mut CUmodule, fname: *const c_char) -> CUresult;
type FnCuModuleLoadDataEx = unsafe extern "C" fn(
    module: *mut CUmodule,
    image: *const c_void,
    num_options: c_uint,
    options: *mut c_int,
    option_values: *mut *mut c_void,
) -> CUresult;
type FnCuModuleLoadFatBinary = unsafe extern "C" fn(module: *mut CUmodule, fat_cubin: *const c_void) -> CUresult;

// Linker
type FnCuLinkCreate = unsafe extern "C" fn(num_options: c_uint, options: *mut c_int, option_values: *mut *mut c_void, state: *mut CUlinkState) -> CUresult;
type FnCuLinkAddData = unsafe extern "C" fn(state: CUlinkState, jit_type: c_int, data: *mut c_void, size: usize, name: *const c_char, num_options: c_uint, options: *mut c_int, option_values: *mut *mut c_void) -> CUresult;
type FnCuLinkAddFile = unsafe extern "C" fn(state: CUlinkState, jit_type: c_int, path: *const c_char, num_options: c_uint, options: *mut c_int, option_values: *mut *mut c_void) -> CUresult;
type FnCuLinkComplete = unsafe extern "C" fn(state: CUlinkState, cubin_out: *mut *mut c_void, size_out: *mut usize) -> CUresult;
type FnCuLinkDestroy = unsafe extern "C" fn(state: CUlinkState) -> CUresult;

// Memory management
type FnCuMemAlloc = unsafe extern "C" fn(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;
type FnCuMemFree = unsafe extern "C" fn(dptr: CUdeviceptr) -> CUresult;
type FnCuMemcpyHtoD = unsafe extern "C" fn(
    dst: CUdeviceptr,
    src: *const c_void,
    byte_count: usize,
) -> CUresult;
type FnCuMemcpyDtoH = unsafe extern "C" fn(
    dst: *mut c_void,
    src: CUdeviceptr,
    byte_count: usize,
) -> CUresult;
type FnCuMemcpyDtoD = unsafe extern "C" fn(
    dst: CUdeviceptr,
    src: CUdeviceptr,
    byte_count: usize,
) -> CUresult;
type FnCuMemcpyHtoDAsync = unsafe extern "C" fn(dst: CUdeviceptr, src: *const c_void, byte_count: usize, hstream: CUstream) -> CUresult;
type FnCuMemcpyDtoHAsync = unsafe extern "C" fn(dst: *mut c_void, src: CUdeviceptr, byte_count: usize, hstream: CUstream) -> CUresult;
type FnCuMemcpyDtoDAsync = unsafe extern "C" fn(dst: CUdeviceptr, src: CUdeviceptr, byte_count: usize, hstream: CUstream) -> CUresult;
type FnCuMemsetD8 =
    unsafe extern "C" fn(dst: CUdeviceptr, value: u8, count: usize) -> CUresult;
type FnCuMemsetD16 =
    unsafe extern "C" fn(dst: CUdeviceptr, value: u16, count: usize) -> CUresult;
type FnCuMemsetD32 =
    unsafe extern "C" fn(dst: CUdeviceptr, value: u32, count: usize) -> CUresult;
type FnCuMemGetInfo = unsafe extern "C" fn(free: *mut usize, total: *mut usize) -> CUresult;
type FnCuMemGetAddressRange = unsafe extern "C" fn(pbase: *mut CUdeviceptr, psize: *mut usize, dptr: CUdeviceptr) -> CUresult;
type FnCuMemAllocHost = unsafe extern "C" fn(pp: *mut *mut c_void, bytesize: usize) -> CUresult;
type FnCuMemFreeHost = unsafe extern "C" fn(p: *mut c_void) -> CUresult;
type FnCuMemHostAlloc = unsafe extern "C" fn(pp: *mut *mut c_void, bytesize: usize, flags: c_uint) -> CUresult;
type FnCuMemHostGetDevicePointer = unsafe extern "C" fn(pdptr: *mut CUdeviceptr, p: *mut c_void, flags: c_uint) -> CUresult;
type FnCuMemHostGetFlags = unsafe extern "C" fn(pflags: *mut c_uint, p: *mut c_void) -> CUresult;
type FnCuMemAllocManaged = unsafe extern "C" fn(dptr: *mut CUdeviceptr, bytesize: usize, flags: c_uint) -> CUresult;
type FnCuMemAllocPitch = unsafe extern "C" fn(dptr: *mut CUdeviceptr, ppitch: *mut usize, width: usize, height: usize, element_size: c_uint) -> CUresult;

// Memory pool
type FnCuMemPoolCreate = unsafe extern "C" fn(pool: *mut CUmemoryPool, props: *const c_void) -> CUresult;
type FnCuMemPoolDestroy = unsafe extern "C" fn(pool: CUmemoryPool) -> CUresult;
type FnCuMemPoolTrimTo = unsafe extern "C" fn(pool: CUmemoryPool, min_bytes_to_keep: usize) -> CUresult;
type FnCuMemPoolSetAttribute = unsafe extern "C" fn(pool: CUmemoryPool, attr: c_int, value: *mut c_void) -> CUresult;
type FnCuMemPoolGetAttribute = unsafe extern "C" fn(pool: CUmemoryPool, attr: c_int, value: *mut c_void) -> CUresult;
type FnCuMemAllocAsync = unsafe extern "C" fn(dptr: *mut CUdeviceptr, bytesize: usize, hstream: CUstream) -> CUresult;
type FnCuMemFreeAsync = unsafe extern "C" fn(dptr: CUdeviceptr, hstream: CUstream) -> CUresult;
type FnCuMemAllocFromPoolAsync = unsafe extern "C" fn(dptr: *mut CUdeviceptr, bytesize: usize, pool: CUmemoryPool, hstream: CUstream) -> CUresult;
type FnCuDeviceGetDefaultMemPool = unsafe extern "C" fn(pool: *mut CUmemoryPool, dev: CUdevice) -> CUresult;
type FnCuDeviceGetMemPool = unsafe extern "C" fn(pool: *mut CUmemoryPool, dev: CUdevice) -> CUresult;
type FnCuDeviceSetMemPool = unsafe extern "C" fn(dev: CUdevice, pool: CUmemoryPool) -> CUresult;

// Execution
type FnCuLaunchKernel = unsafe extern "C" fn(
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
    extra: *mut *mut c_void,
) -> CUresult;
type FnCuFuncGetAttribute = unsafe extern "C" fn(pi: *mut c_int, attrib: c_int, hfunc: CUfunction) -> CUresult;
type FnCuFuncSetAttribute = unsafe extern "C" fn(hfunc: CUfunction, attrib: c_int, value: c_int) -> CUresult;
type FnCuFuncSetCacheConfig = unsafe extern "C" fn(hfunc: CUfunction, config: c_int) -> CUresult;
type FnCuFuncSetSharedMemConfig = unsafe extern "C" fn(hfunc: CUfunction, config: c_int) -> CUresult;
type FnCuFuncGetModule = unsafe extern "C" fn(hmod: *mut CUmodule, hfunc: CUfunction) -> CUresult;
type FnCuFuncGetName = unsafe extern "C" fn(name: *mut *const c_char, hfunc: CUfunction) -> CUresult;
type FnCuOccupancyMaxActiveBlocksPerMultiprocessor = unsafe extern "C" fn(num_blocks: *mut c_int, func: CUfunction, block_size: c_int, dynamic_smem_size: usize) -> CUresult;
type FnCuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags = unsafe extern "C" fn(num_blocks: *mut c_int, func: CUfunction, block_size: c_int, dynamic_smem_size: usize, flags: c_uint) -> CUresult;
type FnCuOccupancyAvailableDynamicSMemPerBlock = unsafe extern "C" fn(dynamic_smem_size: *mut usize, func: CUfunction, num_blocks: c_int, block_size: c_int) -> CUresult;

// Stream management
type FnCuStreamCreate = unsafe extern "C" fn(phstream: *mut CUstream, flags: c_uint) -> CUresult;
type FnCuStreamCreateWithPriority = unsafe extern "C" fn(phstream: *mut CUstream, flags: c_uint, priority: c_int) -> CUresult;
type FnCuStreamDestroy = unsafe extern "C" fn(hstream: CUstream) -> CUresult;
type FnCuStreamSynchronize = unsafe extern "C" fn(hstream: CUstream) -> CUresult;
type FnCuStreamQuery = unsafe extern "C" fn(hstream: CUstream) -> CUresult;
type FnCuStreamWaitEvent = unsafe extern "C" fn(hstream: CUstream, hevent: CUevent, flags: c_uint) -> CUresult;
type FnCuStreamGetPriority = unsafe extern "C" fn(hstream: CUstream, priority: *mut c_int) -> CUresult;
type FnCuStreamGetFlags = unsafe extern "C" fn(hstream: CUstream, flags: *mut c_uint) -> CUresult;
type FnCuStreamGetCtx = unsafe extern "C" fn(hstream: CUstream, pctx: *mut CUcontext) -> CUresult;

// Event management
type FnCuEventCreate = unsafe extern "C" fn(phevent: *mut CUevent, flags: c_uint) -> CUresult;
type FnCuEventDestroy = unsafe extern "C" fn(hevent: CUevent) -> CUresult;
type FnCuEventRecord = unsafe extern "C" fn(hevent: CUevent, hstream: CUstream) -> CUresult;
type FnCuEventRecordWithFlags = unsafe extern "C" fn(hevent: CUevent, hstream: CUstream, flags: c_uint) -> CUresult;
type FnCuEventSynchronize = unsafe extern "C" fn(hevent: CUevent) -> CUresult;
type FnCuEventQuery = unsafe extern "C" fn(hevent: CUevent) -> CUresult;
type FnCuEventElapsedTime =
    unsafe extern "C" fn(ms: *mut f32, start: CUevent, end: CUevent) -> CUresult;

// Pointer queries
type FnCuPointerGetAttribute = unsafe extern "C" fn(data: *mut c_void, attribute: c_int, ptr: CUdeviceptr) -> CUresult;
type FnCuPointerSetAttribute = unsafe extern "C" fn(value: *const c_void, attribute: c_int, ptr: CUdeviceptr) -> CUresult;

// Peer access
type FnCuCtxEnablePeerAccess = unsafe extern "C" fn(peer_ctx: CUcontext, flags: c_uint) -> CUresult;
type FnCuCtxDisablePeerAccess = unsafe extern "C" fn(peer_ctx: CUcontext) -> CUresult;

/// Dynamically loaded CUDA driver library with function pointers.
pub struct CudaDriver {
    _lib: Library,
    // Initialization
    cu_init: FnCuInit,
    cu_driver_get_version: FnCuDriverGetVersion,
    // Device management
    cu_device_get_count: FnCuDeviceGetCount,
    cu_device_get: FnCuDeviceGet,
    cu_device_get_name: FnCuDeviceGetName,
    cu_device_get_attribute: FnCuDeviceGetAttribute,
    cu_device_total_mem: FnCuDeviceTotalMem,
    cu_device_compute_capability: Option<FnCuDeviceComputeCapability>,
    cu_device_get_uuid: Option<FnCuDeviceGetUuid>,
    cu_device_get_pci_bus_id: Option<FnCuDeviceGetPCIBusId>,
    cu_device_get_by_pci_bus_id: Option<FnCuDeviceGetByPCIBusId>,
    cu_device_can_access_peer: Option<FnCuDeviceCanAccessPeer>,
    cu_device_get_p2p_attribute: Option<FnCuDeviceGetP2PAttribute>,
    cu_device_get_default_mem_pool: Option<FnCuDeviceGetDefaultMemPool>,
    cu_device_get_mem_pool: Option<FnCuDeviceGetMemPool>,
    cu_device_set_mem_pool: Option<FnCuDeviceSetMemPool>,
    // Primary context
    cu_device_primary_ctx_retain: Option<FnCuDevicePrimaryCtxRetain>,
    cu_device_primary_ctx_release: Option<FnCuDevicePrimaryCtxRelease>,
    cu_device_primary_ctx_reset: Option<FnCuDevicePrimaryCtxReset>,
    cu_device_primary_ctx_get_state: Option<FnCuDevicePrimaryCtxGetState>,
    cu_device_primary_ctx_set_flags: Option<FnCuDevicePrimaryCtxSetFlags>,
    // Context management
    cu_ctx_create: FnCuCtxCreate,
    cu_ctx_destroy: FnCuCtxDestroy,
    cu_ctx_set_current: FnCuCtxSetCurrent,
    cu_ctx_get_current: FnCuCtxGetCurrent,
    cu_ctx_synchronize: FnCuCtxSynchronize,
    cu_ctx_push_current: Option<FnCuCtxPushCurrent>,
    cu_ctx_pop_current: Option<FnCuCtxPopCurrent>,
    cu_ctx_get_device: Option<FnCuCtxGetDevice>,
    cu_ctx_set_cache_config: Option<FnCuCtxSetCacheConfig>,
    cu_ctx_get_cache_config: Option<FnCuCtxGetCacheConfig>,
    cu_ctx_set_limit: Option<FnCuCtxSetLimit>,
    cu_ctx_get_limit: Option<FnCuCtxGetLimit>,
    cu_ctx_get_stream_priority_range: Option<FnCuCtxGetStreamPriorityRange>,
    cu_ctx_get_api_version: Option<FnCuCtxGetApiVersion>,
    cu_ctx_get_flags: Option<FnCuCtxGetFlags>,
    cu_ctx_set_flags: Option<FnCuCtxSetFlags>,
    cu_ctx_reset_persisting_l2_cache: Option<FnCuCtxResetPersistingL2Cache>,
    cu_ctx_enable_peer_access: Option<FnCuCtxEnablePeerAccess>,
    cu_ctx_disable_peer_access: Option<FnCuCtxDisablePeerAccess>,
    // Module management
    cu_module_load_data: FnCuModuleLoadData,
    cu_module_unload: FnCuModuleUnload,
    cu_module_get_function: FnCuModuleGetFunction,
    cu_module_get_global: FnCuModuleGetGlobal,
    cu_module_load: Option<FnCuModuleLoad>,
    _cu_module_load_data_ex: Option<FnCuModuleLoadDataEx>,
    cu_module_load_fat_binary: Option<FnCuModuleLoadFatBinary>,
    // Linker
    cu_link_create: Option<FnCuLinkCreate>,
    cu_link_add_data: Option<FnCuLinkAddData>,
    cu_link_add_file: Option<FnCuLinkAddFile>,
    cu_link_complete: Option<FnCuLinkComplete>,
    cu_link_destroy: Option<FnCuLinkDestroy>,
    // Memory management
    cu_mem_alloc: FnCuMemAlloc,
    cu_mem_free: FnCuMemFree,
    cu_memcpy_htod: FnCuMemcpyHtoD,
    cu_memcpy_dtoh: FnCuMemcpyDtoH,
    cu_memcpy_dtod: FnCuMemcpyDtoD,
    cu_memcpy_htod_async: Option<FnCuMemcpyHtoDAsync>,
    cu_memcpy_dtoh_async: Option<FnCuMemcpyDtoHAsync>,
    cu_memcpy_dtod_async: Option<FnCuMemcpyDtoDAsync>,
    cu_memset_d8: FnCuMemsetD8,
    cu_memset_d16: Option<FnCuMemsetD16>,
    cu_memset_d32: FnCuMemsetD32,
    cu_mem_get_info: Option<FnCuMemGetInfo>,
    cu_mem_get_address_range: Option<FnCuMemGetAddressRange>,
    cu_mem_alloc_host: Option<FnCuMemAllocHost>,
    cu_mem_free_host: Option<FnCuMemFreeHost>,
    cu_mem_host_alloc: Option<FnCuMemHostAlloc>,
    cu_mem_host_get_device_pointer: Option<FnCuMemHostGetDevicePointer>,
    cu_mem_host_get_flags: Option<FnCuMemHostGetFlags>,
    cu_mem_alloc_managed: Option<FnCuMemAllocManaged>,
    cu_mem_alloc_pitch: Option<FnCuMemAllocPitch>,
    // Memory pools
    _cu_mem_pool_create: Option<FnCuMemPoolCreate>,
    cu_mem_pool_destroy: Option<FnCuMemPoolDestroy>,
    cu_mem_pool_trim_to: Option<FnCuMemPoolTrimTo>,
    _cu_mem_pool_set_attribute: Option<FnCuMemPoolSetAttribute>,
    _cu_mem_pool_get_attribute: Option<FnCuMemPoolGetAttribute>,
    cu_mem_alloc_async: Option<FnCuMemAllocAsync>,
    cu_mem_free_async: Option<FnCuMemFreeAsync>,
    cu_mem_alloc_from_pool_async: Option<FnCuMemAllocFromPoolAsync>,
    // Execution
    cu_launch_kernel: FnCuLaunchKernel,
    cu_func_get_attribute: Option<FnCuFuncGetAttribute>,
    cu_func_set_attribute: Option<FnCuFuncSetAttribute>,
    cu_func_set_cache_config: Option<FnCuFuncSetCacheConfig>,
    cu_func_set_shared_mem_config: Option<FnCuFuncSetSharedMemConfig>,
    cu_func_get_module: Option<FnCuFuncGetModule>,
    cu_func_get_name: Option<FnCuFuncGetName>,
    cu_occupancy_max_active_blocks: Option<FnCuOccupancyMaxActiveBlocksPerMultiprocessor>,
    cu_occupancy_max_active_blocks_with_flags: Option<FnCuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags>,
    cu_occupancy_available_dynamic_smem: Option<FnCuOccupancyAvailableDynamicSMemPerBlock>,
    // Stream management
    cu_stream_create: FnCuStreamCreate,
    cu_stream_create_with_priority: Option<FnCuStreamCreateWithPriority>,
    cu_stream_destroy: FnCuStreamDestroy,
    cu_stream_synchronize: FnCuStreamSynchronize,
    cu_stream_query: FnCuStreamQuery,
    cu_stream_wait_event: Option<FnCuStreamWaitEvent>,
    cu_stream_get_priority: Option<FnCuStreamGetPriority>,
    cu_stream_get_flags: Option<FnCuStreamGetFlags>,
    cu_stream_get_ctx: Option<FnCuStreamGetCtx>,
    // Event management
    cu_event_create: FnCuEventCreate,
    cu_event_destroy: FnCuEventDestroy,
    cu_event_record: FnCuEventRecord,
    cu_event_record_with_flags: Option<FnCuEventRecordWithFlags>,
    cu_event_synchronize: FnCuEventSynchronize,
    cu_event_query: FnCuEventQuery,
    cu_event_elapsed_time: FnCuEventElapsedTime,
    // Pointer queries
    cu_pointer_get_attribute: Option<FnCuPointerGetAttribute>,
    cu_pointer_set_attribute: Option<FnCuPointerSetAttribute>,
}

// SAFETY: The CUDA driver library handles are valid from any thread.
// The CUDA driver API itself handles thread safety via context management.
unsafe impl Send for CudaDriver {}
unsafe impl Sync for CudaDriver {}

impl CudaDriver {
    /// Load the CUDA driver library and resolve all function pointers.
    pub fn load() -> Result<Arc<Self>, String> {
        let lib = Self::load_library()?;

        unsafe {
            let driver = Self {
                cu_init: Self::load_fn(&lib, "cuInit")?,
                cu_driver_get_version: Self::load_fn(&lib, "cuDriverGetVersion")?,
                cu_device_get_count: Self::load_fn(&lib, "cuDeviceGetCount")?,
                cu_device_get: Self::load_fn(&lib, "cuDeviceGet")?,
                cu_device_get_name: Self::load_fn(&lib, "cuDeviceGetName")?,
                cu_device_get_attribute: Self::load_fn(&lib, "cuDeviceGetAttribute")?,
                cu_device_total_mem: Self::load_fn(&lib, "cuDeviceTotalMem_v2")
                    .or_else(|_| Self::load_fn(&lib, "cuDeviceTotalMem"))?,
                cu_device_compute_capability: Self::load_fn_opt(&lib, "cuDeviceComputeCapability"),
                cu_device_get_uuid: Self::load_fn_opt(&lib, "cuDeviceGetUuid"),
                cu_device_get_pci_bus_id: Self::load_fn_opt(&lib, "cuDeviceGetPCIBusId"),
                cu_device_get_by_pci_bus_id: Self::load_fn_opt(&lib, "cuDeviceGetByPCIBusId"),
                cu_device_can_access_peer: Self::load_fn_opt(&lib, "cuDeviceCanAccessPeer"),
                cu_device_get_p2p_attribute: Self::load_fn_opt(&lib, "cuDeviceGetP2PAttribute"),
                cu_device_get_default_mem_pool: Self::load_fn_opt(&lib, "cuDeviceGetDefaultMemPool"),
                cu_device_get_mem_pool: Self::load_fn_opt(&lib, "cuDeviceGetMemPool"),
                cu_device_set_mem_pool: Self::load_fn_opt(&lib, "cuDeviceSetMemPool"),
                // Primary context
                cu_device_primary_ctx_retain: Self::load_fn_opt(&lib, "cuDevicePrimaryCtxRetain"),
                cu_device_primary_ctx_release: Self::load_fn_opt::<FnCuDevicePrimaryCtxRelease>(&lib, "cuDevicePrimaryCtxRelease_v2")
                    .or(Self::load_fn_opt(&lib, "cuDevicePrimaryCtxRelease")),
                cu_device_primary_ctx_reset: Self::load_fn_opt::<FnCuDevicePrimaryCtxReset>(&lib, "cuDevicePrimaryCtxReset_v2")
                    .or(Self::load_fn_opt(&lib, "cuDevicePrimaryCtxReset")),
                cu_device_primary_ctx_get_state: Self::load_fn_opt(&lib, "cuDevicePrimaryCtxGetState"),
                cu_device_primary_ctx_set_flags: Self::load_fn_opt::<FnCuDevicePrimaryCtxSetFlags>(&lib, "cuDevicePrimaryCtxSetFlags_v2")
                    .or(Self::load_fn_opt(&lib, "cuDevicePrimaryCtxSetFlags")),
                // Context
                cu_ctx_create: Self::load_fn(&lib, "cuCtxCreate_v2")
                    .or_else(|_| Self::load_fn(&lib, "cuCtxCreate"))?,
                cu_ctx_destroy: Self::load_fn(&lib, "cuCtxDestroy_v2")
                    .or_else(|_| Self::load_fn(&lib, "cuCtxDestroy"))?,
                cu_ctx_set_current: Self::load_fn(&lib, "cuCtxSetCurrent")?,
                cu_ctx_get_current: Self::load_fn(&lib, "cuCtxGetCurrent")?,
                cu_ctx_synchronize: Self::load_fn(&lib, "cuCtxSynchronize")?,
                cu_ctx_push_current: Self::load_fn_opt::<FnCuCtxPushCurrent>(&lib, "cuCtxPushCurrent_v2")
                    .or(Self::load_fn_opt(&lib, "cuCtxPushCurrent")),
                cu_ctx_pop_current: Self::load_fn_opt::<FnCuCtxPopCurrent>(&lib, "cuCtxPopCurrent_v2")
                    .or(Self::load_fn_opt(&lib, "cuCtxPopCurrent")),
                cu_ctx_get_device: Self::load_fn_opt(&lib, "cuCtxGetDevice"),
                cu_ctx_set_cache_config: Self::load_fn_opt(&lib, "cuCtxSetCacheConfig"),
                cu_ctx_get_cache_config: Self::load_fn_opt(&lib, "cuCtxGetCacheConfig"),
                cu_ctx_set_limit: Self::load_fn_opt(&lib, "cuCtxSetLimit"),
                cu_ctx_get_limit: Self::load_fn_opt(&lib, "cuCtxGetLimit"),
                cu_ctx_get_stream_priority_range: Self::load_fn_opt(&lib, "cuCtxGetStreamPriorityRange"),
                cu_ctx_get_api_version: Self::load_fn_opt(&lib, "cuCtxGetApiVersion"),
                cu_ctx_get_flags: Self::load_fn_opt(&lib, "cuCtxGetFlags"),
                cu_ctx_set_flags: Self::load_fn_opt(&lib, "cuCtxSetFlags"),
                cu_ctx_reset_persisting_l2_cache: Self::load_fn_opt(&lib, "cuCtxResetPersistingL2Cache"),
                cu_ctx_enable_peer_access: Self::load_fn_opt(&lib, "cuCtxEnablePeerAccess"),
                cu_ctx_disable_peer_access: Self::load_fn_opt(&lib, "cuCtxDisablePeerAccess"),
                // Module
                cu_module_load_data: Self::load_fn(&lib, "cuModuleLoadData")?,
                cu_module_unload: Self::load_fn(&lib, "cuModuleUnload")?,
                cu_module_get_function: Self::load_fn(&lib, "cuModuleGetFunction")?,
                cu_module_get_global: Self::load_fn(&lib, "cuModuleGetGlobal_v2")
                    .or_else(|_| Self::load_fn(&lib, "cuModuleGetGlobal"))?,
                cu_module_load: Self::load_fn_opt(&lib, "cuModuleLoad"),
                _cu_module_load_data_ex: Self::load_fn_opt(&lib, "cuModuleLoadDataEx"),
                cu_module_load_fat_binary: Self::load_fn_opt(&lib, "cuModuleLoadFatBinary"),
                // Linker
                cu_link_create: Self::load_fn_opt::<FnCuLinkCreate>(&lib, "cuLinkCreate_v2")
                    .or(Self::load_fn_opt(&lib, "cuLinkCreate")),
                cu_link_add_data: Self::load_fn_opt::<FnCuLinkAddData>(&lib, "cuLinkAddData_v2")
                    .or(Self::load_fn_opt(&lib, "cuLinkAddData")),
                cu_link_add_file: Self::load_fn_opt::<FnCuLinkAddFile>(&lib, "cuLinkAddFile_v2")
                    .or(Self::load_fn_opt(&lib, "cuLinkAddFile")),
                cu_link_complete: Self::load_fn_opt(&lib, "cuLinkComplete"),
                cu_link_destroy: Self::load_fn_opt(&lib, "cuLinkDestroy"),
                // Memory
                cu_mem_alloc: Self::load_fn(&lib, "cuMemAlloc_v2")
                    .or_else(|_| Self::load_fn(&lib, "cuMemAlloc"))?,
                cu_mem_free: Self::load_fn(&lib, "cuMemFree_v2")
                    .or_else(|_| Self::load_fn(&lib, "cuMemFree"))?,
                cu_memcpy_htod: Self::load_fn(&lib, "cuMemcpyHtoD_v2")
                    .or_else(|_| Self::load_fn(&lib, "cuMemcpyHtoD"))?,
                cu_memcpy_dtoh: Self::load_fn(&lib, "cuMemcpyDtoH_v2")
                    .or_else(|_| Self::load_fn(&lib, "cuMemcpyDtoH"))?,
                cu_memcpy_dtod: Self::load_fn(&lib, "cuMemcpyDtoD_v2")
                    .or_else(|_| Self::load_fn(&lib, "cuMemcpyDtoD"))?,
                cu_memcpy_htod_async: Self::load_fn_opt::<FnCuMemcpyHtoDAsync>(&lib, "cuMemcpyHtoDAsync_v2")
                    .or(Self::load_fn_opt(&lib, "cuMemcpyHtoDAsync")),
                cu_memcpy_dtoh_async: Self::load_fn_opt::<FnCuMemcpyDtoHAsync>(&lib, "cuMemcpyDtoHAsync_v2")
                    .or(Self::load_fn_opt(&lib, "cuMemcpyDtoHAsync")),
                cu_memcpy_dtod_async: Self::load_fn_opt::<FnCuMemcpyDtoDAsync>(&lib, "cuMemcpyDtoDAsync_v2")
                    .or(Self::load_fn_opt(&lib, "cuMemcpyDtoDAsync")),
                cu_memset_d8: Self::load_fn(&lib, "cuMemsetD8_v2")
                    .or_else(|_| Self::load_fn(&lib, "cuMemsetD8"))?,
                cu_memset_d16: Self::load_fn_opt::<FnCuMemsetD16>(&lib, "cuMemsetD16_v2")
                    .or(Self::load_fn_opt(&lib, "cuMemsetD16")),
                cu_memset_d32: Self::load_fn(&lib, "cuMemsetD32_v2")
                    .or_else(|_| Self::load_fn(&lib, "cuMemsetD32"))?,
                cu_mem_get_info: Self::load_fn_opt::<FnCuMemGetInfo>(&lib, "cuMemGetInfo_v2")
                    .or(Self::load_fn_opt(&lib, "cuMemGetInfo")),
                cu_mem_get_address_range: Self::load_fn_opt::<FnCuMemGetAddressRange>(&lib, "cuMemGetAddressRange_v2")
                    .or(Self::load_fn_opt(&lib, "cuMemGetAddressRange")),
                cu_mem_alloc_host: Self::load_fn_opt::<FnCuMemAllocHost>(&lib, "cuMemAllocHost_v2")
                    .or(Self::load_fn_opt(&lib, "cuMemAllocHost")),
                cu_mem_free_host: Self::load_fn_opt(&lib, "cuMemFreeHost"),
                cu_mem_host_alloc: Self::load_fn_opt(&lib, "cuMemHostAlloc"),
                cu_mem_host_get_device_pointer: Self::load_fn_opt::<FnCuMemHostGetDevicePointer>(&lib, "cuMemHostGetDevicePointer_v2")
                    .or(Self::load_fn_opt(&lib, "cuMemHostGetDevicePointer")),
                cu_mem_host_get_flags: Self::load_fn_opt(&lib, "cuMemHostGetFlags"),
                cu_mem_alloc_managed: Self::load_fn_opt(&lib, "cuMemAllocManaged"),
                cu_mem_alloc_pitch: Self::load_fn_opt::<FnCuMemAllocPitch>(&lib, "cuMemAllocPitch_v2")
                    .or(Self::load_fn_opt(&lib, "cuMemAllocPitch")),
                // Memory pools
                _cu_mem_pool_create: Self::load_fn_opt(&lib, "cuMemPoolCreate"),
                cu_mem_pool_destroy: Self::load_fn_opt(&lib, "cuMemPoolDestroy"),
                cu_mem_pool_trim_to: Self::load_fn_opt(&lib, "cuMemPoolTrimTo"),
                _cu_mem_pool_set_attribute: Self::load_fn_opt(&lib, "cuMemPoolSetAttribute"),
                _cu_mem_pool_get_attribute: Self::load_fn_opt(&lib, "cuMemPoolGetAttribute"),
                cu_mem_alloc_async: Self::load_fn_opt::<FnCuMemAllocAsync>(&lib, "cuMemAllocAsync_ptsz")
                    .or(Self::load_fn_opt(&lib, "cuMemAllocAsync")),
                cu_mem_free_async: Self::load_fn_opt::<FnCuMemFreeAsync>(&lib, "cuMemFreeAsync_ptsz")
                    .or(Self::load_fn_opt(&lib, "cuMemFreeAsync")),
                cu_mem_alloc_from_pool_async: Self::load_fn_opt::<FnCuMemAllocFromPoolAsync>(&lib, "cuMemAllocFromPoolAsync_ptsz")
                    .or(Self::load_fn_opt(&lib, "cuMemAllocFromPoolAsync")),
                // Execution
                cu_launch_kernel: Self::load_fn(&lib, "cuLaunchKernel")?,
                cu_func_get_attribute: Self::load_fn_opt(&lib, "cuFuncGetAttribute"),
                cu_func_set_attribute: Self::load_fn_opt(&lib, "cuFuncSetAttribute"),
                cu_func_set_cache_config: Self::load_fn_opt(&lib, "cuFuncSetCacheConfig"),
                cu_func_set_shared_mem_config: Self::load_fn_opt(&lib, "cuFuncSetSharedMemConfig"),
                cu_func_get_module: Self::load_fn_opt(&lib, "cuFuncGetModule"),
                cu_func_get_name: Self::load_fn_opt(&lib, "cuFuncGetName"),
                cu_occupancy_max_active_blocks: Self::load_fn_opt(&lib, "cuOccupancyMaxActiveBlocksPerMultiprocessor"),
                cu_occupancy_max_active_blocks_with_flags: Self::load_fn_opt(&lib, "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags"),
                cu_occupancy_available_dynamic_smem: Self::load_fn_opt(&lib, "cuOccupancyAvailableDynamicSMemPerBlock"),
                // Stream
                cu_stream_create: Self::load_fn(&lib, "cuStreamCreate")?,
                cu_stream_create_with_priority: Self::load_fn_opt(&lib, "cuStreamCreateWithPriority"),
                cu_stream_destroy: Self::load_fn(&lib, "cuStreamDestroy_v2")
                    .or_else(|_| Self::load_fn(&lib, "cuStreamDestroy"))?,
                cu_stream_synchronize: Self::load_fn(&lib, "cuStreamSynchronize")?,
                cu_stream_query: Self::load_fn(&lib, "cuStreamQuery")?,
                cu_stream_wait_event: Self::load_fn_opt(&lib, "cuStreamWaitEvent"),
                cu_stream_get_priority: Self::load_fn_opt(&lib, "cuStreamGetPriority"),
                cu_stream_get_flags: Self::load_fn_opt(&lib, "cuStreamGetFlags"),
                cu_stream_get_ctx: Self::load_fn_opt::<FnCuStreamGetCtx>(&lib, "cuStreamGetCtx_v2")
                    .or(Self::load_fn_opt(&lib, "cuStreamGetCtx")),
                // Event
                cu_event_create: Self::load_fn(&lib, "cuEventCreate")?,
                cu_event_destroy: Self::load_fn(&lib, "cuEventDestroy_v2")
                    .or_else(|_| Self::load_fn(&lib, "cuEventDestroy"))?,
                cu_event_record: Self::load_fn(&lib, "cuEventRecord")?,
                cu_event_record_with_flags: Self::load_fn_opt(&lib, "cuEventRecordWithFlags"),
                cu_event_synchronize: Self::load_fn(&lib, "cuEventSynchronize")?,
                cu_event_query: Self::load_fn(&lib, "cuEventQuery")?,
                cu_event_elapsed_time: Self::load_fn(&lib, "cuEventElapsedTime")?,
                // Pointer queries
                cu_pointer_get_attribute: Self::load_fn_opt(&lib, "cuPointerGetAttribute"),
                cu_pointer_set_attribute: Self::load_fn_opt(&lib, "cuPointerSetAttribute"),
                _lib: lib,
            };

            info!("CUDA driver loaded successfully");
            Ok(Arc::new(driver))
        }
    }

    fn load_library() -> Result<Library, String> {
        #[cfg(target_os = "windows")]
        let lib_names = &["nvcuda.dll"];

        #[cfg(target_os = "linux")]
        let lib_names = &["libcuda.so.1", "libcuda.so"];

        #[cfg(target_os = "macos")]
        let lib_names = &["libcuda.dylib"];

        let mut last_err = String::new();
        for name in lib_names {
            match unsafe { Library::new(name) } {
                Ok(lib) => {
                    info!("loaded CUDA driver from: {}", name);
                    return Ok(lib);
                }
                Err(e) => {
                    last_err = format!("{}: {}", name, e);
                    debug!("failed to load {}: {}", name, e);
                }
            }
        }

        Err(format!("failed to load CUDA driver library: {}", last_err))
    }

    unsafe fn load_fn<F: Copy>(lib: &Library, name: &str) -> Result<F, String> {
        let sym: Symbol<F> = lib
            .get(name.as_bytes())
            .map_err(|e| format!("failed to load {}: {}", name, e))?;
        Ok(*sym)
    }

    unsafe fn load_fn_opt<F: Copy>(lib: &Library, name: &str) -> Option<F> {
        lib.get(name.as_bytes()).ok().map(|s: Symbol<F>| *s)
    }

    // ── Initialization ────────────────────────────────────────────

    pub fn init(&self, flags: u32) -> CUresult {
        unsafe { (self.cu_init)(flags as c_uint) }
    }

    pub fn driver_get_version(&self) -> Result<i32, CUresult> {
        let mut version: c_int = 0;
        let res = unsafe { (self.cu_driver_get_version)(&mut version) };
        if res == CUDA_SUCCESS { Ok(version) } else { Err(res) }
    }

    // ── Device Management ─────────────────────────────────────────

    pub fn device_get_count(&self) -> Result<i32, CUresult> {
        let mut count: c_int = 0;
        let res = unsafe { (self.cu_device_get_count)(&mut count) };
        if res == CUDA_SUCCESS { Ok(count) } else { Err(res) }
    }

    pub fn device_get(&self, ordinal: i32) -> Result<CUdevice, CUresult> {
        let mut device: CUdevice = 0;
        let res = unsafe { (self.cu_device_get)(&mut device, ordinal) };
        if res == CUDA_SUCCESS { Ok(device) } else { Err(res) }
    }

    pub fn device_get_name(&self, device: CUdevice) -> Result<String, CUresult> {
        let mut buf = [0u8; 256];
        let res = unsafe {
            (self.cu_device_get_name)(buf.as_mut_ptr() as *mut c_char, 256, device)
        };
        if res == CUDA_SUCCESS {
            let name = unsafe { CStr::from_ptr(buf.as_ptr() as *const c_char) };
            Ok(name.to_string_lossy().into_owned())
        } else {
            Err(res)
        }
    }

    pub fn device_get_attribute(&self, attrib: i32, device: CUdevice) -> Result<i32, CUresult> {
        let mut value: c_int = 0;
        let res = unsafe { (self.cu_device_get_attribute)(&mut value, attrib, device) };
        if res == CUDA_SUCCESS { Ok(value) } else { Err(res) }
    }

    pub fn device_total_mem(&self, device: CUdevice) -> Result<usize, CUresult> {
        let mut bytes: usize = 0;
        let res = unsafe { (self.cu_device_total_mem)(&mut bytes, device) };
        if res == CUDA_SUCCESS { Ok(bytes) } else { Err(res) }
    }

    pub fn device_compute_capability(&self, device: CUdevice) -> Result<(i32, i32), CUresult> {
        if let Some(func) = self.cu_device_compute_capability {
            let mut major: c_int = 0;
            let mut minor: c_int = 0;
            let res = unsafe { func(&mut major, &mut minor, device) };
            if res == CUDA_SUCCESS { Ok((major, minor)) } else { Err(res) }
        } else {
            let major = self.device_get_attribute(75, device)?;
            let minor = self.device_get_attribute(76, device)?;
            Ok((major, minor))
        }
    }

    pub fn device_get_uuid(&self, device: CUdevice) -> Result<[u8; 16], CUresult> {
        if let Some(func) = self.cu_device_get_uuid {
            let mut uuid = CUuuid { bytes: [0u8; 16] };
            let res = unsafe { func(&mut uuid, device) };
            if res == CUDA_SUCCESS { Ok(uuid.bytes) } else { Err(res) }
        } else {
            Err(CUDA_ERROR_NOT_SUPPORTED)
        }
    }

    pub fn device_get_pci_bus_id(&self, device: CUdevice) -> Result<String, CUresult> {
        if let Some(func) = self.cu_device_get_pci_bus_id {
            let mut buf = [0u8; 64];
            let res = unsafe { func(buf.as_mut_ptr() as *mut c_char, 64, device) };
            if res == CUDA_SUCCESS {
                let name = unsafe { CStr::from_ptr(buf.as_ptr() as *const c_char) };
                Ok(name.to_string_lossy().into_owned())
            } else {
                Err(res)
            }
        } else {
            Err(CUDA_ERROR_NOT_SUPPORTED)
        }
    }

    pub fn device_get_by_pci_bus_id(&self, pci_bus_id: &str) -> Result<CUdevice, CUresult> {
        if let Some(func) = self.cu_device_get_by_pci_bus_id {
            let c_str = std::ffi::CString::new(pci_bus_id).map_err(|_| 1)?;
            let mut dev: CUdevice = 0;
            let res = unsafe { func(&mut dev, c_str.as_ptr()) };
            if res == CUDA_SUCCESS { Ok(dev) } else { Err(res) }
        } else {
            Err(CUDA_ERROR_NOT_SUPPORTED)
        }
    }

    pub fn device_can_access_peer(&self, dev: CUdevice, peer_dev: CUdevice) -> Result<bool, CUresult> {
        if let Some(func) = self.cu_device_can_access_peer {
            let mut can: c_int = 0;
            let res = unsafe { func(&mut can, dev, peer_dev) };
            if res == CUDA_SUCCESS { Ok(can != 0) } else { Err(res) }
        } else {
            Ok(false)
        }
    }

    pub fn device_get_p2p_attribute(&self, attrib: i32, src: CUdevice, dst: CUdevice) -> Result<i32, CUresult> {
        if let Some(func) = self.cu_device_get_p2p_attribute {
            let mut value: c_int = 0;
            let res = unsafe { func(&mut value, attrib, src, dst) };
            if res == CUDA_SUCCESS { Ok(value) } else { Err(res) }
        } else {
            Err(CUDA_ERROR_NOT_SUPPORTED)
        }
    }

    pub fn device_get_default_mem_pool(&self, device: CUdevice) -> Result<CUmemoryPool, CUresult> {
        if let Some(func) = self.cu_device_get_default_mem_pool {
            let mut pool: CUmemoryPool = std::ptr::null_mut();
            let res = unsafe { func(&mut pool, device) };
            if res == CUDA_SUCCESS { Ok(pool) } else { Err(res) }
        } else {
            Err(CUDA_ERROR_NOT_SUPPORTED)
        }
    }

    pub fn device_get_mem_pool(&self, device: CUdevice) -> Result<CUmemoryPool, CUresult> {
        if let Some(func) = self.cu_device_get_mem_pool {
            let mut pool: CUmemoryPool = std::ptr::null_mut();
            let res = unsafe { func(&mut pool, device) };
            if res == CUDA_SUCCESS { Ok(pool) } else { Err(res) }
        } else {
            Err(CUDA_ERROR_NOT_SUPPORTED)
        }
    }

    pub fn device_set_mem_pool(&self, device: CUdevice, pool: CUmemoryPool) -> CUresult {
        if let Some(func) = self.cu_device_set_mem_pool {
            unsafe { func(device, pool) }
        } else {
            CUDA_ERROR_NOT_SUPPORTED
        }
    }

    // ── Primary Context ───────────────────────────────────────────

    pub fn device_primary_ctx_retain(&self, device: CUdevice) -> Result<CUcontext, CUresult> {
        if let Some(func) = self.cu_device_primary_ctx_retain {
            let mut ctx: CUcontext = std::ptr::null_mut();
            let res = unsafe { func(&mut ctx, device) };
            if res == CUDA_SUCCESS { Ok(ctx) } else { Err(res) }
        } else {
            Err(CUDA_ERROR_NOT_SUPPORTED)
        }
    }

    pub fn device_primary_ctx_release(&self, device: CUdevice) -> CUresult {
        if let Some(func) = self.cu_device_primary_ctx_release {
            unsafe { func(device) }
        } else {
            CUDA_ERROR_NOT_SUPPORTED
        }
    }

    pub fn device_primary_ctx_reset(&self, device: CUdevice) -> CUresult {
        if let Some(func) = self.cu_device_primary_ctx_reset {
            unsafe { func(device) }
        } else {
            CUDA_ERROR_NOT_SUPPORTED
        }
    }

    pub fn device_primary_ctx_get_state(&self, device: CUdevice) -> Result<(u32, bool), CUresult> {
        if let Some(func) = self.cu_device_primary_ctx_get_state {
            let mut flags: c_uint = 0;
            let mut active: c_int = 0;
            let res = unsafe { func(device, &mut flags, &mut active) };
            if res == CUDA_SUCCESS { Ok((flags, active != 0)) } else { Err(res) }
        } else {
            Err(CUDA_ERROR_NOT_SUPPORTED)
        }
    }

    pub fn device_primary_ctx_set_flags(&self, device: CUdevice, flags: u32) -> CUresult {
        if let Some(func) = self.cu_device_primary_ctx_set_flags {
            unsafe { func(device, flags as c_uint) }
        } else {
            CUDA_ERROR_NOT_SUPPORTED
        }
    }

    // ── Context Management ────────────────────────────────────────

    pub fn ctx_create(&self, flags: u32, device: CUdevice) -> Result<CUcontext, CUresult> {
        let mut ctx: CUcontext = std::ptr::null_mut();
        let res = unsafe { (self.cu_ctx_create)(&mut ctx, flags as c_uint, device) };
        if res == CUDA_SUCCESS { Ok(ctx) } else { Err(res) }
    }

    pub fn ctx_destroy(&self, ctx: CUcontext) -> CUresult {
        unsafe { (self.cu_ctx_destroy)(ctx) }
    }

    pub fn ctx_set_current(&self, ctx: CUcontext) -> CUresult {
        unsafe { (self.cu_ctx_set_current)(ctx) }
    }

    pub fn ctx_get_current(&self) -> Result<CUcontext, CUresult> {
        let mut ctx: CUcontext = std::ptr::null_mut();
        let res = unsafe { (self.cu_ctx_get_current)(&mut ctx) };
        if res == CUDA_SUCCESS { Ok(ctx) } else { Err(res) }
    }

    pub fn ctx_synchronize(&self) -> CUresult {
        unsafe { (self.cu_ctx_synchronize)() }
    }

    pub fn ctx_push_current(&self, ctx: CUcontext) -> CUresult {
        if let Some(func) = self.cu_ctx_push_current {
            unsafe { func(ctx) }
        } else {
            CUDA_ERROR_NOT_SUPPORTED
        }
    }

    pub fn ctx_pop_current(&self) -> Result<CUcontext, CUresult> {
        if let Some(func) = self.cu_ctx_pop_current {
            let mut ctx: CUcontext = std::ptr::null_mut();
            let res = unsafe { func(&mut ctx) };
            if res == CUDA_SUCCESS { Ok(ctx) } else { Err(res) }
        } else {
            Err(CUDA_ERROR_NOT_SUPPORTED)
        }
    }

    pub fn ctx_get_device(&self) -> Result<CUdevice, CUresult> {
        if let Some(func) = self.cu_ctx_get_device {
            let mut dev: CUdevice = 0;
            let res = unsafe { func(&mut dev) };
            if res == CUDA_SUCCESS { Ok(dev) } else { Err(res) }
        } else {
            Err(CUDA_ERROR_NOT_SUPPORTED)
        }
    }

    pub fn ctx_set_cache_config(&self, config: i32) -> CUresult {
        if let Some(func) = self.cu_ctx_set_cache_config {
            unsafe { func(config) }
        } else {
            CUDA_ERROR_NOT_SUPPORTED
        }
    }

    pub fn ctx_get_cache_config(&self) -> Result<i32, CUresult> {
        if let Some(func) = self.cu_ctx_get_cache_config {
            let mut config: c_int = 0;
            let res = unsafe { func(&mut config) };
            if res == CUDA_SUCCESS { Ok(config) } else { Err(res) }
        } else {
            Err(CUDA_ERROR_NOT_SUPPORTED)
        }
    }

    pub fn ctx_set_limit(&self, limit: i32, value: u64) -> CUresult {
        if let Some(func) = self.cu_ctx_set_limit {
            unsafe { func(limit, value as usize) }
        } else {
            CUDA_ERROR_NOT_SUPPORTED
        }
    }

    pub fn ctx_get_limit(&self, limit: i32) -> Result<u64, CUresult> {
        if let Some(func) = self.cu_ctx_get_limit {
            let mut value: usize = 0;
            let res = unsafe { func(&mut value, limit) };
            if res == CUDA_SUCCESS { Ok(value as u64) } else { Err(res) }
        } else {
            Err(CUDA_ERROR_NOT_SUPPORTED)
        }
    }

    pub fn ctx_get_stream_priority_range(&self) -> Result<(i32, i32), CUresult> {
        if let Some(func) = self.cu_ctx_get_stream_priority_range {
            let mut least: c_int = 0;
            let mut greatest: c_int = 0;
            let res = unsafe { func(&mut least, &mut greatest) };
            if res == CUDA_SUCCESS { Ok((least, greatest)) } else { Err(res) }
        } else {
            Ok((0, 0))
        }
    }

    pub fn ctx_get_api_version(&self, ctx: CUcontext) -> Result<u32, CUresult> {
        if let Some(func) = self.cu_ctx_get_api_version {
            let mut version: c_uint = 0;
            let res = unsafe { func(ctx, &mut version) };
            if res == CUDA_SUCCESS { Ok(version) } else { Err(res) }
        } else {
            Ok(12000) // fallback
        }
    }

    pub fn ctx_get_flags(&self) -> Result<u32, CUresult> {
        if let Some(func) = self.cu_ctx_get_flags {
            let mut flags: c_uint = 0;
            let res = unsafe { func(&mut flags) };
            if res == CUDA_SUCCESS { Ok(flags) } else { Err(res) }
        } else {
            Ok(0)
        }
    }

    pub fn ctx_set_flags(&self, flags: u32) -> CUresult {
        if let Some(func) = self.cu_ctx_set_flags {
            unsafe { func(flags as c_uint) }
        } else {
            CUDA_ERROR_NOT_SUPPORTED
        }
    }

    pub fn ctx_reset_persisting_l2_cache(&self) -> CUresult {
        if let Some(func) = self.cu_ctx_reset_persisting_l2_cache {
            unsafe { func() }
        } else {
            CUDA_SUCCESS // no-op if not supported
        }
    }

    pub fn ctx_enable_peer_access(&self, peer_ctx: CUcontext, flags: u32) -> CUresult {
        if let Some(func) = self.cu_ctx_enable_peer_access {
            unsafe { func(peer_ctx, flags as c_uint) }
        } else {
            CUDA_ERROR_NOT_SUPPORTED
        }
    }

    pub fn ctx_disable_peer_access(&self, peer_ctx: CUcontext) -> CUresult {
        if let Some(func) = self.cu_ctx_disable_peer_access {
            unsafe { func(peer_ctx) }
        } else {
            CUDA_ERROR_NOT_SUPPORTED
        }
    }

    // ── Module Management ─────────────────────────────────────────

    pub fn module_load_data(&self, image: &[u8]) -> Result<CUmodule, CUresult> {
        let mut module: CUmodule = std::ptr::null_mut();
        let res = unsafe {
            (self.cu_module_load_data)(&mut module, image.as_ptr() as *const c_void)
        };
        if res == CUDA_SUCCESS { Ok(module) } else { Err(res) }
    }

    pub fn module_unload(&self, module: CUmodule) -> CUresult {
        unsafe { (self.cu_module_unload)(module) }
    }

    pub fn module_get_function(&self, module: CUmodule, name: &str) -> Result<CUfunction, CUresult> {
        let c_name = std::ffi::CString::new(name).map_err(|_| 1)?;
        let mut func: CUfunction = std::ptr::null_mut();
        let res = unsafe { (self.cu_module_get_function)(&mut func, module, c_name.as_ptr()) };
        if res == CUDA_SUCCESS { Ok(func) } else { Err(res) }
    }

    pub fn module_get_global(&self, module: CUmodule, name: &str) -> Result<(CUdeviceptr, usize), CUresult> {
        let c_name = std::ffi::CString::new(name).map_err(|_| 1)?;
        let mut dptr: CUdeviceptr = 0;
        let mut size: usize = 0;
        let res = unsafe { (self.cu_module_get_global)(&mut dptr, &mut size, module, c_name.as_ptr()) };
        if res == CUDA_SUCCESS { Ok((dptr, size)) } else { Err(res) }
    }

    pub fn module_load(&self, fname: &str) -> Result<CUmodule, CUresult> {
        if let Some(func) = self.cu_module_load {
            let c_name = std::ffi::CString::new(fname).map_err(|_| 1)?;
            let mut module: CUmodule = std::ptr::null_mut();
            let res = unsafe { func(&mut module, c_name.as_ptr()) };
            if res == CUDA_SUCCESS { Ok(module) } else { Err(res) }
        } else {
            Err(CUDA_ERROR_NOT_SUPPORTED)
        }
    }

    pub fn module_load_data_ex(&self, image: &[u8]) -> Result<CUmodule, CUresult> {
        // Simplified: ignore JIT options, forward to ModuleLoadData
        self.module_load_data(image)
    }

    pub fn module_load_fat_binary(&self, fat_cubin: &[u8]) -> Result<CUmodule, CUresult> {
        if let Some(func) = self.cu_module_load_fat_binary {
            let mut module: CUmodule = std::ptr::null_mut();
            let res = unsafe { func(&mut module, fat_cubin.as_ptr() as *const c_void) };
            if res == CUDA_SUCCESS { Ok(module) } else { Err(res) }
        } else {
            // Fallback to ModuleLoadData
            self.module_load_data(fat_cubin)
        }
    }

    // ── Linker ────────────────────────────────────────────────────

    pub fn link_create(&self) -> Result<CUlinkState, CUresult> {
        if let Some(func) = self.cu_link_create {
            let mut state: CUlinkState = std::ptr::null_mut();
            let res = unsafe { func(0, std::ptr::null_mut(), std::ptr::null_mut(), &mut state) };
            if res == CUDA_SUCCESS { Ok(state) } else { Err(res) }
        } else {
            Err(CUDA_ERROR_NOT_SUPPORTED)
        }
    }

    pub fn link_add_data(&self, state: CUlinkState, jit_type: i32, data: &[u8], name: &str) -> CUresult {
        if let Some(func) = self.cu_link_add_data {
            let c_name = match std::ffi::CString::new(name) { Ok(s) => s, Err(_) => return 1 };
            unsafe { func(state, jit_type, data.as_ptr() as *mut c_void, data.len(), c_name.as_ptr(), 0, std::ptr::null_mut(), std::ptr::null_mut()) }
        } else {
            CUDA_ERROR_NOT_SUPPORTED
        }
    }

    pub fn link_add_file(&self, state: CUlinkState, jit_type: i32, path: &str) -> CUresult {
        if let Some(func) = self.cu_link_add_file {
            let c_path = match std::ffi::CString::new(path) { Ok(s) => s, Err(_) => return 1 };
            unsafe { func(state, jit_type, c_path.as_ptr(), 0, std::ptr::null_mut(), std::ptr::null_mut()) }
        } else {
            CUDA_ERROR_NOT_SUPPORTED
        }
    }

    pub fn link_complete(&self, state: CUlinkState) -> Result<Vec<u8>, CUresult> {
        if let Some(func) = self.cu_link_complete {
            let mut cubin_ptr: *mut c_void = std::ptr::null_mut();
            let mut size: usize = 0;
            let res = unsafe { func(state, &mut cubin_ptr, &mut size) };
            if res == CUDA_SUCCESS {
                let data = unsafe { std::slice::from_raw_parts(cubin_ptr as *const u8, size) }.to_vec();
                Ok(data)
            } else {
                Err(res)
            }
        } else {
            Err(CUDA_ERROR_NOT_SUPPORTED)
        }
    }

    pub fn link_destroy(&self, state: CUlinkState) -> CUresult {
        if let Some(func) = self.cu_link_destroy {
            unsafe { func(state) }
        } else {
            CUDA_ERROR_NOT_SUPPORTED
        }
    }

    // ── Memory Management ─────────────────────────────────────────

    pub fn mem_alloc(&self, byte_size: usize) -> Result<CUdeviceptr, CUresult> {
        let mut dptr: CUdeviceptr = 0;
        let res = unsafe { (self.cu_mem_alloc)(&mut dptr, byte_size) };
        if res == CUDA_SUCCESS { Ok(dptr) } else { Err(res) }
    }

    pub fn mem_free(&self, dptr: CUdeviceptr) -> CUresult {
        unsafe { (self.cu_mem_free)(dptr) }
    }

    pub fn memcpy_htod(&self, dst: CUdeviceptr, src: &[u8]) -> CUresult {
        unsafe { (self.cu_memcpy_htod)(dst, src.as_ptr() as *const c_void, src.len()) }
    }

    pub fn memcpy_dtoh(&self, dst: &mut [u8], src: CUdeviceptr) -> CUresult {
        unsafe { (self.cu_memcpy_dtoh)(dst.as_mut_ptr() as *mut c_void, src, dst.len()) }
    }

    pub fn memcpy_dtod(&self, dst: CUdeviceptr, src: CUdeviceptr, byte_count: usize) -> CUresult {
        unsafe { (self.cu_memcpy_dtod)(dst, src, byte_count) }
    }

    pub fn memcpy_htod_async(&self, dst: CUdeviceptr, src: &[u8], stream: CUstream) -> CUresult {
        if let Some(func) = self.cu_memcpy_htod_async {
            unsafe { func(dst, src.as_ptr() as *const c_void, src.len(), stream) }
        } else {
            self.memcpy_htod(dst, src)
        }
    }

    pub fn memcpy_dtoh_async(&self, dst: &mut [u8], src: CUdeviceptr, stream: CUstream) -> CUresult {
        if let Some(func) = self.cu_memcpy_dtoh_async {
            unsafe { func(dst.as_mut_ptr() as *mut c_void, src, dst.len(), stream) }
        } else {
            self.memcpy_dtoh(dst, src)
        }
    }

    pub fn memcpy_dtod_async(&self, dst: CUdeviceptr, src: CUdeviceptr, byte_count: usize, stream: CUstream) -> CUresult {
        if let Some(func) = self.cu_memcpy_dtod_async {
            unsafe { func(dst, src, byte_count, stream) }
        } else {
            self.memcpy_dtod(dst, src, byte_count)
        }
    }

    pub fn memset_d8(&self, dst: CUdeviceptr, value: u8, count: usize) -> CUresult {
        unsafe { (self.cu_memset_d8)(dst, value, count) }
    }

    pub fn memset_d16(&self, dst: CUdeviceptr, value: u16, count: usize) -> CUresult {
        if let Some(func) = self.cu_memset_d16 {
            unsafe { func(dst, value, count) }
        } else {
            CUDA_ERROR_NOT_SUPPORTED
        }
    }

    pub fn memset_d32(&self, dst: CUdeviceptr, value: u32, count: usize) -> CUresult {
        unsafe { (self.cu_memset_d32)(dst, value, count) }
    }

    pub fn mem_get_info(&self) -> Result<(usize, usize), CUresult> {
        if let Some(func) = self.cu_mem_get_info {
            let mut free: usize = 0;
            let mut total: usize = 0;
            let res = unsafe { func(&mut free, &mut total) };
            if res == CUDA_SUCCESS { Ok((free, total)) } else { Err(res) }
        } else {
            Err(CUDA_ERROR_NOT_SUPPORTED)
        }
    }

    pub fn mem_get_address_range(&self, dptr: CUdeviceptr) -> Result<(CUdeviceptr, usize), CUresult> {
        if let Some(func) = self.cu_mem_get_address_range {
            let mut base: CUdeviceptr = 0;
            let mut size: usize = 0;
            let res = unsafe { func(&mut base, &mut size, dptr) };
            if res == CUDA_SUCCESS { Ok((base, size)) } else { Err(res) }
        } else {
            Err(CUDA_ERROR_NOT_SUPPORTED)
        }
    }

    pub fn mem_alloc_host(&self, byte_size: usize) -> Result<*mut c_void, CUresult> {
        if let Some(func) = self.cu_mem_alloc_host {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            let res = unsafe { func(&mut ptr, byte_size) };
            if res == CUDA_SUCCESS { Ok(ptr) } else { Err(res) }
        } else {
            Err(CUDA_ERROR_NOT_SUPPORTED)
        }
    }

    pub fn mem_free_host(&self, ptr: *mut c_void) -> CUresult {
        if let Some(func) = self.cu_mem_free_host {
            unsafe { func(ptr) }
        } else {
            CUDA_ERROR_NOT_SUPPORTED
        }
    }

    pub fn mem_host_alloc(&self, byte_size: usize, flags: u32) -> Result<*mut c_void, CUresult> {
        if let Some(func) = self.cu_mem_host_alloc {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            let res = unsafe { func(&mut ptr, byte_size, flags as c_uint) };
            if res == CUDA_SUCCESS { Ok(ptr) } else { Err(res) }
        } else {
            self.mem_alloc_host(byte_size)
        }
    }

    pub fn mem_host_get_device_pointer(&self, host_ptr: *mut c_void, flags: u32) -> Result<CUdeviceptr, CUresult> {
        if let Some(func) = self.cu_mem_host_get_device_pointer {
            let mut dptr: CUdeviceptr = 0;
            let res = unsafe { func(&mut dptr, host_ptr, flags as c_uint) };
            if res == CUDA_SUCCESS { Ok(dptr) } else { Err(res) }
        } else {
            Err(CUDA_ERROR_NOT_SUPPORTED)
        }
    }

    pub fn mem_host_get_flags(&self, host_ptr: *mut c_void) -> Result<u32, CUresult> {
        if let Some(func) = self.cu_mem_host_get_flags {
            let mut flags: c_uint = 0;
            let res = unsafe { func(&mut flags, host_ptr) };
            if res == CUDA_SUCCESS { Ok(flags) } else { Err(res) }
        } else {
            Err(CUDA_ERROR_NOT_SUPPORTED)
        }
    }

    pub fn mem_alloc_managed(&self, byte_size: usize, flags: u32) -> Result<CUdeviceptr, CUresult> {
        if let Some(func) = self.cu_mem_alloc_managed {
            let mut dptr: CUdeviceptr = 0;
            let res = unsafe { func(&mut dptr, byte_size, flags as c_uint) };
            if res == CUDA_SUCCESS { Ok(dptr) } else { Err(res) }
        } else {
            // Fallback to regular device alloc
            self.mem_alloc(byte_size)
        }
    }

    pub fn mem_alloc_pitch(&self, width: usize, height: usize, element_size: u32) -> Result<(CUdeviceptr, usize), CUresult> {
        if let Some(func) = self.cu_mem_alloc_pitch {
            let mut dptr: CUdeviceptr = 0;
            let mut pitch: usize = 0;
            let res = unsafe { func(&mut dptr, &mut pitch, width, height, element_size as c_uint) };
            if res == CUDA_SUCCESS { Ok((dptr, pitch)) } else { Err(res) }
        } else {
            Err(CUDA_ERROR_NOT_SUPPORTED)
        }
    }

    // ── Memory Pools ──────────────────────────────────────────────

    pub fn mem_alloc_async(&self, byte_size: usize, stream: CUstream) -> Result<CUdeviceptr, CUresult> {
        if let Some(func) = self.cu_mem_alloc_async {
            let mut dptr: CUdeviceptr = 0;
            let res = unsafe { func(&mut dptr, byte_size, stream) };
            if res == CUDA_SUCCESS { Ok(dptr) } else { Err(res) }
        } else {
            self.mem_alloc(byte_size)
        }
    }

    pub fn mem_free_async(&self, dptr: CUdeviceptr, stream: CUstream) -> CUresult {
        if let Some(func) = self.cu_mem_free_async {
            unsafe { func(dptr, stream) }
        } else {
            self.mem_free(dptr)
        }
    }

    pub fn mem_alloc_from_pool_async(&self, byte_size: usize, pool: CUmemoryPool, stream: CUstream) -> Result<CUdeviceptr, CUresult> {
        if let Some(func) = self.cu_mem_alloc_from_pool_async {
            let mut dptr: CUdeviceptr = 0;
            let res = unsafe { func(&mut dptr, byte_size, pool, stream) };
            if res == CUDA_SUCCESS { Ok(dptr) } else { Err(res) }
        } else {
            self.mem_alloc(byte_size)
        }
    }

    pub fn mem_pool_destroy(&self, pool: CUmemoryPool) -> CUresult {
        if let Some(func) = self.cu_mem_pool_destroy {
            unsafe { func(pool) }
        } else {
            CUDA_ERROR_NOT_SUPPORTED
        }
    }

    pub fn mem_pool_trim_to(&self, pool: CUmemoryPool, min_bytes: usize) -> CUresult {
        if let Some(func) = self.cu_mem_pool_trim_to {
            unsafe { func(pool, min_bytes) }
        } else {
            CUDA_ERROR_NOT_SUPPORTED
        }
    }

    // ── Execution ─────────────────────────────────────────────────

    pub unsafe fn launch_kernel(
        &self,
        func: CUfunction,
        grid_dim: [u32; 3],
        block_dim: [u32; 3],
        shared_mem_bytes: u32,
        stream: CUstream,
        kernel_params: &mut [*mut c_void],
    ) -> CUresult {
        (self.cu_launch_kernel)(
            func,
            grid_dim[0] as c_uint, grid_dim[1] as c_uint, grid_dim[2] as c_uint,
            block_dim[0] as c_uint, block_dim[1] as c_uint, block_dim[2] as c_uint,
            shared_mem_bytes as c_uint,
            stream,
            kernel_params.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    }

    pub fn func_get_attribute(&self, attrib: i32, func: CUfunction) -> Result<i32, CUresult> {
        if let Some(f) = self.cu_func_get_attribute {
            let mut val: c_int = 0;
            let res = unsafe { f(&mut val, attrib, func) };
            if res == CUDA_SUCCESS { Ok(val) } else { Err(res) }
        } else {
            Err(CUDA_ERROR_NOT_SUPPORTED)
        }
    }

    pub fn func_set_attribute(&self, func: CUfunction, attrib: i32, value: i32) -> CUresult {
        if let Some(f) = self.cu_func_set_attribute {
            unsafe { f(func, attrib, value) }
        } else {
            CUDA_ERROR_NOT_SUPPORTED
        }
    }

    pub fn func_set_cache_config(&self, func: CUfunction, config: i32) -> CUresult {
        if let Some(f) = self.cu_func_set_cache_config {
            unsafe { f(func, config) }
        } else {
            CUDA_ERROR_NOT_SUPPORTED
        }
    }

    pub fn func_set_shared_mem_config(&self, func: CUfunction, config: i32) -> CUresult {
        if let Some(f) = self.cu_func_set_shared_mem_config {
            unsafe { f(func, config) }
        } else {
            CUDA_ERROR_NOT_SUPPORTED
        }
    }

    pub fn func_get_module(&self, func: CUfunction) -> Result<CUmodule, CUresult> {
        if let Some(f) = self.cu_func_get_module {
            let mut module: CUmodule = std::ptr::null_mut();
            let res = unsafe { f(&mut module, func) };
            if res == CUDA_SUCCESS { Ok(module) } else { Err(res) }
        } else {
            Err(CUDA_ERROR_NOT_SUPPORTED)
        }
    }

    pub fn func_get_name(&self, func: CUfunction) -> Result<String, CUresult> {
        if let Some(f) = self.cu_func_get_name {
            let mut name_ptr: *const c_char = std::ptr::null();
            let res = unsafe { f(&mut name_ptr, func) };
            if res == CUDA_SUCCESS && !name_ptr.is_null() {
                let name = unsafe { CStr::from_ptr(name_ptr) };
                Ok(name.to_string_lossy().into_owned())
            } else if res == CUDA_SUCCESS {
                Ok(String::new())
            } else {
                Err(res)
            }
        } else {
            Err(CUDA_ERROR_NOT_SUPPORTED)
        }
    }

    pub fn occupancy_max_active_blocks(&self, func: CUfunction, block_size: i32, dynamic_smem_size: u64) -> Result<i32, CUresult> {
        if let Some(f) = self.cu_occupancy_max_active_blocks {
            let mut num_blocks: c_int = 0;
            let res = unsafe { f(&mut num_blocks, func, block_size, dynamic_smem_size as usize) };
            if res == CUDA_SUCCESS { Ok(num_blocks) } else { Err(res) }
        } else {
            Err(CUDA_ERROR_NOT_SUPPORTED)
        }
    }

    pub fn occupancy_max_active_blocks_with_flags(&self, func: CUfunction, block_size: i32, dynamic_smem_size: u64, flags: u32) -> Result<i32, CUresult> {
        if let Some(f) = self.cu_occupancy_max_active_blocks_with_flags {
            let mut num_blocks: c_int = 0;
            let res = unsafe { f(&mut num_blocks, func, block_size, dynamic_smem_size as usize, flags as c_uint) };
            if res == CUDA_SUCCESS { Ok(num_blocks) } else { Err(res) }
        } else {
            self.occupancy_max_active_blocks(func, block_size, dynamic_smem_size)
        }
    }

    pub fn occupancy_available_dynamic_smem(&self, func: CUfunction, num_blocks: i32, block_size: i32) -> Result<u64, CUresult> {
        if let Some(f) = self.cu_occupancy_available_dynamic_smem {
            let mut smem: usize = 0;
            let res = unsafe { f(&mut smem, func, num_blocks, block_size) };
            if res == CUDA_SUCCESS { Ok(smem as u64) } else { Err(res) }
        } else {
            Err(CUDA_ERROR_NOT_SUPPORTED)
        }
    }

    // ── Stream Management ─────────────────────────────────────────

    pub fn stream_create(&self, flags: u32) -> Result<CUstream, CUresult> {
        let mut stream: CUstream = std::ptr::null_mut();
        let res = unsafe { (self.cu_stream_create)(&mut stream, flags as c_uint) };
        if res == CUDA_SUCCESS { Ok(stream) } else { Err(res) }
    }

    pub fn stream_create_with_priority(&self, flags: u32, priority: i32) -> Result<CUstream, CUresult> {
        if let Some(func) = self.cu_stream_create_with_priority {
            let mut stream: CUstream = std::ptr::null_mut();
            let res = unsafe { func(&mut stream, flags as c_uint, priority) };
            if res == CUDA_SUCCESS { Ok(stream) } else { Err(res) }
        } else {
            self.stream_create(flags)
        }
    }

    pub fn stream_destroy(&self, stream: CUstream) -> CUresult {
        unsafe { (self.cu_stream_destroy)(stream) }
    }

    pub fn stream_synchronize(&self, stream: CUstream) -> CUresult {
        unsafe { (self.cu_stream_synchronize)(stream) }
    }

    pub fn stream_query(&self, stream: CUstream) -> CUresult {
        unsafe { (self.cu_stream_query)(stream) }
    }

    pub fn stream_wait_event(&self, stream: CUstream, event: CUevent, flags: u32) -> CUresult {
        if let Some(func) = self.cu_stream_wait_event {
            unsafe { func(stream, event, flags as c_uint) }
        } else {
            CUDA_ERROR_NOT_SUPPORTED
        }
    }

    pub fn stream_get_priority(&self, stream: CUstream) -> Result<i32, CUresult> {
        if let Some(func) = self.cu_stream_get_priority {
            let mut priority: c_int = 0;
            let res = unsafe { func(stream, &mut priority) };
            if res == CUDA_SUCCESS { Ok(priority) } else { Err(res) }
        } else {
            Ok(0)
        }
    }

    pub fn stream_get_flags(&self, stream: CUstream) -> Result<u32, CUresult> {
        if let Some(func) = self.cu_stream_get_flags {
            let mut flags: c_uint = 0;
            let res = unsafe { func(stream, &mut flags) };
            if res == CUDA_SUCCESS { Ok(flags) } else { Err(res) }
        } else {
            Ok(0)
        }
    }

    pub fn stream_get_ctx(&self, stream: CUstream) -> Result<CUcontext, CUresult> {
        if let Some(func) = self.cu_stream_get_ctx {
            let mut ctx: CUcontext = std::ptr::null_mut();
            let res = unsafe { func(stream, &mut ctx) };
            if res == CUDA_SUCCESS { Ok(ctx) } else { Err(res) }
        } else {
            Err(CUDA_ERROR_NOT_SUPPORTED)
        }
    }

    // ── Event Management ──────────────────────────────────────────

    pub fn event_create(&self, flags: u32) -> Result<CUevent, CUresult> {
        let mut event: CUevent = std::ptr::null_mut();
        let res = unsafe { (self.cu_event_create)(&mut event, flags as c_uint) };
        if res == CUDA_SUCCESS { Ok(event) } else { Err(res) }
    }

    pub fn event_destroy(&self, event: CUevent) -> CUresult {
        unsafe { (self.cu_event_destroy)(event) }
    }

    pub fn event_record(&self, event: CUevent, stream: CUstream) -> CUresult {
        unsafe { (self.cu_event_record)(event, stream) }
    }

    pub fn event_record_with_flags(&self, event: CUevent, stream: CUstream, flags: u32) -> CUresult {
        if let Some(func) = self.cu_event_record_with_flags {
            unsafe { func(event, stream, flags as c_uint) }
        } else {
            self.event_record(event, stream)
        }
    }

    pub fn event_synchronize(&self, event: CUevent) -> CUresult {
        unsafe { (self.cu_event_synchronize)(event) }
    }

    pub fn event_query(&self, event: CUevent) -> CUresult {
        unsafe { (self.cu_event_query)(event) }
    }

    pub fn event_elapsed_time(&self, start: CUevent, end: CUevent) -> Result<f32, CUresult> {
        let mut ms: f32 = 0.0;
        let res = unsafe { (self.cu_event_elapsed_time)(&mut ms, start, end) };
        if res == CUDA_SUCCESS { Ok(ms) } else { Err(res) }
    }

    // ── Pointer Queries ───────────────────────────────────────────

    pub fn pointer_get_attribute(&self, attribute: i32, ptr: CUdeviceptr) -> Result<u64, CUresult> {
        if let Some(func) = self.cu_pointer_get_attribute {
            let mut data: u64 = 0;
            let res = unsafe { func(&mut data as *mut u64 as *mut c_void, attribute, ptr) };
            if res == CUDA_SUCCESS { Ok(data) } else { Err(res) }
        } else {
            Err(CUDA_ERROR_NOT_SUPPORTED)
        }
    }

    pub fn pointer_set_attribute(&self, attribute: i32, ptr: CUdeviceptr, value: u64) -> CUresult {
        if let Some(func) = self.cu_pointer_set_attribute {
            unsafe { func(&value as *const u64 as *const c_void, attribute, ptr) }
        } else {
            CUDA_ERROR_NOT_SUPPORTED
        }
    }
}

/// Convert a CUresult error code to a human-readable string.
pub fn cuda_error_name(result: CUresult) -> &'static str {
    match result {
        0 => "CUDA_SUCCESS",
        1 => "CUDA_ERROR_INVALID_VALUE",
        2 => "CUDA_ERROR_OUT_OF_MEMORY",
        3 => "CUDA_ERROR_NOT_INITIALIZED",
        4 => "CUDA_ERROR_DEINITIALIZED",
        100 => "CUDA_ERROR_NO_DEVICE",
        101 => "CUDA_ERROR_INVALID_DEVICE",
        200 => "CUDA_ERROR_INVALID_IMAGE",
        201 => "CUDA_ERROR_INVALID_CONTEXT",
        209 => "CUDA_ERROR_NO_BINARY_FOR_GPU",
        300 => "CUDA_ERROR_NOT_FOUND",
        400 => "CUDA_ERROR_INVALID_HANDLE",
        500 => "CUDA_ERROR_NOT_READY",
        700 => "CUDA_ERROR_ILLEGAL_ADDRESS",
        701 => "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES",
        702 => "CUDA_ERROR_LAUNCH_TIMEOUT",
        719 => "CUDA_ERROR_LAUNCH_FAILED",
        801 => "CUDA_ERROR_NOT_SUPPORTED",
        _ => "CUDA_ERROR_UNKNOWN",
    }
}
