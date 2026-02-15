use serde::{Deserialize, Serialize};

use crate::handle::NetworkHandle;

/// Serialized kernel parameter value.
#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct KernelParam {
    /// Raw bytes of the parameter value
    pub data: Vec<u8>,
}

/// CUDA Driver API commands sent from client to server.
#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub enum CudaCommand {
    // ── Initialization ──────────────────────────────────────
    Init { flags: u32 },
    DriverGetVersion,

    // ── Device Management ───────────────────────────────────
    DeviceGetCount,
    DeviceGet { ordinal: i32 },
    DeviceGetName { device: NetworkHandle },
    DeviceGetAttribute { attrib: i32, device: NetworkHandle },
    DeviceTotalMem { device: NetworkHandle },
    DeviceComputeCapability { device: NetworkHandle },
    DeviceGetUuid { device: NetworkHandle },
    DeviceGetP2PAttribute { attrib: i32, src_device: NetworkHandle, dst_device: NetworkHandle },
    DeviceCanAccessPeer { device: NetworkHandle, peer_device: NetworkHandle },
    DeviceGetByPCIBusId { pci_bus_id: String },
    DeviceGetPCIBusId { device: NetworkHandle },
    DeviceGetDefaultMemPool { device: NetworkHandle },
    DeviceGetMemPool { device: NetworkHandle },
    DeviceSetMemPool { device: NetworkHandle, mem_pool: NetworkHandle },
    DeviceGetTexture1DLinearMaxWidth { format: u32, num_channels: u32, device: NetworkHandle },
    DeviceGetExecAffinitySupport { affinity_type: i32, device: NetworkHandle },

    // ── Primary Context ─────────────────────────────────────
    DevicePrimaryCtxRetain { device: NetworkHandle },
    DevicePrimaryCtxRelease { device: NetworkHandle },
    DevicePrimaryCtxReset { device: NetworkHandle },
    DevicePrimaryCtxGetState { device: NetworkHandle },
    DevicePrimaryCtxSetFlags { device: NetworkHandle, flags: u32 },

    // ── Context Management ──────────────────────────────────
    CtxCreate { flags: u32, device: NetworkHandle },
    CtxDestroy { ctx: NetworkHandle },
    CtxSetCurrent { ctx: NetworkHandle },
    CtxGetCurrent,
    CtxSynchronize,
    CtxPushCurrent { ctx: NetworkHandle },
    CtxPopCurrent,
    CtxGetDevice,
    CtxSetCacheConfig { config: i32 },
    CtxGetCacheConfig,
    CtxSetLimit { limit: i32, value: u64 },
    CtxGetLimit { limit: i32 },
    CtxGetStreamPriorityRange,
    CtxGetApiVersion { ctx: NetworkHandle },
    CtxGetFlags,
    CtxSetFlags { flags: u32 },
    CtxResetPersistingL2Cache,

    // ── Module Management ───────────────────────────────────
    ModuleLoadData { image: Vec<u8> },
    ModuleUnload { module: NetworkHandle },
    ModuleGetFunction { module: NetworkHandle, name: String },
    ModuleGetGlobal { module: NetworkHandle, name: String },

    // ── Memory Management ───────────────────────────────────
    MemAlloc { byte_size: u64 },
    MemFree { dptr: NetworkHandle },
    MemcpyHtoD {
        dst: NetworkHandle,
        src_data: Vec<u8>,
        byte_count: u64,
    },
    MemcpyDtoH {
        src: NetworkHandle,
        byte_count: u64,
    },
    MemcpyDtoD {
        dst: NetworkHandle,
        src: NetworkHandle,
        byte_count: u64,
    },
    MemcpyHtoDAsync {
        dst: NetworkHandle,
        src_data: Vec<u8>,
        byte_count: u64,
        stream: NetworkHandle,
    },
    MemcpyDtoHAsync {
        src: NetworkHandle,
        byte_count: u64,
        stream: NetworkHandle,
    },
    MemcpyDtoDAsync {
        dst: NetworkHandle,
        src: NetworkHandle,
        byte_count: u64,
        stream: NetworkHandle,
    },
    MemsetD8 {
        dst: NetworkHandle,
        value: u8,
        count: u64,
    },
    MemsetD16 {
        dst: NetworkHandle,
        value: u16,
        count: u64,
    },
    MemsetD32 {
        dst: NetworkHandle,
        value: u32,
        count: u64,
    },
    MemsetD8Async {
        dst: NetworkHandle,
        value: u8,
        count: u64,
        stream: NetworkHandle,
    },
    MemsetD16Async {
        dst: NetworkHandle,
        value: u16,
        count: u64,
        stream: NetworkHandle,
    },
    MemsetD32Async {
        dst: NetworkHandle,
        value: u32,
        count: u64,
        stream: NetworkHandle,
    },
    MemGetInfo,
    MemGetAddressRange { dptr: NetworkHandle },
    MemAllocHost { byte_size: u64 },
    MemFreeHost { ptr: NetworkHandle },
    MemHostAlloc { byte_size: u64, flags: u32 },
    MemHostGetDevicePointer { host_ptr: NetworkHandle, flags: u32 },
    MemHostGetFlags { host_ptr: NetworkHandle },
    MemAllocManaged { byte_size: u64, flags: u32 },
    MemAllocPitch { width: u64, height: u64, element_size: u32 },
    MemHostRegister { byte_size: u64, flags: u32 },
    MemHostUnregister { ptr: NetworkHandle },
    MemPrefetchAsync { dptr: NetworkHandle, count: u64, dst_device: NetworkHandle, stream: NetworkHandle },
    MemAdvise { dptr: NetworkHandle, count: u64, advice: i32, device: NetworkHandle },
    MemRangeGetAttribute { dptr: NetworkHandle, count: u64, attribute: i32 },

    // ── Execution Control ───────────────────────────────────
    LaunchKernel {
        func: NetworkHandle,
        grid_dim: [u32; 3],
        block_dim: [u32; 3],
        shared_mem_bytes: u32,
        stream: NetworkHandle,
        kernel_params: Vec<KernelParam>,
    },
    LaunchCooperativeKernel {
        func: NetworkHandle,
        grid_dim: [u32; 3],
        block_dim: [u32; 3],
        shared_mem_bytes: u32,
        stream: NetworkHandle,
        kernel_params: Vec<KernelParam>,
    },
    FuncGetAttribute { attrib: i32, func: NetworkHandle },
    FuncSetAttribute { attrib: i32, func: NetworkHandle, value: i32 },
    FuncSetCacheConfig { func: NetworkHandle, config: i32 },
    FuncSetSharedMemConfig { func: NetworkHandle, config: i32 },
    FuncGetModule { func: NetworkHandle },
    FuncGetName { func: NetworkHandle },
    OccupancyMaxActiveBlocksPerMultiprocessor { func: NetworkHandle, block_size: i32, dynamic_smem_size: u64 },
    OccupancyMaxActiveBlocksPerMultiprocessorWithFlags { func: NetworkHandle, block_size: i32, dynamic_smem_size: u64, flags: u32 },
    OccupancyAvailableDynamicSMemPerBlock { func: NetworkHandle, num_blocks: i32, block_size: i32 },

    // ── Stream Management ───────────────────────────────────
    StreamCreate { flags: u32 },
    StreamCreateWithPriority { flags: u32, priority: i32 },
    StreamDestroy { stream: NetworkHandle },
    StreamSynchronize { stream: NetworkHandle },
    StreamQuery { stream: NetworkHandle },
    StreamWaitEvent { stream: NetworkHandle, event: NetworkHandle, flags: u32 },
    StreamGetPriority { stream: NetworkHandle },
    StreamGetFlags { stream: NetworkHandle },
    StreamGetCtx { stream: NetworkHandle },

    // ── Event Management ────────────────────────────────────
    EventCreate { flags: u32 },
    EventDestroy { event: NetworkHandle },
    EventRecord { event: NetworkHandle, stream: NetworkHandle },
    EventRecordWithFlags { event: NetworkHandle, stream: NetworkHandle, flags: u32 },
    EventSynchronize { event: NetworkHandle },
    EventQuery { event: NetworkHandle },
    EventElapsedTime { start: NetworkHandle, end: NetworkHandle },

    // ── Pointer Queries ─────────────────────────────────────
    PointerGetAttribute { attribute: i32, ptr: NetworkHandle },
    PointerGetAttributes { num_attributes: i32, attributes: Vec<i32>, ptr: NetworkHandle },
    PointerSetAttribute { attribute: i32, ptr: NetworkHandle, value: u64 },

    // ── Peer Access ─────────────────────────────────────────
    CtxEnablePeerAccess { peer_ctx: NetworkHandle, flags: u32 },
    CtxDisablePeerAccess { peer_ctx: NetworkHandle },

    // ── Memory Pools ────────────────────────────────────────
    MemPoolCreate { device: NetworkHandle, props_flags: u32 },
    MemPoolDestroy { pool: NetworkHandle },
    MemPoolTrimTo { pool: NetworkHandle, min_bytes_to_keep: u64 },
    MemPoolSetAttribute { pool: NetworkHandle, attr: i32, value: u64 },
    MemPoolGetAttribute { pool: NetworkHandle, attr: i32 },
    MemAllocAsync { byte_size: u64, stream: NetworkHandle },
    MemFreeAsync { dptr: NetworkHandle, stream: NetworkHandle },
    MemAllocFromPoolAsync { byte_size: u64, pool: NetworkHandle, stream: NetworkHandle },

    // ── Module Extended ─────────────────────────────────────
    ModuleLoad { fname: String },
    ModuleLoadDataEx { image: Vec<u8>, num_options: u32, options: Vec<i32>, option_values: Vec<u64> },
    ModuleLoadFatBinary { fat_cubin: Vec<u8> },
    LinkCreate { num_options: u32, options: Vec<i32>, option_values: Vec<u64> },
    LinkAddData { link: NetworkHandle, jit_type: i32, data: Vec<u8>, name: String, num_options: u32, options: Vec<i32>, option_values: Vec<u64> },
    LinkAddFile { link: NetworkHandle, jit_type: i32, path: String, num_options: u32, options: Vec<i32>, option_values: Vec<u64> },
    LinkComplete { link: NetworkHandle },
    LinkDestroy { link: NetworkHandle },
}

/// CUDA Driver API responses sent from server to client.
#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub enum CudaResponse {
    /// Generic success with no return data.
    Success,

    /// Error from the CUDA runtime.
    Error { code: i32, message: String },

    /// cuDriverGetVersion result.
    DriverVersion(i32),

    /// cuDeviceGetCount result.
    DeviceCount(i32),

    /// cuDeviceGet result.
    Device(NetworkHandle),

    /// cuDeviceGetName result.
    DeviceName(String),

    /// cuDeviceGetAttribute result.
    DeviceAttribute(i32),

    /// cuDeviceTotalMem result.
    DeviceTotalMem(u64),

    /// cuDeviceComputeCapability result.
    ComputeCapability { major: i32, minor: i32 },

    /// cuDeviceGetUuid result.
    DeviceUuid(Vec<u8>),

    /// cuDeviceGetPCIBusId result.
    DevicePCIBusId(String),

    /// cuDeviceGetP2PAttribute result.
    P2PAttribute(i32),

    /// Boolean result (cuDeviceCanAccessPeer, cuDeviceGetExecAffinitySupport).
    BoolResult(bool),

    /// cuDevicePrimaryCtxGetState result.
    PrimaryCtxState { flags: u32, active: bool },

    /// Handle to a memory pool.
    MemPool(NetworkHandle),

    /// cuDeviceGetTexture1DLinearMaxWidth result.
    Texture1DMaxWidth(u64),

    /// cuCtxCreate / cuCtxGetCurrent / cuDevicePrimaryCtxRetain result.
    Context(NetworkHandle),

    /// cuCtxGetDevice result.
    ContextDevice(NetworkHandle),

    /// cuCtxGetCacheConfig result.
    CacheConfig(i32),

    /// cuCtxGetLimit result.
    ContextLimit(u64),

    /// cuCtxGetStreamPriorityRange result.
    StreamPriorityRange { least: i32, greatest: i32 },

    /// cuCtxGetApiVersion result.
    ContextApiVersion(u32),

    /// cuCtxGetFlags result.
    ContextFlags(u32),

    /// cuModuleLoadData result.
    Module(NetworkHandle),

    /// cuModuleGetFunction result.
    Function(NetworkHandle),

    /// cuModuleGetGlobal result.
    GlobalPtr { ptr: NetworkHandle, size: u64 },

    /// cuMemAlloc result.
    MemAllocated(NetworkHandle),

    /// cuMemAllocPitch result.
    MemAllocPitch { dptr: NetworkHandle, pitch: u64 },

    /// cuMemGetInfo result.
    MemInfo { free: u64, total: u64 },

    /// cuMemGetAddressRange result.
    MemAddressRange { base: NetworkHandle, size: u64 },

    /// cuMemcpyDtoH result.
    MemoryData(Vec<u8>),

    /// Host memory pointer handle.
    HostPtr(NetworkHandle),

    /// cuMemHostGetDevicePointer result.
    HostDevicePtr(NetworkHandle),

    /// cuMemHostGetFlags result.
    HostFlags(u32),

    /// cuMemRangeGetAttribute result.
    MemRangeAttribute(Vec<u8>),

    /// cuStreamCreate result.
    Stream(NetworkHandle),

    /// cuStreamQuery result (true = complete).
    StreamStatus(bool),

    /// cuStreamGetPriority result.
    StreamPriority(i32),

    /// cuStreamGetFlags result.
    StreamFlags(u32),

    /// cuStreamGetCtx result.
    StreamCtx(NetworkHandle),

    /// cuEventCreate result.
    Event(NetworkHandle),

    /// cuEventQuery result (true = complete).
    EventStatus(bool),

    /// cuEventElapsedTime result.
    ElapsedTime(f32),

    /// cuPointerGetAttribute result.
    PointerAttribute(u64),

    /// cuPointerGetAttributes result.
    PointerAttributes(Vec<u64>),

    /// cuFuncGetAttribute result.
    FuncAttribute(i32),

    /// cuFuncGetModule result.
    FuncModule(NetworkHandle),

    /// cuFuncGetName result.
    FuncName(String),

    /// cuOccupancyMaxActiveBlocksPerMultiprocessor result.
    OccupancyBlocks(i32),

    /// cuOccupancyAvailableDynamicSMemPerBlock result.
    OccupancyDynamicSmem(u64),

    /// cuMemPoolGetAttribute result.
    MemPoolAttribute(u64),

    /// cuLinkCreate result.
    Linker(NetworkHandle),

    /// cuLinkComplete result.
    LinkCompleted { cubin_data: Vec<u8> },
}
