mod context;
mod device;
mod event;
mod execution;
mod memory;
mod module;
mod stream;

use std::ffi::c_void;
use std::sync::Arc;

use dashmap::DashMap;
use tracing::{error, info, warn};

use rgpu_protocol::cuda_commands::{CudaCommand, CudaResponse};
use rgpu_protocol::handle::{NetworkHandle, ResourceType};

use crate::cuda_driver::{self, CudaDriver, CUDA_SUCCESS};
use crate::session::Session;

/// Server-side CUDA command executor.
/// Executes CUDA driver API commands on real GPU hardware via dynamically loaded CUDA driver.
pub struct CudaExecutor {
    /// GPU information discovered at startup
    pub(crate) gpu_infos: Vec<rgpu_protocol::gpu_info::GpuInfo>,
    /// The real CUDA driver (loaded via libloading)
    pub(crate) driver: Option<Arc<CudaDriver>>,
    /// Maps NetworkHandle -> real CUdevice ordinal
    pub(crate) device_handles: DashMap<NetworkHandle, cuda_driver::CUdevice>,
    /// Maps NetworkHandle -> real CUcontext pointer
    pub(crate) context_handles: DashMap<NetworkHandle, cuda_driver::CUcontext>,
    /// Maps NetworkHandle -> real CUmodule pointer
    pub(crate) module_handles: DashMap<NetworkHandle, cuda_driver::CUmodule>,
    /// Maps NetworkHandle -> real CUfunction pointer
    pub(crate) function_handles: DashMap<NetworkHandle, cuda_driver::CUfunction>,
    /// Maps NetworkHandle -> real CUdeviceptr (GPU memory address)
    pub(crate) memory_handles: DashMap<NetworkHandle, cuda_driver::CUdeviceptr>,
    /// Maps NetworkHandle -> allocated byte size for memory
    pub(crate) memory_sizes: DashMap<NetworkHandle, u64>,
    /// Maps NetworkHandle -> real CUstream pointer
    pub(crate) stream_handles: DashMap<NetworkHandle, cuda_driver::CUstream>,
    /// Maps NetworkHandle -> real CUevent pointer
    pub(crate) event_handles: DashMap<NetworkHandle, cuda_driver::CUevent>,
    /// Maps NetworkHandle -> real host memory pointer (cuMemAllocHost / cuMemHostAlloc)
    pub(crate) host_memory_handles: DashMap<NetworkHandle, *mut c_void>,
    /// Maps NetworkHandle -> real CUmemoryPool pointer
    pub(crate) mempool_handles: DashMap<NetworkHandle, cuda_driver::CUmemoryPool>,
    /// Maps NetworkHandle -> real CUlinkState pointer
    pub(crate) linker_handles: DashMap<NetworkHandle, cuda_driver::CUlinkState>,
    /// Per-session current CUDA context tracking.
    /// CUDA context state is thread-local, but tokio migrates tasks between threads.
    /// We track each session's current context and re-establish it before each command.
    pub(crate) session_current_ctx: DashMap<u32, cuda_driver::CUcontext>,
}

// SAFETY: CUDA driver pointers are valid across threads when used with proper context management
unsafe impl Send for CudaExecutor {}
unsafe impl Sync for CudaExecutor {}

impl CudaExecutor {
    pub fn new(gpu_infos: Vec<rgpu_protocol::gpu_info::GpuInfo>) -> Self {
        // Try to load the CUDA driver
        let driver = match CudaDriver::load() {
            Ok(d) => {
                // Initialize CUDA
                let res = d.init(0);
                if res == CUDA_SUCCESS {
                    info!("CUDA driver initialized successfully");
                    Some(d)
                } else {
                    error!(
                        "cuInit failed: {} ({})",
                        cuda_driver::cuda_error_name(res),
                        res
                    );
                    None
                }
            }
            Err(e) => {
                warn!("CUDA driver not available: {} - using fallback mode", e);
                None
            }
        };

        Self {
            gpu_infos,
            driver,
            device_handles: DashMap::new(),
            context_handles: DashMap::new(),
            module_handles: DashMap::new(),
            function_handles: DashMap::new(),
            memory_handles: DashMap::new(),
            memory_sizes: DashMap::new(),
            stream_handles: DashMap::new(),
            event_handles: DashMap::new(),
            host_memory_handles: DashMap::new(),
            mempool_handles: DashMap::new(),
            linker_handles: DashMap::new(),
            session_current_ctx: DashMap::new(),
        }
    }

    /// Check if the real CUDA driver is available.
    pub(crate) fn driver(&self) -> Result<&CudaDriver, CudaResponse> {
        self.driver.as_deref().ok_or(CudaResponse::Error {
            code: 3, // CUDA_ERROR_NOT_INITIALIZED
            message: "CUDA driver not loaded on server".to_string(),
        })
    }

    /// Convert a CUresult to a CudaResponse::Error.
    pub(crate) fn cuda_err(code: cuda_driver::CUresult) -> CudaResponse {
        CudaResponse::Error {
            code,
            message: cuda_driver::cuda_error_name(code).to_string(),
        }
    }

    /// Re-establish the CUDA context for a session on the current thread.
    /// This is needed because tokio may migrate async tasks between OS threads,
    /// and CUDA context state is thread-local.
    fn ensure_session_context(&self, session: &Session) {
        if let Some(d) = &self.driver {
            if let Some(ctx) = self.session_current_ctx.get(&session.session_id) {
                let _ = d.ctx_set_current(*ctx);
            }
        }
    }

    /// Execute a CUDA command and return the response.
    pub fn execute(&self, session: &Session, cmd: CudaCommand) -> CudaResponse {
        // Re-establish the CUDA context for this session on the current thread.
        // Skip for commands that don't need a context (Init, DriverGetVersion, DeviceGet*).
        match &cmd {
            CudaCommand::Init { .. }
            | CudaCommand::DriverGetVersion
            | CudaCommand::DeviceGetCount
            | CudaCommand::DeviceGet { .. } => {}
            _ => self.ensure_session_context(session),
        }

        match cmd {
            // ── Init / Driver ──────────────────────────────────────
            CudaCommand::Init { flags } => self.handle_init(session, flags),
            CudaCommand::DriverGetVersion => self.handle_driver_get_version(),

            // ── Device ─────────────────────────────────────────────
            CudaCommand::DeviceGetCount => self.handle_device_get_count(session),
            CudaCommand::DeviceGet { ordinal } => self.handle_device_get(session, ordinal),
            CudaCommand::DeviceGetName { device } => self.handle_device_get_name(device),
            CudaCommand::DeviceGetAttribute { attrib, device } => self.handle_device_get_attribute(attrib, device),
            CudaCommand::DeviceTotalMem { device } => self.handle_device_total_mem(device),
            CudaCommand::DeviceComputeCapability { device } => self.handle_device_compute_capability(device),
            CudaCommand::DeviceGetUuid { device } => self.handle_device_get_uuid(device),
            CudaCommand::DeviceGetP2PAttribute { attrib, src_device, dst_device } => self.handle_device_get_p2p_attribute(attrib, src_device, dst_device),
            CudaCommand::DeviceCanAccessPeer { device, peer_device } => self.handle_device_can_access_peer(device, peer_device),
            CudaCommand::DeviceGetByPCIBusId { pci_bus_id } => self.handle_device_get_by_pci_bus_id(session, pci_bus_id),
            CudaCommand::DeviceGetPCIBusId { device } => self.handle_device_get_pci_bus_id(device),
            CudaCommand::DeviceGetDefaultMemPool { device } => self.handle_device_get_default_mem_pool(session, device),
            CudaCommand::DeviceGetMemPool { device } => self.handle_device_get_mem_pool(session, device),
            CudaCommand::DeviceSetMemPool { device, mem_pool } => self.handle_device_set_mem_pool(device, mem_pool),
            CudaCommand::DeviceGetTexture1DLinearMaxWidth { format: _, num_channels: _, device: _ } => self.handle_device_get_texture_1d_linear_max_width(),
            CudaCommand::DeviceGetExecAffinitySupport { affinity_type: _, device: _ } => self.handle_device_get_exec_affinity_support(),

            // ── Primary Context ────────────────────────────────────
            CudaCommand::DevicePrimaryCtxRetain { device } => self.handle_device_primary_ctx_retain(session, device),
            CudaCommand::DevicePrimaryCtxRelease { device } => self.handle_device_primary_ctx_release(device),
            CudaCommand::DevicePrimaryCtxReset { device } => self.handle_device_primary_ctx_reset(device),
            CudaCommand::DevicePrimaryCtxGetState { device } => self.handle_device_primary_ctx_get_state(device),
            CudaCommand::DevicePrimaryCtxSetFlags { device, flags } => self.handle_device_primary_ctx_set_flags(device, flags),

            // ── Context Management ─────────────────────────────────
            CudaCommand::CtxCreate { flags, device } => self.handle_ctx_create(session, flags, device),
            CudaCommand::CtxDestroy { ctx } => self.handle_ctx_destroy(session, ctx),
            CudaCommand::CtxSetCurrent { ctx } => self.handle_ctx_set_current(session, ctx),
            CudaCommand::CtxGetCurrent => self.handle_ctx_get_current(session),
            CudaCommand::CtxSynchronize => self.handle_ctx_synchronize(),
            CudaCommand::CtxPushCurrent { ctx } => self.handle_ctx_push_current(session, ctx),
            CudaCommand::CtxPopCurrent => self.handle_ctx_pop_current(session),
            CudaCommand::CtxGetDevice => self.handle_ctx_get_device(session),
            CudaCommand::CtxSetCacheConfig { config } => self.handle_ctx_set_cache_config(config),
            CudaCommand::CtxGetCacheConfig => self.handle_ctx_get_cache_config(),
            CudaCommand::CtxSetLimit { limit, value } => self.handle_ctx_set_limit(limit, value),
            CudaCommand::CtxGetLimit { limit } => self.handle_ctx_get_limit(limit),
            CudaCommand::CtxGetStreamPriorityRange => self.handle_ctx_get_stream_priority_range(),
            CudaCommand::CtxGetApiVersion { ctx } => self.handle_ctx_get_api_version(ctx),
            CudaCommand::CtxGetFlags => self.handle_ctx_get_flags(),
            CudaCommand::CtxSetFlags { flags } => self.handle_ctx_set_flags(flags),
            CudaCommand::CtxResetPersistingL2Cache => self.handle_ctx_reset_persisting_l2_cache(),
            CudaCommand::CtxEnablePeerAccess { peer_ctx, flags } => self.handle_ctx_enable_peer_access(peer_ctx, flags),
            CudaCommand::CtxDisablePeerAccess { peer_ctx } => self.handle_ctx_disable_peer_access(peer_ctx),

            // ── Module Management ──────────────────────────────────
            CudaCommand::ModuleLoadData { image } => self.handle_module_load_data(session, image),
            CudaCommand::ModuleUnload { module } => self.handle_module_unload(session, module),
            CudaCommand::ModuleGetFunction { module, name } => self.handle_module_get_function(session, module, name),
            CudaCommand::ModuleGetGlobal { module, name } => self.handle_module_get_global(session, module, name),
            CudaCommand::ModuleLoad { fname } => self.handle_module_load(session, fname),
            CudaCommand::ModuleLoadDataEx { image, num_options: _, options: _, option_values: _ } => self.handle_module_load_data_ex(session, image),
            CudaCommand::ModuleLoadFatBinary { fat_cubin } => self.handle_module_load_fat_binary(session, fat_cubin),
            CudaCommand::LinkCreate { num_options: _, options: _, option_values: _ } => self.handle_link_create(session),
            CudaCommand::LinkAddData { link, jit_type, data, name, num_options: _, options: _, option_values: _ } => self.handle_link_add_data(link, jit_type, data, name),
            CudaCommand::LinkAddFile { link, jit_type, path, num_options: _, options: _, option_values: _ } => self.handle_link_add_file(link, jit_type, path),
            CudaCommand::LinkComplete { link } => self.handle_link_complete(link),
            CudaCommand::LinkDestroy { link } => self.handle_link_destroy(session, link),

            // ── Memory Management ──────────────────────────────────
            CudaCommand::MemAlloc { byte_size } => self.handle_mem_alloc(session, byte_size),
            CudaCommand::MemFree { dptr } => self.handle_mem_free(session, dptr),
            CudaCommand::MemcpyHtoD { dst, src_data, byte_count } => self.handle_memcpy_htod(session, dst, src_data, byte_count),
            CudaCommand::MemcpyDtoH { src, byte_count } => self.handle_memcpy_dtoh(session, src, byte_count),
            CudaCommand::MemcpyDtoD { dst, src, byte_count } => self.handle_memcpy_dtod(dst, src, byte_count),
            CudaCommand::MemcpyHtoDAsync { dst, src_data, byte_count, stream: _ } => self.handle_memcpy_htod_async(session, dst, src_data, byte_count),
            CudaCommand::MemcpyDtoHAsync { src, byte_count, stream: _ } => self.handle_memcpy_dtoh_async(session, src, byte_count),
            CudaCommand::MemcpyDtoDAsync { dst, src, byte_count, stream: _ } => self.handle_memcpy_dtod_async(dst, src, byte_count),
            CudaCommand::Memcpy2DHtoD { dst, dst_x_in_bytes, dst_y, dst_pitch, src_data, width_in_bytes, height } => {
                self.handle_memcpy_2d_htod(dst, dst_x_in_bytes, dst_y, dst_pitch, src_data, width_in_bytes, height)
            }
            CudaCommand::Memcpy2DDtoH { src, src_x_in_bytes, src_y, src_pitch, width_in_bytes, height } => {
                self.handle_memcpy_2d_dtoh(src, src_x_in_bytes, src_y, src_pitch, width_in_bytes, height)
            }
            CudaCommand::Memcpy2DDtoD { dst, dst_x_in_bytes, dst_y, dst_pitch, src, src_x_in_bytes, src_y, src_pitch, width_in_bytes, height } => {
                self.handle_memcpy_2d_dtod(dst, dst_x_in_bytes, dst_y, dst_pitch, src, src_x_in_bytes, src_y, src_pitch, width_in_bytes, height)
            }
            CudaCommand::MemsetD8 { dst, value, count } => self.handle_memset_d8(dst, value, count),
            CudaCommand::MemsetD16 { dst, value, count } => self.handle_memset_d16(dst, value, count),
            CudaCommand::MemsetD32 { dst, value, count } => self.handle_memset_d32(dst, value, count),
            CudaCommand::MemsetD8Async { dst, value, count, stream: _ } => self.handle_memset_d8_async(dst, value, count),
            CudaCommand::MemsetD16Async { dst, value, count, stream: _ } => self.handle_memset_d16_async(dst, value, count),
            CudaCommand::MemsetD32Async { dst, value, count, stream: _ } => self.handle_memset_d32_async(dst, value, count),
            CudaCommand::MemGetInfo => self.handle_mem_get_info(),
            CudaCommand::MemGetAddressRange { dptr } => self.handle_mem_get_address_range(dptr),
            CudaCommand::MemAllocHost { byte_size } => self.handle_mem_alloc_host(session, byte_size),
            CudaCommand::MemFreeHost { ptr } => self.handle_mem_free_host(session, ptr),
            CudaCommand::MemHostAlloc { byte_size, flags } => self.handle_mem_host_alloc(session, byte_size, flags),
            CudaCommand::MemHostGetDevicePointer { host_ptr, flags } => self.handle_mem_host_get_device_pointer(session, host_ptr, flags),
            CudaCommand::MemHostGetFlags { host_ptr } => self.handle_mem_host_get_flags(host_ptr),
            CudaCommand::MemAllocManaged { byte_size, flags } => self.handle_mem_alloc_managed(session, byte_size, flags),
            CudaCommand::MemAllocPitch { width, height, element_size } => self.handle_mem_alloc_pitch(session, width, height, element_size),
            CudaCommand::MemHostRegister { byte_size: _, flags: _ } => self.handle_mem_host_register(),
            CudaCommand::MemHostUnregister { ptr: _ } => self.handle_mem_host_unregister(),
            CudaCommand::MemPrefetchAsync { dptr: _, count: _, dst_device: _, stream: _ } => self.handle_mem_prefetch_async(),
            CudaCommand::MemAdvise { dptr: _, count: _, advice: _, device: _ } => self.handle_mem_advise(),
            CudaCommand::MemRangeGetAttribute { dptr: _, count: _, attribute: _ } => self.handle_mem_range_get_attribute(),
            CudaCommand::PointerGetAttribute { attribute, ptr } => self.handle_pointer_get_attribute(attribute, ptr),
            CudaCommand::PointerGetAttributes { num_attributes: _, attributes, ptr } => self.handle_pointer_get_attributes(attributes, ptr),
            CudaCommand::PointerSetAttribute { attribute, ptr, value } => self.handle_pointer_set_attribute(attribute, ptr, value),
            CudaCommand::MemPoolCreate { device, props_flags: _ } => self.handle_mem_pool_create(session, device),
            CudaCommand::MemPoolDestroy { pool } => self.handle_mem_pool_destroy(session, pool),
            CudaCommand::MemPoolTrimTo { pool, min_bytes_to_keep } => self.handle_mem_pool_trim_to(pool, min_bytes_to_keep),
            CudaCommand::MemPoolSetAttribute { pool, attr: _, value: _ } => self.handle_mem_pool_set_attribute(pool),
            CudaCommand::MemPoolGetAttribute { pool, attr: _ } => self.handle_mem_pool_get_attribute(pool),
            CudaCommand::MemAllocAsync { byte_size, stream } => self.handle_mem_alloc_async(session, byte_size, stream),
            CudaCommand::MemFreeAsync { dptr, stream } => self.handle_mem_free_async(session, dptr, stream),
            CudaCommand::MemAllocFromPoolAsync { byte_size, pool, stream } => self.handle_mem_alloc_from_pool_async(session, byte_size, pool, stream),

            // ── Execution Control ──────────────────────────────────
            CudaCommand::LaunchKernel { func, grid_dim, block_dim, shared_mem_bytes, stream, kernel_params } => {
                self.handle_launch_kernel(session, func, grid_dim, block_dim, shared_mem_bytes, stream, kernel_params)
            }
            CudaCommand::LaunchCooperativeKernel { func, grid_dim, block_dim, shared_mem_bytes, stream, kernel_params } => {
                self.handle_launch_cooperative_kernel(session, func, grid_dim, block_dim, shared_mem_bytes, stream, kernel_params)
            }
            CudaCommand::FuncGetAttribute { attrib, func } => self.handle_func_get_attribute(attrib, func),
            CudaCommand::FuncSetAttribute { attrib, func, value } => self.handle_func_set_attribute(attrib, func, value),
            CudaCommand::FuncSetCacheConfig { func, config } => self.handle_func_set_cache_config(func, config),
            CudaCommand::FuncSetSharedMemConfig { func, config } => self.handle_func_set_shared_mem_config(func, config),
            CudaCommand::FuncGetModule { func } => self.handle_func_get_module(session, func),
            CudaCommand::FuncGetName { func } => self.handle_func_get_name(func),
            CudaCommand::OccupancyMaxActiveBlocksPerMultiprocessor { func, block_size, dynamic_smem_size } => {
                self.handle_occupancy_max_active_blocks(func, block_size, dynamic_smem_size)
            }
            CudaCommand::OccupancyMaxActiveBlocksPerMultiprocessorWithFlags { func, block_size, dynamic_smem_size, flags } => {
                self.handle_occupancy_max_active_blocks_with_flags(func, block_size, dynamic_smem_size, flags)
            }
            CudaCommand::OccupancyAvailableDynamicSMemPerBlock { func, num_blocks, block_size } => {
                self.handle_occupancy_available_dynamic_smem(func, num_blocks, block_size)
            }

            // ── Stream Management ──────────────────────────────────
            CudaCommand::StreamCreate { flags } => self.handle_stream_create(session, flags),
            CudaCommand::StreamCreateWithPriority { flags, priority } => self.handle_stream_create_with_priority(session, flags, priority),
            CudaCommand::StreamDestroy { stream } => self.handle_stream_destroy(session, stream),
            CudaCommand::StreamSynchronize { stream } => self.handle_stream_synchronize(stream),
            CudaCommand::StreamQuery { stream } => self.handle_stream_query(stream),
            CudaCommand::StreamWaitEvent { stream, event, flags } => self.handle_stream_wait_event(stream, event, flags),
            CudaCommand::StreamGetPriority { stream } => self.handle_stream_get_priority(stream),
            CudaCommand::StreamGetFlags { stream } => self.handle_stream_get_flags(stream),
            CudaCommand::StreamGetCtx { stream } => self.handle_stream_get_ctx(session, stream),

            // ── Event Management ───────────────────────────────────
            CudaCommand::EventCreate { flags } => self.handle_event_create(session, flags),
            CudaCommand::EventDestroy { event } => self.handle_event_destroy(session, event),
            CudaCommand::EventRecord { event, stream } => self.handle_event_record(event, stream),
            CudaCommand::EventRecordWithFlags { event, stream, flags } => self.handle_event_record_with_flags(event, stream, flags),
            CudaCommand::EventSynchronize { event } => self.handle_event_synchronize(event),
            CudaCommand::EventQuery { event } => self.handle_event_query(event),
            CudaCommand::EventElapsedTime { start, end } => self.handle_event_elapsed_time(start, end),
        }
    }

    /// Fallback device attribute values when no real CUDA driver is available.
    pub(crate) fn get_device_attribute_fallback(&self, attrib: i32) -> i32 {
        match attrib {
            1 => 1024,       // CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
            2 => 1024,       // CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X
            3 => 1024,       // CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y
            4 => 64,         // CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z
            5 => 2147483647, // CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X
            6 => 65535,      // CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y
            7 => 65535,      // CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z
            8 => 49152,      // CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
            13 => 32,        // CU_DEVICE_ATTRIBUTE_WARP_SIZE
            16 => 65536,     // CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK
            21 => 1,         // CU_DEVICE_ATTRIBUTE_CLOCK_RATE (dummy)
            29 => 128,       // CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
            75 => 8,         // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
            76 => 6,         // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
            _ => 0,
        }
    }

    /// Get a reference to the CUDA driver, if loaded.
    /// Used by NvencExecutor to set CUDA context before NVENC calls.
    pub fn driver_ref(&self) -> Option<&CudaDriver> {
        self.driver.as_deref()
    }

    /// Resolve a CUDA context NetworkHandle to a real CUcontext pointer.
    /// Used by NvencExecutor to get the real CUDA context for encoding sessions.
    pub fn get_context_ptr(&self, handle: &NetworkHandle) -> Option<*mut c_void> {
        self.context_handles.get(handle).map(|ctx| *ctx as *mut c_void)
    }

    /// Resolve a CUDA device memory NetworkHandle to a real CUdeviceptr value.
    /// Used by NvencExecutor to register CUDA device memory as encoder input.
    pub fn get_device_ptr(&self, handle: &NetworkHandle) -> Option<u64> {
        self.memory_handles.get(handle).map(|dptr| *dptr)
    }

    /// Clean up all GPU resources owned by a disconnecting session.
    /// Destroys resources in reverse-dependency order to avoid dangling references.
    pub fn cleanup_session(&self, session: &Session) {
        // Remove per-session context tracking
        self.session_current_ctx.remove(&session.session_id);

        let handles = session.all_handles();
        if handles.is_empty() {
            return;
        }

        let driver = match &self.driver {
            Some(d) => d,
            None => return,
        };

        let mut cleaned = 0u32;

        // Pass 1: Events
        for h in handles.iter().filter(|h| h.resource_type == ResourceType::CuEvent) {
            if let Some((_, evt)) = self.event_handles.remove(h) {
                driver.event_destroy(evt);
                cleaned += 1;
            }
        }

        // Pass 2: Streams
        for h in handles.iter().filter(|h| h.resource_type == ResourceType::CuStream) {
            if let Some((_, stream)) = self.stream_handles.remove(h) {
                driver.stream_destroy(stream);
                cleaned += 1;
            }
        }

        // Pass 3: Device memory
        for h in handles.iter().filter(|h| h.resource_type == ResourceType::CuDevicePtr) {
            if let Some((_, ptr)) = self.memory_handles.remove(h) {
                driver.mem_free(ptr);
                self.memory_sizes.remove(h);
                cleaned += 1;
            }
        }

        // Pass 4: Host memory
        for h in handles.iter().filter(|h| h.resource_type == ResourceType::CuHostPtr) {
            if let Some((_, ptr)) = self.host_memory_handles.remove(h) {
                driver.mem_free_host(ptr);
                cleaned += 1;
            }
        }

        // Pass 5: Linkers
        for h in handles.iter().filter(|h| h.resource_type == ResourceType::CuLinker) {
            if let Some((_, link)) = self.linker_handles.remove(h) {
                driver.link_destroy(link);
                cleaned += 1;
            }
        }

        // Pass 6: Functions (no driver call, just remove tracking)
        for h in handles.iter().filter(|h| h.resource_type == ResourceType::CuFunction) {
            if self.function_handles.remove(h).is_some() {
                cleaned += 1;
            }
        }

        // Pass 7: Modules
        for h in handles.iter().filter(|h| h.resource_type == ResourceType::CuModule) {
            if let Some((_, module)) = self.module_handles.remove(h) {
                driver.module_unload(module);
                cleaned += 1;
            }
        }

        // Pass 8: Contexts
        for h in handles.iter().filter(|h| h.resource_type == ResourceType::CuContext) {
            if let Some((_, ctx)) = self.context_handles.remove(h) {
                driver.ctx_destroy(ctx);
                cleaned += 1;
            }
        }

        // Pass 9: Memory pools (no destroy for default pool)
        for h in handles.iter().filter(|h| h.resource_type == ResourceType::CuMemPool) {
            self.mempool_handles.remove(h);
        }

        // Devices are not destroyable, skip CuDevice

        if cleaned > 0 {
            info!(
                session_id = session.session_id,
                "cleaned up {} CUDA resource(s)", cleaned
            );
        }
    }
}
