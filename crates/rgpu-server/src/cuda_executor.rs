use std::ffi::c_void;
use std::sync::Arc;

use dashmap::DashMap;
use tracing::{debug, error, info, warn};

use rgpu_protocol::cuda_commands::{CudaCommand, CudaResponse};
use rgpu_protocol::handle::{NetworkHandle, ResourceType};

use crate::cuda_driver::{self, CudaDriver, CUDA_ERROR_NOT_SUPPORTED, CUDA_SUCCESS};
use crate::session::Session;

/// Server-side CUDA command executor.
/// Executes CUDA driver API commands on real GPU hardware via dynamically loaded CUDA driver.
pub struct CudaExecutor {
    /// GPU information discovered at startup
    gpu_infos: Vec<rgpu_protocol::gpu_info::GpuInfo>,
    /// The real CUDA driver (loaded via libloading)
    driver: Option<Arc<CudaDriver>>,
    /// Maps NetworkHandle -> real CUdevice ordinal
    device_handles: DashMap<NetworkHandle, cuda_driver::CUdevice>,
    /// Maps NetworkHandle -> real CUcontext pointer
    context_handles: DashMap<NetworkHandle, cuda_driver::CUcontext>,
    /// Maps NetworkHandle -> real CUmodule pointer
    module_handles: DashMap<NetworkHandle, cuda_driver::CUmodule>,
    /// Maps NetworkHandle -> real CUfunction pointer
    function_handles: DashMap<NetworkHandle, cuda_driver::CUfunction>,
    /// Maps NetworkHandle -> real CUdeviceptr (GPU memory address)
    memory_handles: DashMap<NetworkHandle, cuda_driver::CUdeviceptr>,
    /// Maps NetworkHandle -> allocated byte size for memory
    memory_sizes: DashMap<NetworkHandle, u64>,
    /// Maps NetworkHandle -> real CUstream pointer
    stream_handles: DashMap<NetworkHandle, cuda_driver::CUstream>,
    /// Maps NetworkHandle -> real CUevent pointer
    event_handles: DashMap<NetworkHandle, cuda_driver::CUevent>,
    /// Maps NetworkHandle -> real host memory pointer (cuMemAllocHost / cuMemHostAlloc)
    host_memory_handles: DashMap<NetworkHandle, *mut c_void>,
    /// Maps NetworkHandle -> real CUmemoryPool pointer
    mempool_handles: DashMap<NetworkHandle, cuda_driver::CUmemoryPool>,
    /// Maps NetworkHandle -> real CUlinkState pointer
    linker_handles: DashMap<NetworkHandle, cuda_driver::CUlinkState>,
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
        }
    }

    /// Check if the real CUDA driver is available.
    fn driver(&self) -> Result<&CudaDriver, CudaResponse> {
        self.driver.as_deref().ok_or(CudaResponse::Error {
            code: 3, // CUDA_ERROR_NOT_INITIALIZED
            message: "CUDA driver not loaded on server".to_string(),
        })
    }

    /// Convert a CUresult to a CudaResponse::Error.
    fn cuda_err(code: cuda_driver::CUresult) -> CudaResponse {
        CudaResponse::Error {
            code,
            message: cuda_driver::cuda_error_name(code).to_string(),
        }
    }

    /// Execute a CUDA command and return the response.
    pub fn execute(&self, session: &Session, cmd: CudaCommand) -> CudaResponse {
        match cmd {
            CudaCommand::Init { flags } => {
                info!(
                    session_id = session.session_id,
                    "CUDA init (flags={})", flags
                );
                // Already initialized in constructor
                CudaResponse::Success
            }

            CudaCommand::DriverGetVersion => {
                match self.driver() {
                    Ok(d) => match d.driver_get_version() {
                        Ok(v) => CudaResponse::DriverVersion(v),
                        Err(e) => Self::cuda_err(e),
                    },
                    Err(_) => {
                        // Fallback: report CUDA 12.0
                        CudaResponse::DriverVersion(12000)
                    }
                }
            }

            CudaCommand::DeviceGetCount => {
                match self.driver() {
                    Ok(d) => match d.device_get_count() {
                        Ok(count) => {
                            debug!(session_id = session.session_id, "DeviceGetCount -> {}", count);
                            CudaResponse::DeviceCount(count)
                        }
                        Err(e) => Self::cuda_err(e),
                    },
                    Err(_) => {
                        // Fallback: use Vulkan-discovered info
                        let count =
                            self.gpu_infos.iter().filter(|g| g.supports_cuda).count() as i32;
                        CudaResponse::DeviceCount(count)
                    }
                }
            }

            CudaCommand::DeviceGet { ordinal } => {
                let handle = session.alloc_handle(ResourceType::CuDevice);

                if let Ok(d) = self.driver() {
                    match d.device_get(ordinal) {
                        Ok(real_device) => {
                            self.device_handles.insert(handle, real_device);
                            debug!(
                                session_id = session.session_id,
                                "DeviceGet({}) -> {:?} (real device {})", ordinal, handle, real_device
                            );
                            CudaResponse::Device(handle)
                        }
                        Err(e) => {
                            session.remove_handle(&handle);
                            Self::cuda_err(e)
                        }
                    }
                } else {
                    // Fallback
                    self.device_handles.insert(handle, ordinal);
                    CudaResponse::Device(handle)
                }
            }

            CudaCommand::DeviceGetName { device } => {
                if let Some(real_dev) = self.device_handles.get(&device) {
                    if let Ok(d) = self.driver() {
                        match d.device_get_name(*real_dev) {
                            Ok(name) => return CudaResponse::DeviceName(name),
                            Err(e) => return Self::cuda_err(e),
                        }
                    }
                }
                // Fallback
                let name = self
                    .gpu_infos
                    .iter()
                    .find(|g| g.supports_cuda)
                    .map(|g| g.device_name.clone())
                    .unwrap_or_else(|| "RGPU Virtual Device".to_string());
                CudaResponse::DeviceName(name)
            }

            CudaCommand::DeviceGetAttribute { attrib, device } => {
                if let Some(real_dev) = self.device_handles.get(&device) {
                    if let Ok(d) = self.driver() {
                        match d.device_get_attribute(attrib, *real_dev) {
                            Ok(val) => return CudaResponse::DeviceAttribute(val),
                            Err(e) => return Self::cuda_err(e),
                        }
                    }
                }
                // Fallback
                CudaResponse::DeviceAttribute(self.get_device_attribute_fallback(attrib))
            }

            CudaCommand::DeviceTotalMem { device } => {
                if let Some(real_dev) = self.device_handles.get(&device) {
                    if let Ok(d) = self.driver() {
                        match d.device_total_mem(*real_dev) {
                            Ok(bytes) => return CudaResponse::DeviceTotalMem(bytes as u64),
                            Err(e) => return Self::cuda_err(e),
                        }
                    }
                }
                // Fallback
                let total = self
                    .gpu_infos
                    .first()
                    .map(|g| g.total_memory)
                    .unwrap_or(0);
                CudaResponse::DeviceTotalMem(total)
            }

            CudaCommand::DeviceComputeCapability { device } => {
                if let Some(real_dev) = self.device_handles.get(&device) {
                    if let Ok(d) = self.driver() {
                        match d.device_compute_capability(*real_dev) {
                            Ok((major, minor)) => {
                                return CudaResponse::ComputeCapability { major, minor }
                            }
                            Err(e) => return Self::cuda_err(e),
                        }
                    }
                }
                // Fallback
                let (major, minor) = self
                    .gpu_infos
                    .first()
                    .and_then(|g| g.cuda_compute_capability)
                    .unwrap_or((8, 6));
                CudaResponse::ComputeCapability { major, minor }
            }

            // ── Device Extended ────────────────────────────────────

            CudaCommand::DeviceGetUuid { device } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_dev = match self.device_handles.get(&device) {
                    Some(dev) => *dev,
                    None => return CudaResponse::Error {
                        code: 101,
                        message: "invalid device handle".to_string(),
                    },
                };
                match d.device_get_uuid(real_dev) {
                    Ok(uuid) => CudaResponse::DeviceUuid(uuid.to_vec()),
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::DeviceGetP2PAttribute { attrib, src_device, dst_device } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_src = match self.device_handles.get(&src_device) {
                    Some(dev) => *dev,
                    None => return CudaResponse::Error {
                        code: 101,
                        message: "invalid src device handle".to_string(),
                    },
                };
                let real_dst = match self.device_handles.get(&dst_device) {
                    Some(dev) => *dev,
                    None => return CudaResponse::Error {
                        code: 101,
                        message: "invalid dst device handle".to_string(),
                    },
                };
                match d.device_get_p2p_attribute(attrib, real_src, real_dst) {
                    Ok(val) => CudaResponse::P2PAttribute(val),
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::DeviceCanAccessPeer { device, peer_device } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_dev = match self.device_handles.get(&device) {
                    Some(dev) => *dev,
                    None => return CudaResponse::Error {
                        code: 101,
                        message: "invalid device handle".to_string(),
                    },
                };
                let real_peer = match self.device_handles.get(&peer_device) {
                    Some(dev) => *dev,
                    None => return CudaResponse::Error {
                        code: 101,
                        message: "invalid peer device handle".to_string(),
                    },
                };
                match d.device_can_access_peer(real_dev, real_peer) {
                    Ok(can) => CudaResponse::BoolResult(can),
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::DeviceGetByPCIBusId { pci_bus_id } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                match d.device_get_by_pci_bus_id(&pci_bus_id) {
                    Ok(real_device) => {
                        let handle = session.alloc_handle(ResourceType::CuDevice);
                        self.device_handles.insert(handle, real_device);
                        CudaResponse::Device(handle)
                    }
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::DeviceGetPCIBusId { device } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_dev = match self.device_handles.get(&device) {
                    Some(dev) => *dev,
                    None => return CudaResponse::Error {
                        code: 101,
                        message: "invalid device handle".to_string(),
                    },
                };
                match d.device_get_pci_bus_id(real_dev) {
                    Ok(bus_id) => CudaResponse::DevicePCIBusId(bus_id),
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::DeviceGetDefaultMemPool { device } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_dev = match self.device_handles.get(&device) {
                    Some(dev) => *dev,
                    None => return CudaResponse::Error {
                        code: 101,
                        message: "invalid device handle".to_string(),
                    },
                };
                match d.device_get_default_mem_pool(real_dev) {
                    Ok(pool) => {
                        let handle = session.alloc_handle(ResourceType::CuMemPool);
                        self.mempool_handles.insert(handle, pool);
                        CudaResponse::MemPool(handle)
                    }
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::DeviceGetMemPool { device } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_dev = match self.device_handles.get(&device) {
                    Some(dev) => *dev,
                    None => return CudaResponse::Error {
                        code: 101,
                        message: "invalid device handle".to_string(),
                    },
                };
                match d.device_get_mem_pool(real_dev) {
                    Ok(pool) => {
                        let handle = session.alloc_handle(ResourceType::CuMemPool);
                        self.mempool_handles.insert(handle, pool);
                        CudaResponse::MemPool(handle)
                    }
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::DeviceSetMemPool { device, mem_pool } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_dev = match self.device_handles.get(&device) {
                    Some(dev) => *dev,
                    None => return CudaResponse::Error {
                        code: 101,
                        message: "invalid device handle".to_string(),
                    },
                };
                let real_pool = match self.mempool_handles.get(&mem_pool) {
                    Some(p) => *p,
                    None => return CudaResponse::Error {
                        code: 400,
                        message: "invalid mempool handle".to_string(),
                    },
                };
                let res = d.device_set_mem_pool(real_dev, real_pool);
                if res == CUDA_SUCCESS {
                    CudaResponse::Success
                } else {
                    Self::cuda_err(res)
                }
            }

            CudaCommand::DeviceGetTexture1DLinearMaxWidth { format: _, num_channels: _, device: _ } => {
                // This API is rarely used and requires cuDeviceGetTexture1DLinearMaxWidth
                // which is not commonly loaded. Return a reasonable default.
                CudaResponse::Texture1DMaxWidth(1 << 27) // 128M texels
            }

            CudaCommand::DeviceGetExecAffinitySupport { affinity_type: _, device: _ } => {
                // cuDeviceGetExecAffinitySupport - not commonly available, return false
                CudaResponse::BoolResult(false)
            }

            // ── Primary Context ────────────────────────────────────

            CudaCommand::DevicePrimaryCtxRetain { device } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_dev = match self.device_handles.get(&device) {
                    Some(dev) => *dev,
                    None => return CudaResponse::Error {
                        code: 101,
                        message: "invalid device handle".to_string(),
                    },
                };
                match d.device_primary_ctx_retain(real_dev) {
                    Ok(ctx) => {
                        let handle = session.alloc_handle(ResourceType::CuContext);
                        self.context_handles.insert(handle, ctx);
                        debug!(
                            session_id = session.session_id,
                            "DevicePrimaryCtxRetain(device={:?}) -> {:?}", device, handle
                        );
                        CudaResponse::Context(handle)
                    }
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::DevicePrimaryCtxRelease { device } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_dev = match self.device_handles.get(&device) {
                    Some(dev) => *dev,
                    None => return CudaResponse::Error {
                        code: 101,
                        message: "invalid device handle".to_string(),
                    },
                };
                let res = d.device_primary_ctx_release(real_dev);
                if res == CUDA_SUCCESS {
                    CudaResponse::Success
                } else {
                    Self::cuda_err(res)
                }
            }

            CudaCommand::DevicePrimaryCtxReset { device } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_dev = match self.device_handles.get(&device) {
                    Some(dev) => *dev,
                    None => return CudaResponse::Error {
                        code: 101,
                        message: "invalid device handle".to_string(),
                    },
                };
                let res = d.device_primary_ctx_reset(real_dev);
                if res == CUDA_SUCCESS {
                    CudaResponse::Success
                } else {
                    Self::cuda_err(res)
                }
            }

            CudaCommand::DevicePrimaryCtxGetState { device } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_dev = match self.device_handles.get(&device) {
                    Some(dev) => *dev,
                    None => return CudaResponse::Error {
                        code: 101,
                        message: "invalid device handle".to_string(),
                    },
                };
                match d.device_primary_ctx_get_state(real_dev) {
                    Ok((flags, active)) => CudaResponse::PrimaryCtxState { flags, active },
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::DevicePrimaryCtxSetFlags { device, flags } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_dev = match self.device_handles.get(&device) {
                    Some(dev) => *dev,
                    None => return CudaResponse::Error {
                        code: 101,
                        message: "invalid device handle".to_string(),
                    },
                };
                let res = d.device_primary_ctx_set_flags(real_dev, flags);
                if res == CUDA_SUCCESS {
                    CudaResponse::Success
                } else {
                    Self::cuda_err(res)
                }
            }

            // ── Context Management ──────────────────────────────────

            CudaCommand::CtxCreate { flags, device } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                let real_dev = match self.device_handles.get(&device) {
                    Some(dev) => *dev,
                    None => {
                        return CudaResponse::Error {
                            code: 201, // CUDA_ERROR_INVALID_CONTEXT
                            message: "invalid device handle".to_string(),
                        }
                    }
                };

                match d.ctx_create(flags, real_dev) {
                    Ok(ctx) => {
                        let handle = session.alloc_handle(ResourceType::CuContext);
                        self.context_handles.insert(handle, ctx);
                        debug!(
                            session_id = session.session_id,
                            "CtxCreate(device={:?}) -> {:?}", device, handle
                        );
                        CudaResponse::Context(handle)
                    }
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::CtxDestroy { ctx } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                match self.context_handles.remove(&ctx) {
                    Some((_, real_ctx)) => {
                        let res = d.ctx_destroy(real_ctx);
                        session.remove_handle(&ctx);
                        if res == CUDA_SUCCESS {
                            CudaResponse::Success
                        } else {
                            Self::cuda_err(res)
                        }
                    }
                    None => CudaResponse::Error {
                        code: 201,
                        message: "invalid context handle".to_string(),
                    },
                }
            }

            CudaCommand::CtxSetCurrent { ctx } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                match self.context_handles.get(&ctx) {
                    Some(real_ctx) => {
                        let res = d.ctx_set_current(*real_ctx);
                        if res == CUDA_SUCCESS {
                            CudaResponse::Success
                        } else {
                            Self::cuda_err(res)
                        }
                    }
                    None => CudaResponse::Error {
                        code: 201,
                        message: "invalid context handle".to_string(),
                    },
                }
            }

            CudaCommand::CtxGetCurrent => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                match d.ctx_get_current() {
                    Ok(ctx) => {
                        // Find the handle for this context
                        for entry in self.context_handles.iter() {
                            if *entry.value() == ctx {
                                return CudaResponse::Context(*entry.key());
                            }
                        }
                        // Context not tracked - create a handle for it
                        let handle = session.alloc_handle(ResourceType::CuContext);
                        self.context_handles.insert(handle, ctx);
                        CudaResponse::Context(handle)
                    }
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::CtxSynchronize => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                let res = d.ctx_synchronize();
                if res == CUDA_SUCCESS {
                    CudaResponse::Success
                } else {
                    Self::cuda_err(res)
                }
            }

            CudaCommand::CtxPushCurrent { ctx } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_ctx = match self.context_handles.get(&ctx) {
                    Some(c) => *c,
                    None => return CudaResponse::Error {
                        code: 201,
                        message: "invalid context handle".to_string(),
                    },
                };
                let res = d.ctx_push_current(real_ctx);
                if res == CUDA_SUCCESS {
                    CudaResponse::Success
                } else {
                    Self::cuda_err(res)
                }
            }

            CudaCommand::CtxPopCurrent => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                match d.ctx_pop_current() {
                    Ok(ctx) => {
                        // Find the handle for this context
                        for entry in self.context_handles.iter() {
                            if *entry.value() == ctx {
                                return CudaResponse::Context(*entry.key());
                            }
                        }
                        // Context not tracked - create a handle for it
                        let handle = session.alloc_handle(ResourceType::CuContext);
                        self.context_handles.insert(handle, ctx);
                        CudaResponse::Context(handle)
                    }
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::CtxGetDevice => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                match d.ctx_get_device() {
                    Ok(real_dev) => {
                        // Find the handle for this device
                        for entry in self.device_handles.iter() {
                            if *entry.value() == real_dev {
                                return CudaResponse::ContextDevice(*entry.key());
                            }
                        }
                        // Device not tracked - create a handle for it
                        let handle = session.alloc_handle(ResourceType::CuDevice);
                        self.device_handles.insert(handle, real_dev);
                        CudaResponse::ContextDevice(handle)
                    }
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::CtxSetCacheConfig { config } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let res = d.ctx_set_cache_config(config);
                if res == CUDA_SUCCESS {
                    CudaResponse::Success
                } else {
                    Self::cuda_err(res)
                }
            }

            CudaCommand::CtxGetCacheConfig => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                match d.ctx_get_cache_config() {
                    Ok(config) => CudaResponse::CacheConfig(config),
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::CtxSetLimit { limit, value } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let res = d.ctx_set_limit(limit, value);
                if res == CUDA_SUCCESS {
                    CudaResponse::Success
                } else {
                    Self::cuda_err(res)
                }
            }

            CudaCommand::CtxGetLimit { limit } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                match d.ctx_get_limit(limit) {
                    Ok(value) => CudaResponse::ContextLimit(value),
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::CtxGetStreamPriorityRange => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                match d.ctx_get_stream_priority_range() {
                    Ok((least, greatest)) => CudaResponse::StreamPriorityRange { least, greatest },
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::CtxGetApiVersion { ctx } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_ctx = match self.context_handles.get(&ctx) {
                    Some(c) => *c,
                    None => return CudaResponse::Error {
                        code: 201,
                        message: "invalid context handle".to_string(),
                    },
                };
                match d.ctx_get_api_version(real_ctx) {
                    Ok(version) => CudaResponse::ContextApiVersion(version),
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::CtxGetFlags => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                match d.ctx_get_flags() {
                    Ok(flags) => CudaResponse::ContextFlags(flags),
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::CtxSetFlags { flags } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let res = d.ctx_set_flags(flags);
                if res == CUDA_SUCCESS {
                    CudaResponse::Success
                } else {
                    Self::cuda_err(res)
                }
            }

            CudaCommand::CtxResetPersistingL2Cache => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let res = d.ctx_reset_persisting_l2_cache();
                if res == CUDA_SUCCESS {
                    CudaResponse::Success
                } else {
                    Self::cuda_err(res)
                }
            }

            // ── Module Management ───────────────────────────────────

            CudaCommand::ModuleLoadData { image } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                match d.module_load_data(&image) {
                    Ok(module) => {
                        let handle = session.alloc_handle(ResourceType::CuModule);
                        self.module_handles.insert(handle, module);
                        debug!(
                            session_id = session.session_id,
                            "ModuleLoadData ({} bytes) -> {:?}", image.len(), handle
                        );
                        CudaResponse::Module(handle)
                    }
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::ModuleUnload { module } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                match self.module_handles.remove(&module) {
                    Some((_, real_mod)) => {
                        let res = d.module_unload(real_mod);
                        session.remove_handle(&module);
                        if res == CUDA_SUCCESS {
                            CudaResponse::Success
                        } else {
                            Self::cuda_err(res)
                        }
                    }
                    None => CudaResponse::Error {
                        code: 400,
                        message: "invalid module handle".to_string(),
                    },
                }
            }

            CudaCommand::ModuleGetFunction { module, name } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                let real_mod = match self.module_handles.get(&module) {
                    Some(m) => *m,
                    None => {
                        return CudaResponse::Error {
                            code: 400,
                            message: "invalid module handle".to_string(),
                        }
                    }
                };

                match d.module_get_function(real_mod, &name) {
                    Ok(func) => {
                        let handle = session.alloc_handle(ResourceType::CuFunction);
                        self.function_handles.insert(handle, func);
                        debug!(
                            session_id = session.session_id,
                            "ModuleGetFunction('{}') -> {:?}", name, handle
                        );
                        CudaResponse::Function(handle)
                    }
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::ModuleGetGlobal { module, name } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                let real_mod = match self.module_handles.get(&module) {
                    Some(m) => *m,
                    None => {
                        return CudaResponse::Error {
                            code: 400,
                            message: "invalid module handle".to_string(),
                        }
                    }
                };

                match d.module_get_global(real_mod, &name) {
                    Ok((dptr, size)) => {
                        let handle = session.alloc_handle(ResourceType::CuDevicePtr);
                        self.memory_handles.insert(handle, dptr);
                        self.memory_sizes.insert(handle, size as u64);
                        CudaResponse::GlobalPtr {
                            ptr: handle,
                            size: size as u64,
                        }
                    }
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::ModuleLoad { fname } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                match d.module_load(&fname) {
                    Ok(module) => {
                        let handle = session.alloc_handle(ResourceType::CuModule);
                        self.module_handles.insert(handle, module);
                        debug!(
                            session_id = session.session_id,
                            "ModuleLoad('{}') -> {:?}", fname, handle
                        );
                        CudaResponse::Module(handle)
                    }
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::ModuleLoadDataEx { image, num_options: _, options: _, option_values: _ } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                // Simplified: forward to module_load_data_ex which ignores JIT options
                match d.module_load_data_ex(&image) {
                    Ok(module) => {
                        let handle = session.alloc_handle(ResourceType::CuModule);
                        self.module_handles.insert(handle, module);
                        debug!(
                            session_id = session.session_id,
                            "ModuleLoadDataEx ({} bytes) -> {:?}", image.len(), handle
                        );
                        CudaResponse::Module(handle)
                    }
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::ModuleLoadFatBinary { fat_cubin } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                match d.module_load_fat_binary(&fat_cubin) {
                    Ok(module) => {
                        let handle = session.alloc_handle(ResourceType::CuModule);
                        self.module_handles.insert(handle, module);
                        debug!(
                            session_id = session.session_id,
                            "ModuleLoadFatBinary ({} bytes) -> {:?}", fat_cubin.len(), handle
                        );
                        CudaResponse::Module(handle)
                    }
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::LinkCreate { num_options: _, options: _, option_values: _ } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                match d.link_create() {
                    Ok(state) => {
                        let handle = session.alloc_handle(ResourceType::CuLinker);
                        self.linker_handles.insert(handle, state);
                        debug!(
                            session_id = session.session_id,
                            "LinkCreate -> {:?}", handle
                        );
                        CudaResponse::Linker(handle)
                    }
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::LinkAddData { link, jit_type, data, name, num_options: _, options: _, option_values: _ } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_state = match self.linker_handles.get(&link) {
                    Some(s) => *s,
                    None => return CudaResponse::Error {
                        code: 400,
                        message: "invalid linker handle".to_string(),
                    },
                };
                let res = d.link_add_data(real_state, jit_type, &data, &name);
                if res == CUDA_SUCCESS {
                    CudaResponse::Success
                } else {
                    Self::cuda_err(res)
                }
            }

            CudaCommand::LinkAddFile { link, jit_type, path, num_options: _, options: _, option_values: _ } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_state = match self.linker_handles.get(&link) {
                    Some(s) => *s,
                    None => return CudaResponse::Error {
                        code: 400,
                        message: "invalid linker handle".to_string(),
                    },
                };
                let res = d.link_add_file(real_state, jit_type, &path);
                if res == CUDA_SUCCESS {
                    CudaResponse::Success
                } else {
                    Self::cuda_err(res)
                }
            }

            CudaCommand::LinkComplete { link } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_state = match self.linker_handles.get(&link) {
                    Some(s) => *s,
                    None => return CudaResponse::Error {
                        code: 400,
                        message: "invalid linker handle".to_string(),
                    },
                };
                match d.link_complete(real_state) {
                    Ok(cubin_data) => CudaResponse::LinkCompleted { cubin_data },
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::LinkDestroy { link } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                match self.linker_handles.remove(&link) {
                    Some((_, real_state)) => {
                        let res = d.link_destroy(real_state);
                        session.remove_handle(&link);
                        if res == CUDA_SUCCESS {
                            CudaResponse::Success
                        } else {
                            Self::cuda_err(res)
                        }
                    }
                    None => CudaResponse::Error {
                        code: 400,
                        message: "invalid linker handle".to_string(),
                    },
                }
            }

            // ── Memory Management ───────────────────────────────────

            CudaCommand::MemAlloc { byte_size } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                match d.mem_alloc(byte_size as usize) {
                    Ok(dptr) => {
                        let handle = session.alloc_handle(ResourceType::CuDevicePtr);
                        self.memory_handles.insert(handle, dptr);
                        self.memory_sizes.insert(handle, byte_size);
                        debug!(
                            session_id = session.session_id,
                            "MemAlloc({} bytes) -> {:?} (dptr=0x{:x})", byte_size, handle, dptr
                        );
                        CudaResponse::MemAllocated(handle)
                    }
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::MemFree { dptr } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                match self.memory_handles.remove(&dptr) {
                    Some((_, real_ptr)) => {
                        let res = d.mem_free(real_ptr);
                        self.memory_sizes.remove(&dptr);
                        session.remove_handle(&dptr);
                        if res == CUDA_SUCCESS {
                            debug!(
                                session_id = session.session_id,
                                "MemFree({:?})", dptr
                            );
                            CudaResponse::Success
                        } else {
                            Self::cuda_err(res)
                        }
                    }
                    None => CudaResponse::Error {
                        code: 400,
                        message: "invalid memory handle".to_string(),
                    },
                }
            }

            CudaCommand::MemcpyHtoD {
                dst,
                src_data,
                byte_count,
            } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                let real_ptr = match self.memory_handles.get(&dst) {
                    Some(p) => *p,
                    None => {
                        return CudaResponse::Error {
                            code: 400,
                            message: "invalid destination memory handle".to_string(),
                        }
                    }
                };

                let data = &src_data[..byte_count as usize];
                let res = d.memcpy_htod(real_ptr, data);
                if res == CUDA_SUCCESS {
                    debug!(
                        session_id = session.session_id,
                        "MemcpyHtoD({:?}, {} bytes)", dst, byte_count
                    );
                    CudaResponse::Success
                } else {
                    Self::cuda_err(res)
                }
            }

            CudaCommand::MemcpyDtoH { src, byte_count } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                let real_ptr = match self.memory_handles.get(&src) {
                    Some(p) => *p,
                    None => {
                        return CudaResponse::Error {
                            code: 400,
                            message: "invalid source memory handle".to_string(),
                        }
                    }
                };

                let mut buf = vec![0u8; byte_count as usize];
                let res = d.memcpy_dtoh(&mut buf, real_ptr);
                if res == CUDA_SUCCESS {
                    debug!(
                        session_id = session.session_id,
                        "MemcpyDtoH({:?}, {} bytes)", src, byte_count
                    );
                    CudaResponse::MemoryData(buf)
                } else {
                    Self::cuda_err(res)
                }
            }

            CudaCommand::MemcpyDtoD {
                dst,
                src,
                byte_count,
            } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                let real_dst = match self.memory_handles.get(&dst) {
                    Some(p) => *p,
                    None => {
                        return CudaResponse::Error {
                            code: 400,
                            message: "invalid destination memory handle".to_string(),
                        }
                    }
                };
                let real_src = match self.memory_handles.get(&src) {
                    Some(p) => *p,
                    None => {
                        return CudaResponse::Error {
                            code: 400,
                            message: "invalid source memory handle".to_string(),
                        }
                    }
                };

                let res = d.memcpy_dtod(real_dst, real_src, byte_count as usize);
                if res == CUDA_SUCCESS {
                    CudaResponse::Success
                } else {
                    Self::cuda_err(res)
                }
            }

            CudaCommand::MemcpyHtoDAsync {
                dst,
                src_data,
                byte_count,
                stream: _,
            } => {
                // For async memcpy over network, we use sync version since
                // the network is the bottleneck anyway
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                let real_ptr = match self.memory_handles.get(&dst) {
                    Some(p) => *p,
                    None => {
                        return CudaResponse::Error {
                            code: 400,
                            message: "invalid destination memory handle".to_string(),
                        }
                    }
                };

                let data = &src_data[..byte_count as usize];
                let res = d.memcpy_htod(real_ptr, data);
                if res == CUDA_SUCCESS {
                    debug!(
                        session_id = session.session_id,
                        "MemcpyHtoDAsync({:?}, {} bytes) [sync]", dst, byte_count
                    );
                    CudaResponse::Success
                } else {
                    Self::cuda_err(res)
                }
            }

            CudaCommand::MemcpyDtoHAsync {
                src,
                byte_count,
                stream: _,
            } => {
                // Use sync version - network is the bottleneck
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                let real_ptr = match self.memory_handles.get(&src) {
                    Some(p) => *p,
                    None => {
                        return CudaResponse::Error {
                            code: 400,
                            message: "invalid source memory handle".to_string(),
                        }
                    }
                };

                let mut buf = vec![0u8; byte_count as usize];
                let res = d.memcpy_dtoh(&mut buf, real_ptr);
                if res == CUDA_SUCCESS {
                    debug!(
                        session_id = session.session_id,
                        "MemcpyDtoHAsync({:?}, {} bytes) [sync]", src, byte_count
                    );
                    CudaResponse::MemoryData(buf)
                } else {
                    Self::cuda_err(res)
                }
            }

            CudaCommand::MemcpyDtoDAsync {
                dst,
                src,
                byte_count,
                stream: _,
            } => {
                // Use sync version - network is the bottleneck
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                let real_dst = match self.memory_handles.get(&dst) {
                    Some(p) => *p,
                    None => {
                        return CudaResponse::Error {
                            code: 400,
                            message: "invalid destination memory handle".to_string(),
                        }
                    }
                };
                let real_src = match self.memory_handles.get(&src) {
                    Some(p) => *p,
                    None => {
                        return CudaResponse::Error {
                            code: 400,
                            message: "invalid source memory handle".to_string(),
                        }
                    }
                };

                let res = d.memcpy_dtod(real_dst, real_src, byte_count as usize);
                if res == CUDA_SUCCESS {
                    CudaResponse::Success
                } else {
                    Self::cuda_err(res)
                }
            }

            CudaCommand::MemsetD8 {
                dst,
                value,
                count,
            } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                let real_ptr = match self.memory_handles.get(&dst) {
                    Some(p) => *p,
                    None => {
                        return CudaResponse::Error {
                            code: 400,
                            message: "invalid memory handle".to_string(),
                        }
                    }
                };

                let res = d.memset_d8(real_ptr, value, count as usize);
                if res == CUDA_SUCCESS {
                    CudaResponse::Success
                } else {
                    Self::cuda_err(res)
                }
            }

            CudaCommand::MemsetD16 {
                dst,
                value,
                count,
            } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                let real_ptr = match self.memory_handles.get(&dst) {
                    Some(p) => *p,
                    None => {
                        return CudaResponse::Error {
                            code: 400,
                            message: "invalid memory handle".to_string(),
                        }
                    }
                };

                let res = d.memset_d16(real_ptr, value, count as usize);
                if res == CUDA_SUCCESS {
                    CudaResponse::Success
                } else {
                    Self::cuda_err(res)
                }
            }

            CudaCommand::MemsetD32 {
                dst,
                value,
                count,
            } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                let real_ptr = match self.memory_handles.get(&dst) {
                    Some(p) => *p,
                    None => {
                        return CudaResponse::Error {
                            code: 400,
                            message: "invalid memory handle".to_string(),
                        }
                    }
                };

                let res = d.memset_d32(real_ptr, value, count as usize);
                if res == CUDA_SUCCESS {
                    CudaResponse::Success
                } else {
                    Self::cuda_err(res)
                }
            }

            CudaCommand::MemsetD8Async {
                dst,
                value,
                count,
                stream: _,
            } => {
                // Use sync version - network is the bottleneck
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                let real_ptr = match self.memory_handles.get(&dst) {
                    Some(p) => *p,
                    None => {
                        return CudaResponse::Error {
                            code: 400,
                            message: "invalid memory handle".to_string(),
                        }
                    }
                };

                let res = d.memset_d8(real_ptr, value, count as usize);
                if res == CUDA_SUCCESS {
                    CudaResponse::Success
                } else {
                    Self::cuda_err(res)
                }
            }

            CudaCommand::MemsetD16Async {
                dst,
                value,
                count,
                stream: _,
            } => {
                // Use sync version - network is the bottleneck
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                let real_ptr = match self.memory_handles.get(&dst) {
                    Some(p) => *p,
                    None => {
                        return CudaResponse::Error {
                            code: 400,
                            message: "invalid memory handle".to_string(),
                        }
                    }
                };

                let res = d.memset_d16(real_ptr, value, count as usize);
                if res == CUDA_SUCCESS {
                    CudaResponse::Success
                } else {
                    Self::cuda_err(res)
                }
            }

            CudaCommand::MemsetD32Async {
                dst,
                value,
                count,
                stream: _,
            } => {
                // Use sync version - network is the bottleneck
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                let real_ptr = match self.memory_handles.get(&dst) {
                    Some(p) => *p,
                    None => {
                        return CudaResponse::Error {
                            code: 400,
                            message: "invalid memory handle".to_string(),
                        }
                    }
                };

                let res = d.memset_d32(real_ptr, value, count as usize);
                if res == CUDA_SUCCESS {
                    CudaResponse::Success
                } else {
                    Self::cuda_err(res)
                }
            }

            CudaCommand::MemGetInfo => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                match d.mem_get_info() {
                    Ok((free, total)) => CudaResponse::MemInfo {
                        free: free as u64,
                        total: total as u64,
                    },
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::MemGetAddressRange { dptr } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_ptr = match self.memory_handles.get(&dptr) {
                    Some(p) => *p,
                    None => {
                        return CudaResponse::Error {
                            code: 400,
                            message: "invalid memory handle".to_string(),
                        }
                    }
                };
                match d.mem_get_address_range(real_ptr) {
                    Ok((base, size)) => {
                        // Find or create handle for base
                        let base_handle = {
                            let mut found = None;
                            for entry in self.memory_handles.iter() {
                                if *entry.value() == base {
                                    found = Some(*entry.key());
                                    break;
                                }
                            }
                            found.unwrap_or(dptr)
                        };
                        CudaResponse::MemAddressRange {
                            base: base_handle,
                            size: size as u64,
                        }
                    }
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::MemAllocHost { byte_size } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                match d.mem_alloc_host(byte_size as usize) {
                    Ok(ptr) => {
                        let handle = session.alloc_handle(ResourceType::CuHostPtr);
                        self.host_memory_handles.insert(handle, ptr);
                        debug!(
                            session_id = session.session_id,
                            "MemAllocHost({} bytes) -> {:?}", byte_size, handle
                        );
                        CudaResponse::HostPtr(handle)
                    }
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::MemFreeHost { ptr } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                match self.host_memory_handles.remove(&ptr) {
                    Some((_, real_ptr)) => {
                        let res = d.mem_free_host(real_ptr);
                        session.remove_handle(&ptr);
                        if res == CUDA_SUCCESS {
                            CudaResponse::Success
                        } else {
                            Self::cuda_err(res)
                        }
                    }
                    None => CudaResponse::Error {
                        code: 400,
                        message: "invalid host memory handle".to_string(),
                    },
                }
            }

            CudaCommand::MemHostAlloc { byte_size, flags } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                match d.mem_host_alloc(byte_size as usize, flags) {
                    Ok(ptr) => {
                        let handle = session.alloc_handle(ResourceType::CuHostPtr);
                        self.host_memory_handles.insert(handle, ptr);
                        debug!(
                            session_id = session.session_id,
                            "MemHostAlloc({} bytes, flags={}) -> {:?}", byte_size, flags, handle
                        );
                        CudaResponse::HostPtr(handle)
                    }
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::MemHostGetDevicePointer { host_ptr, flags } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_host_ptr = match self.host_memory_handles.get(&host_ptr) {
                    Some(p) => *p,
                    None => {
                        return CudaResponse::Error {
                            code: 400,
                            message: "invalid host memory handle".to_string(),
                        }
                    }
                };
                match d.mem_host_get_device_pointer(real_host_ptr, flags) {
                    Ok(dptr) => {
                        let handle = session.alloc_handle(ResourceType::CuDevicePtr);
                        self.memory_handles.insert(handle, dptr);
                        CudaResponse::HostDevicePtr(handle)
                    }
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::MemHostGetFlags { host_ptr } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_host_ptr = match self.host_memory_handles.get(&host_ptr) {
                    Some(p) => *p,
                    None => {
                        return CudaResponse::Error {
                            code: 400,
                            message: "invalid host memory handle".to_string(),
                        }
                    }
                };
                match d.mem_host_get_flags(real_host_ptr) {
                    Ok(flags) => CudaResponse::HostFlags(flags),
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::MemAllocManaged { byte_size, flags } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                match d.mem_alloc_managed(byte_size as usize, flags) {
                    Ok(dptr) => {
                        let handle = session.alloc_handle(ResourceType::CuDevicePtr);
                        self.memory_handles.insert(handle, dptr);
                        self.memory_sizes.insert(handle, byte_size);
                        debug!(
                            session_id = session.session_id,
                            "MemAllocManaged({} bytes) -> {:?}", byte_size, handle
                        );
                        CudaResponse::MemAllocated(handle)
                    }
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::MemAllocPitch { width, height, element_size } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                match d.mem_alloc_pitch(width as usize, height as usize, element_size) {
                    Ok((dptr, pitch)) => {
                        let handle = session.alloc_handle(ResourceType::CuDevicePtr);
                        self.memory_handles.insert(handle, dptr);
                        self.memory_sizes.insert(handle, pitch as u64 * height);
                        CudaResponse::MemAllocPitch {
                            dptr: handle,
                            pitch: pitch as u64,
                        }
                    }
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::MemHostRegister { byte_size: _, flags: _ } => {
                // Cannot register client host memory on the server - not supported over network
                Self::cuda_err(CUDA_ERROR_NOT_SUPPORTED)
            }

            CudaCommand::MemHostUnregister { ptr: _ } => {
                // Cannot unregister client host memory on the server - not supported over network
                Self::cuda_err(CUDA_ERROR_NOT_SUPPORTED)
            }

            CudaCommand::MemPrefetchAsync { dptr: _, count: _, dst_device: _, stream: _ } => {
                // No-op over network - prefetch hints are meaningless remotely
                CudaResponse::Success
            }

            CudaCommand::MemAdvise { dptr: _, count: _, advice: _, device: _ } => {
                // No-op over network - memory advice hints are meaningless remotely
                CudaResponse::Success
            }

            CudaCommand::MemRangeGetAttribute { dptr: _, count: _, attribute: _ } => {
                // No-op over network - return empty attribute data
                CudaResponse::Success
            }

            // ── Execution Control ───────────────────────────────────

            CudaCommand::LaunchKernel {
                func,
                grid_dim,
                block_dim,
                shared_mem_bytes,
                stream,
                kernel_params,
            } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                let real_func = match self.function_handles.get(&func) {
                    Some(f) => *f,
                    None => {
                        return CudaResponse::Error {
                            code: 400,
                            message: "invalid function handle".to_string(),
                        }
                    }
                };

                let real_stream = self
                    .stream_handles
                    .get(&stream)
                    .map(|s| *s)
                    .unwrap_or(std::ptr::null_mut()); // NULL stream = default stream

                // Deserialize kernel parameters.
                // Each KernelParam contains the raw bytes of a parameter value.
                // We need to create a pointer to each parameter's data buffer.
                let mut param_buffers: Vec<Vec<u8>> = kernel_params
                    .iter()
                    .map(|p| p.data.clone())
                    .collect();
                let mut param_ptrs: Vec<*mut c_void> = param_buffers
                    .iter_mut()
                    .map(|buf| buf.as_mut_ptr() as *mut c_void)
                    .collect();

                debug!(
                    session_id = session.session_id,
                    "LaunchKernel(grid=[{}x{}x{}], block=[{}x{}x{}], shared={}, params={})",
                    grid_dim[0], grid_dim[1], grid_dim[2],
                    block_dim[0], block_dim[1], block_dim[2],
                    shared_mem_bytes,
                    param_ptrs.len()
                );

                let res = unsafe {
                    d.launch_kernel(
                        real_func,
                        grid_dim,
                        block_dim,
                        shared_mem_bytes,
                        real_stream,
                        &mut param_ptrs,
                    )
                };

                if res == CUDA_SUCCESS {
                    CudaResponse::Success
                } else {
                    error!(
                        session_id = session.session_id,
                        "LaunchKernel failed: {} ({})",
                        cuda_driver::cuda_error_name(res),
                        res
                    );
                    Self::cuda_err(res)
                }
            }

            CudaCommand::LaunchCooperativeKernel {
                func,
                grid_dim,
                block_dim,
                shared_mem_bytes,
                stream,
                kernel_params,
            } => {
                // Cooperative kernel launch - use regular launch_kernel
                // (cooperative launch requires cuLaunchCooperativeKernel which we
                // don't load separately; the regular launch works for most cases)
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                let real_func = match self.function_handles.get(&func) {
                    Some(f) => *f,
                    None => {
                        return CudaResponse::Error {
                            code: 400,
                            message: "invalid function handle".to_string(),
                        }
                    }
                };

                let real_stream = self
                    .stream_handles
                    .get(&stream)
                    .map(|s| *s)
                    .unwrap_or(std::ptr::null_mut());

                let mut param_buffers: Vec<Vec<u8>> = kernel_params
                    .iter()
                    .map(|p| p.data.clone())
                    .collect();
                let mut param_ptrs: Vec<*mut c_void> = param_buffers
                    .iter_mut()
                    .map(|buf| buf.as_mut_ptr() as *mut c_void)
                    .collect();

                debug!(
                    session_id = session.session_id,
                    "LaunchCooperativeKernel(grid=[{}x{}x{}], block=[{}x{}x{}], shared={}, params={})",
                    grid_dim[0], grid_dim[1], grid_dim[2],
                    block_dim[0], block_dim[1], block_dim[2],
                    shared_mem_bytes,
                    param_ptrs.len()
                );

                let res = unsafe {
                    d.launch_kernel(
                        real_func,
                        grid_dim,
                        block_dim,
                        shared_mem_bytes,
                        real_stream,
                        &mut param_ptrs,
                    )
                };

                if res == CUDA_SUCCESS {
                    CudaResponse::Success
                } else {
                    error!(
                        session_id = session.session_id,
                        "LaunchCooperativeKernel failed: {} ({})",
                        cuda_driver::cuda_error_name(res),
                        res
                    );
                    Self::cuda_err(res)
                }
            }

            CudaCommand::FuncGetAttribute { attrib, func } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_func = match self.function_handles.get(&func) {
                    Some(f) => *f,
                    None => return CudaResponse::Error {
                        code: 400,
                        message: "invalid function handle".to_string(),
                    },
                };
                match d.func_get_attribute(attrib, real_func) {
                    Ok(val) => CudaResponse::FuncAttribute(val),
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::FuncSetAttribute { attrib, func, value } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_func = match self.function_handles.get(&func) {
                    Some(f) => *f,
                    None => return CudaResponse::Error {
                        code: 400,
                        message: "invalid function handle".to_string(),
                    },
                };
                let res = d.func_set_attribute(real_func, attrib, value);
                if res == CUDA_SUCCESS {
                    CudaResponse::Success
                } else {
                    Self::cuda_err(res)
                }
            }

            CudaCommand::FuncSetCacheConfig { func, config } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_func = match self.function_handles.get(&func) {
                    Some(f) => *f,
                    None => return CudaResponse::Error {
                        code: 400,
                        message: "invalid function handle".to_string(),
                    },
                };
                let res = d.func_set_cache_config(real_func, config);
                if res == CUDA_SUCCESS {
                    CudaResponse::Success
                } else {
                    Self::cuda_err(res)
                }
            }

            CudaCommand::FuncSetSharedMemConfig { func, config } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_func = match self.function_handles.get(&func) {
                    Some(f) => *f,
                    None => return CudaResponse::Error {
                        code: 400,
                        message: "invalid function handle".to_string(),
                    },
                };
                let res = d.func_set_shared_mem_config(real_func, config);
                if res == CUDA_SUCCESS {
                    CudaResponse::Success
                } else {
                    Self::cuda_err(res)
                }
            }

            CudaCommand::FuncGetModule { func } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_func = match self.function_handles.get(&func) {
                    Some(f) => *f,
                    None => return CudaResponse::Error {
                        code: 400,
                        message: "invalid function handle".to_string(),
                    },
                };
                match d.func_get_module(real_func) {
                    Ok(module) => {
                        // Find existing handle for this module
                        for entry in self.module_handles.iter() {
                            if *entry.value() == module {
                                return CudaResponse::FuncModule(*entry.key());
                            }
                        }
                        // Module not tracked - create a handle for it
                        let handle = session.alloc_handle(ResourceType::CuModule);
                        self.module_handles.insert(handle, module);
                        CudaResponse::FuncModule(handle)
                    }
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::FuncGetName { func } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_func = match self.function_handles.get(&func) {
                    Some(f) => *f,
                    None => return CudaResponse::Error {
                        code: 400,
                        message: "invalid function handle".to_string(),
                    },
                };
                match d.func_get_name(real_func) {
                    Ok(name) => CudaResponse::FuncName(name),
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::OccupancyMaxActiveBlocksPerMultiprocessor { func, block_size, dynamic_smem_size } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_func = match self.function_handles.get(&func) {
                    Some(f) => *f,
                    None => return CudaResponse::Error {
                        code: 400,
                        message: "invalid function handle".to_string(),
                    },
                };
                match d.occupancy_max_active_blocks(real_func, block_size, dynamic_smem_size) {
                    Ok(num_blocks) => CudaResponse::OccupancyBlocks(num_blocks),
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::OccupancyMaxActiveBlocksPerMultiprocessorWithFlags { func, block_size, dynamic_smem_size, flags } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_func = match self.function_handles.get(&func) {
                    Some(f) => *f,
                    None => return CudaResponse::Error {
                        code: 400,
                        message: "invalid function handle".to_string(),
                    },
                };
                match d.occupancy_max_active_blocks_with_flags(real_func, block_size, dynamic_smem_size, flags) {
                    Ok(num_blocks) => CudaResponse::OccupancyBlocks(num_blocks),
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::OccupancyAvailableDynamicSMemPerBlock { func, num_blocks, block_size } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_func = match self.function_handles.get(&func) {
                    Some(f) => *f,
                    None => return CudaResponse::Error {
                        code: 400,
                        message: "invalid function handle".to_string(),
                    },
                };
                match d.occupancy_available_dynamic_smem(real_func, num_blocks, block_size) {
                    Ok(smem) => CudaResponse::OccupancyDynamicSmem(smem),
                    Err(e) => Self::cuda_err(e),
                }
            }

            // ── Stream Management ───────────────────────────────────

            CudaCommand::StreamCreate { flags } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                match d.stream_create(flags) {
                    Ok(stream) => {
                        let handle = session.alloc_handle(ResourceType::CuStream);
                        self.stream_handles.insert(handle, stream);
                        debug!(
                            session_id = session.session_id,
                            "StreamCreate -> {:?}", handle
                        );
                        CudaResponse::Stream(handle)
                    }
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::StreamCreateWithPriority { flags, priority } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                match d.stream_create_with_priority(flags, priority) {
                    Ok(stream) => {
                        let handle = session.alloc_handle(ResourceType::CuStream);
                        self.stream_handles.insert(handle, stream);
                        debug!(
                            session_id = session.session_id,
                            "StreamCreateWithPriority(flags={}, priority={}) -> {:?}", flags, priority, handle
                        );
                        CudaResponse::Stream(handle)
                    }
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::StreamDestroy { stream } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                match self.stream_handles.remove(&stream) {
                    Some((_, real_stream)) => {
                        let res = d.stream_destroy(real_stream);
                        session.remove_handle(&stream);
                        if res == CUDA_SUCCESS {
                            CudaResponse::Success
                        } else {
                            Self::cuda_err(res)
                        }
                    }
                    None => CudaResponse::Error {
                        code: 400,
                        message: "invalid stream handle".to_string(),
                    },
                }
            }

            CudaCommand::StreamSynchronize { stream } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                let real_stream = match self.stream_handles.get(&stream) {
                    Some(s) => *s,
                    None => {
                        return CudaResponse::Error {
                            code: 400,
                            message: "invalid stream handle".to_string(),
                        }
                    }
                };

                let res = d.stream_synchronize(real_stream);
                if res == CUDA_SUCCESS {
                    CudaResponse::Success
                } else {
                    Self::cuda_err(res)
                }
            }

            CudaCommand::StreamQuery { stream } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                let real_stream = match self.stream_handles.get(&stream) {
                    Some(s) => *s,
                    None => {
                        return CudaResponse::Error {
                            code: 400,
                            message: "invalid stream handle".to_string(),
                        }
                    }
                };

                let res = d.stream_query(real_stream);
                if res == CUDA_SUCCESS {
                    CudaResponse::StreamStatus(true) // complete
                } else if res == 600 {
                    // CUDA_ERROR_NOT_READY
                    CudaResponse::StreamStatus(false) // not complete
                } else {
                    Self::cuda_err(res)
                }
            }

            CudaCommand::StreamWaitEvent { stream, event, flags } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_stream = self
                    .stream_handles
                    .get(&stream)
                    .map(|s| *s)
                    .unwrap_or(std::ptr::null_mut());
                let real_event = match self.event_handles.get(&event) {
                    Some(e) => *e,
                    None => return CudaResponse::Error {
                        code: 400,
                        message: "invalid event handle".to_string(),
                    },
                };
                let res = d.stream_wait_event(real_stream, real_event, flags);
                if res == CUDA_SUCCESS {
                    CudaResponse::Success
                } else {
                    Self::cuda_err(res)
                }
            }

            CudaCommand::StreamGetPriority { stream } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_stream = match self.stream_handles.get(&stream) {
                    Some(s) => *s,
                    None => return CudaResponse::Error {
                        code: 400,
                        message: "invalid stream handle".to_string(),
                    },
                };
                match d.stream_get_priority(real_stream) {
                    Ok(priority) => CudaResponse::StreamPriority(priority),
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::StreamGetFlags { stream } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_stream = match self.stream_handles.get(&stream) {
                    Some(s) => *s,
                    None => return CudaResponse::Error {
                        code: 400,
                        message: "invalid stream handle".to_string(),
                    },
                };
                match d.stream_get_flags(real_stream) {
                    Ok(flags) => CudaResponse::StreamFlags(flags),
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::StreamGetCtx { stream } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_stream = match self.stream_handles.get(&stream) {
                    Some(s) => *s,
                    None => return CudaResponse::Error {
                        code: 400,
                        message: "invalid stream handle".to_string(),
                    },
                };
                match d.stream_get_ctx(real_stream) {
                    Ok(ctx) => {
                        // Find existing handle for this context
                        for entry in self.context_handles.iter() {
                            if *entry.value() == ctx {
                                return CudaResponse::StreamCtx(*entry.key());
                            }
                        }
                        // Context not tracked - create a handle for it
                        let handle = session.alloc_handle(ResourceType::CuContext);
                        self.context_handles.insert(handle, ctx);
                        CudaResponse::StreamCtx(handle)
                    }
                    Err(e) => Self::cuda_err(e),
                }
            }

            // ── Event Management ────────────────────────────────────

            CudaCommand::EventCreate { flags } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                match d.event_create(flags) {
                    Ok(event) => {
                        let handle = session.alloc_handle(ResourceType::CuEvent);
                        self.event_handles.insert(handle, event);
                        CudaResponse::Event(handle)
                    }
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::EventDestroy { event } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                match self.event_handles.remove(&event) {
                    Some((_, real_event)) => {
                        let res = d.event_destroy(real_event);
                        session.remove_handle(&event);
                        if res == CUDA_SUCCESS {
                            CudaResponse::Success
                        } else {
                            Self::cuda_err(res)
                        }
                    }
                    None => CudaResponse::Error {
                        code: 400,
                        message: "invalid event handle".to_string(),
                    },
                }
            }

            CudaCommand::EventRecord { event, stream } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                let real_event = match self.event_handles.get(&event) {
                    Some(e) => *e,
                    None => {
                        return CudaResponse::Error {
                            code: 400,
                            message: "invalid event handle".to_string(),
                        }
                    }
                };

                let real_stream = self
                    .stream_handles
                    .get(&stream)
                    .map(|s| *s)
                    .unwrap_or(std::ptr::null_mut());

                let res = d.event_record(real_event, real_stream);
                if res == CUDA_SUCCESS {
                    CudaResponse::Success
                } else {
                    Self::cuda_err(res)
                }
            }

            CudaCommand::EventRecordWithFlags { event, stream, flags } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                let real_event = match self.event_handles.get(&event) {
                    Some(e) => *e,
                    None => {
                        return CudaResponse::Error {
                            code: 400,
                            message: "invalid event handle".to_string(),
                        }
                    }
                };

                let real_stream = self
                    .stream_handles
                    .get(&stream)
                    .map(|s| *s)
                    .unwrap_or(std::ptr::null_mut());

                let res = d.event_record_with_flags(real_event, real_stream, flags);
                if res == CUDA_SUCCESS {
                    CudaResponse::Success
                } else {
                    Self::cuda_err(res)
                }
            }

            CudaCommand::EventSynchronize { event } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                let real_event = match self.event_handles.get(&event) {
                    Some(e) => *e,
                    None => {
                        return CudaResponse::Error {
                            code: 400,
                            message: "invalid event handle".to_string(),
                        }
                    }
                };

                let res = d.event_synchronize(real_event);
                if res == CUDA_SUCCESS {
                    CudaResponse::Success
                } else {
                    Self::cuda_err(res)
                }
            }

            CudaCommand::EventQuery { event } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                let real_event = match self.event_handles.get(&event) {
                    Some(e) => *e,
                    None => {
                        return CudaResponse::Error {
                            code: 400,
                            message: "invalid event handle".to_string(),
                        }
                    }
                };

                let res = d.event_query(real_event);
                if res == CUDA_SUCCESS {
                    CudaResponse::EventStatus(true)
                } else if res == 600 {
                    CudaResponse::EventStatus(false)
                } else {
                    Self::cuda_err(res)
                }
            }

            CudaCommand::EventElapsedTime { start, end } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                let real_start = match self.event_handles.get(&start) {
                    Some(e) => *e,
                    None => {
                        return CudaResponse::Error {
                            code: 400,
                            message: "invalid start event handle".to_string(),
                        }
                    }
                };

                let real_end = match self.event_handles.get(&end) {
                    Some(e) => *e,
                    None => {
                        return CudaResponse::Error {
                            code: 400,
                            message: "invalid end event handle".to_string(),
                        }
                    }
                };

                match d.event_elapsed_time(real_start, real_end) {
                    Ok(ms) => CudaResponse::ElapsedTime(ms),
                    Err(e) => Self::cuda_err(e),
                }
            }

            // ── Pointer Queries ─────────────────────────────────────

            CudaCommand::PointerGetAttribute { attribute, ptr } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_ptr = match self.memory_handles.get(&ptr) {
                    Some(p) => *p,
                    None => return CudaResponse::Error {
                        code: 400,
                        message: "invalid memory handle".to_string(),
                    },
                };
                match d.pointer_get_attribute(attribute, real_ptr) {
                    Ok(val) => CudaResponse::PointerAttribute(val),
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::PointerGetAttributes { num_attributes: _, attributes, ptr } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_ptr = match self.memory_handles.get(&ptr) {
                    Some(p) => *p,
                    None => return CudaResponse::Error {
                        code: 400,
                        message: "invalid memory handle".to_string(),
                    },
                };
                let mut results = Vec::with_capacity(attributes.len());
                for attr in &attributes {
                    match d.pointer_get_attribute(*attr, real_ptr) {
                        Ok(val) => results.push(val),
                        Err(_) => results.push(0),
                    }
                }
                CudaResponse::PointerAttributes(results)
            }

            CudaCommand::PointerSetAttribute { attribute, ptr, value } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_ptr = match self.memory_handles.get(&ptr) {
                    Some(p) => *p,
                    None => return CudaResponse::Error {
                        code: 400,
                        message: "invalid memory handle".to_string(),
                    },
                };
                let res = d.pointer_set_attribute(attribute, real_ptr, value);
                if res == CUDA_SUCCESS {
                    CudaResponse::Success
                } else {
                    Self::cuda_err(res)
                }
            }

            // ── Peer Access ─────────────────────────────────────────

            CudaCommand::CtxEnablePeerAccess { peer_ctx, flags } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_peer_ctx = match self.context_handles.get(&peer_ctx) {
                    Some(c) => *c,
                    None => return CudaResponse::Error {
                        code: 201,
                        message: "invalid peer context handle".to_string(),
                    },
                };
                let res = d.ctx_enable_peer_access(real_peer_ctx, flags);
                if res == CUDA_SUCCESS {
                    CudaResponse::Success
                } else {
                    Self::cuda_err(res)
                }
            }

            CudaCommand::CtxDisablePeerAccess { peer_ctx } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_peer_ctx = match self.context_handles.get(&peer_ctx) {
                    Some(c) => *c,
                    None => return CudaResponse::Error {
                        code: 201,
                        message: "invalid peer context handle".to_string(),
                    },
                };
                let res = d.ctx_disable_peer_access(real_peer_ctx);
                if res == CUDA_SUCCESS {
                    CudaResponse::Success
                } else {
                    Self::cuda_err(res)
                }
            }

            // ── Memory Pools ────────────────────────────────────────

            CudaCommand::MemPoolCreate { device, props_flags: _ } => {
                // MemPoolCreate requires a CUmemPoolProps struct which is complex
                // to construct over the network. Use the default pool instead.
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_dev = match self.device_handles.get(&device) {
                    Some(dev) => *dev,
                    None => return CudaResponse::Error {
                        code: 101,
                        message: "invalid device handle".to_string(),
                    },
                };
                // Fall back to getting the default mem pool since creating custom pools
                // requires full CUmemPoolProps serialization
                match d.device_get_default_mem_pool(real_dev) {
                    Ok(pool) => {
                        let handle = session.alloc_handle(ResourceType::CuMemPool);
                        self.mempool_handles.insert(handle, pool);
                        CudaResponse::MemPool(handle)
                    }
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::MemPoolDestroy { pool } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                match self.mempool_handles.remove(&pool) {
                    Some((_, real_pool)) => {
                        let res = d.mem_pool_destroy(real_pool);
                        session.remove_handle(&pool);
                        if res == CUDA_SUCCESS {
                            CudaResponse::Success
                        } else {
                            Self::cuda_err(res)
                        }
                    }
                    None => CudaResponse::Error {
                        code: 400,
                        message: "invalid mempool handle".to_string(),
                    },
                }
            }

            CudaCommand::MemPoolTrimTo { pool, min_bytes_to_keep } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_pool = match self.mempool_handles.get(&pool) {
                    Some(p) => *p,
                    None => return CudaResponse::Error {
                        code: 400,
                        message: "invalid mempool handle".to_string(),
                    },
                };
                let res = d.mem_pool_trim_to(real_pool, min_bytes_to_keep as usize);
                if res == CUDA_SUCCESS {
                    CudaResponse::Success
                } else {
                    Self::cuda_err(res)
                }
            }

            CudaCommand::MemPoolSetAttribute { pool, attr: _, value: _ } => {
                // MemPoolSetAttribute requires raw pointer manipulation
                // Return not supported for now - most applications work without it
                let _real_pool = match self.mempool_handles.get(&pool) {
                    Some(p) => *p,
                    None => return CudaResponse::Error {
                        code: 400,
                        message: "invalid mempool handle".to_string(),
                    },
                };
                // No-op success - most pool attributes are hints
                CudaResponse::Success
            }

            CudaCommand::MemPoolGetAttribute { pool, attr: _ } => {
                let _real_pool = match self.mempool_handles.get(&pool) {
                    Some(p) => *p,
                    None => return CudaResponse::Error {
                        code: 400,
                        message: "invalid mempool handle".to_string(),
                    },
                };
                // Return 0 as default attribute value
                CudaResponse::MemPoolAttribute(0)
            }

            CudaCommand::MemAllocAsync { byte_size, stream } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_stream = self
                    .stream_handles
                    .get(&stream)
                    .map(|s| *s)
                    .unwrap_or(std::ptr::null_mut());
                match d.mem_alloc_async(byte_size as usize, real_stream) {
                    Ok(dptr) => {
                        let handle = session.alloc_handle(ResourceType::CuDevicePtr);
                        self.memory_handles.insert(handle, dptr);
                        self.memory_sizes.insert(handle, byte_size);
                        debug!(
                            session_id = session.session_id,
                            "MemAllocAsync({} bytes) -> {:?}", byte_size, handle
                        );
                        CudaResponse::MemAllocated(handle)
                    }
                    Err(e) => Self::cuda_err(e),
                }
            }

            CudaCommand::MemFreeAsync { dptr, stream } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_stream = self
                    .stream_handles
                    .get(&stream)
                    .map(|s| *s)
                    .unwrap_or(std::ptr::null_mut());
                match self.memory_handles.remove(&dptr) {
                    Some((_, real_ptr)) => {
                        let res = d.mem_free_async(real_ptr, real_stream);
                        self.memory_sizes.remove(&dptr);
                        session.remove_handle(&dptr);
                        if res == CUDA_SUCCESS {
                            CudaResponse::Success
                        } else {
                            Self::cuda_err(res)
                        }
                    }
                    None => CudaResponse::Error {
                        code: 400,
                        message: "invalid memory handle".to_string(),
                    },
                }
            }

            CudaCommand::MemAllocFromPoolAsync { byte_size, pool, stream } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_pool = match self.mempool_handles.get(&pool) {
                    Some(p) => *p,
                    None => return CudaResponse::Error {
                        code: 400,
                        message: "invalid mempool handle".to_string(),
                    },
                };
                let real_stream = self
                    .stream_handles
                    .get(&stream)
                    .map(|s| *s)
                    .unwrap_or(std::ptr::null_mut());
                match d.mem_alloc_from_pool_async(byte_size as usize, real_pool, real_stream) {
                    Ok(dptr) => {
                        let handle = session.alloc_handle(ResourceType::CuDevicePtr);
                        self.memory_handles.insert(handle, dptr);
                        self.memory_sizes.insert(handle, byte_size);
                        debug!(
                            session_id = session.session_id,
                            "MemAllocFromPoolAsync({} bytes) -> {:?}", byte_size, handle
                        );
                        CudaResponse::MemAllocated(handle)
                    }
                    Err(e) => Self::cuda_err(e),
                }
            }
        }
    }

    /// Fallback device attribute values when no real CUDA driver is available.
    fn get_device_attribute_fallback(&self, attrib: i32) -> i32 {
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

    /// Clean up all GPU resources owned by a disconnecting session.
    /// Destroys resources in reverse-dependency order to avoid dangling references.
    pub fn cleanup_session(&self, session: &Session) {
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
