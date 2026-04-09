use tracing::debug;

use rgpu_protocol::cuda_commands::CudaResponse;
use rgpu_protocol::handle::{NetworkHandle, ResourceType};

use crate::cuda_driver::CUDA_SUCCESS;
use crate::session::Session;

use super::CudaExecutor;

impl CudaExecutor {
    pub(crate) fn handle_init(&self, session: &Session, flags: u32) -> CudaResponse {
        tracing::info!(
            session_id = session.session_id,
            "CUDA init (flags={})", flags
        );
        // Already initialized in constructor
        CudaResponse::Success
    }

    pub(crate) fn handle_driver_get_version(&self) -> CudaResponse {
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

    pub(crate) fn handle_device_get_count(&self, session: &Session) -> CudaResponse {
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

    pub(crate) fn handle_device_get(&self, session: &Session, ordinal: i32) -> CudaResponse {
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

    pub(crate) fn handle_device_get_name(&self, device: NetworkHandle) -> CudaResponse {
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

    pub(crate) fn handle_device_get_attribute(&self, attrib: i32, device: NetworkHandle) -> CudaResponse {
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

    pub(crate) fn handle_device_total_mem(&self, device: NetworkHandle) -> CudaResponse {
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

    pub(crate) fn handle_device_compute_capability(&self, device: NetworkHandle) -> CudaResponse {
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

    pub(crate) fn handle_device_get_uuid(&self, device: NetworkHandle) -> CudaResponse {
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

    pub(crate) fn handle_device_get_p2p_attribute(&self, attrib: i32, src_device: NetworkHandle, dst_device: NetworkHandle) -> CudaResponse {
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

    pub(crate) fn handle_device_can_access_peer(&self, device: NetworkHandle, peer_device: NetworkHandle) -> CudaResponse {
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

    pub(crate) fn handle_device_get_by_pci_bus_id(&self, session: &Session, pci_bus_id: String) -> CudaResponse {
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

    pub(crate) fn handle_device_get_pci_bus_id(&self, device: NetworkHandle) -> CudaResponse {
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

    pub(crate) fn handle_device_get_default_mem_pool(&self, session: &Session, device: NetworkHandle) -> CudaResponse {
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

    pub(crate) fn handle_device_get_mem_pool(&self, session: &Session, device: NetworkHandle) -> CudaResponse {
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

    pub(crate) fn handle_device_set_mem_pool(&self, device: NetworkHandle, mem_pool: NetworkHandle) -> CudaResponse {
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

    pub(crate) fn handle_device_get_texture_1d_linear_max_width(&self) -> CudaResponse {
        // This API is rarely used and requires cuDeviceGetTexture1DLinearMaxWidth
        // which is not commonly loaded. Return a reasonable default.
        CudaResponse::Texture1DMaxWidth(1 << 27) // 128M texels
    }

    pub(crate) fn handle_device_get_exec_affinity_support(&self) -> CudaResponse {
        // cuDeviceGetExecAffinitySupport - not commonly available, return false
        CudaResponse::BoolResult(false)
    }

    pub(crate) fn handle_device_primary_ctx_retain(&self, session: &Session, device: NetworkHandle) -> CudaResponse {
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

    pub(crate) fn handle_device_primary_ctx_release(&self, device: NetworkHandle) -> CudaResponse {
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

    pub(crate) fn handle_device_primary_ctx_reset(&self, device: NetworkHandle) -> CudaResponse {
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

    pub(crate) fn handle_device_primary_ctx_get_state(&self, device: NetworkHandle) -> CudaResponse {
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

    pub(crate) fn handle_device_primary_ctx_set_flags(&self, device: NetworkHandle, flags: u32) -> CudaResponse {
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
}
