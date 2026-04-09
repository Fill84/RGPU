use tracing::debug;

use rgpu_protocol::cuda_commands::CudaResponse;
use rgpu_protocol::handle::{NetworkHandle, ResourceType};

use crate::cuda_driver::CUDA_SUCCESS;
use crate::session::Session;

use super::CudaExecutor;

impl CudaExecutor {
    pub(crate) fn handle_ctx_create(&self, session: &Session, flags: u32, device: NetworkHandle) -> CudaResponse {
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
                // Track as current context for this session
                self.session_current_ctx.insert(session.session_id, ctx);
                debug!(
                    session_id = session.session_id,
                    "CtxCreate(device={:?}) -> {:?}", device, handle
                );
                CudaResponse::Context(handle)
            }
            Err(e) => Self::cuda_err(e),
        }
    }

    pub(crate) fn handle_ctx_destroy(&self, session: &Session, ctx: NetworkHandle) -> CudaResponse {
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

    pub(crate) fn handle_ctx_set_current(&self, session: &Session, ctx: NetworkHandle) -> CudaResponse {
        let d = match self.driver() {
            Ok(d) => d,
            Err(e) => return e,
        };

        // Handle null context (detach current context)
        if ctx.is_null() {
            let res = d.ctx_set_current(std::ptr::null_mut());
            if res == CUDA_SUCCESS {
                self.session_current_ctx.remove(&session.session_id);
                CudaResponse::Success
            } else {
                Self::cuda_err(res)
            }
        } else {
            match self.context_handles.get(&ctx) {
                Some(real_ctx) => {
                    let res = d.ctx_set_current(*real_ctx);
                    if res == CUDA_SUCCESS {
                        self.session_current_ctx.insert(session.session_id, *real_ctx);
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
    }

    pub(crate) fn handle_ctx_get_current(&self, session: &Session) -> CudaResponse {
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

    pub(crate) fn handle_ctx_synchronize(&self) -> CudaResponse {
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

    pub(crate) fn handle_ctx_push_current(&self, session: &Session, ctx: NetworkHandle) -> CudaResponse {
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
        // Use cuCtxSetCurrent instead of cuCtxPushCurrent to avoid
        // thread-local stack issues with async runtimes
        let res = d.ctx_set_current(real_ctx);
        if res == CUDA_SUCCESS {
            self.session_current_ctx.insert(session.session_id, real_ctx);
            CudaResponse::Success
        } else {
            Self::cuda_err(res)
        }
    }

    pub(crate) fn handle_ctx_pop_current(&self, session: &Session) -> CudaResponse {
        let d = match self.driver() {
            Ok(d) => d,
            Err(e) => return e,
        };
        // Instead of relying on the thread-local context stack, use
        // our session-tracked current context
        if let Some(ctx_entry) = self.session_current_ctx.get(&session.session_id) {
            let ctx = *ctx_entry;
            drop(ctx_entry);
            // Find the handle for this context
            for entry in self.context_handles.iter() {
                if *entry.value() == ctx {
                    // Detach context
                    let _ = d.ctx_set_current(std::ptr::null_mut());
                    self.session_current_ctx.remove(&session.session_id);
                    return CudaResponse::Context(*entry.key());
                }
            }
        }
        // No tracked context — try real pop as fallback
        match d.ctx_pop_current() {
            Ok(ctx) => {
                self.session_current_ctx.remove(&session.session_id);
                for entry in self.context_handles.iter() {
                    if *entry.value() == ctx {
                        return CudaResponse::Context(*entry.key());
                    }
                }
                let handle = session.alloc_handle(ResourceType::CuContext);
                self.context_handles.insert(handle, ctx);
                CudaResponse::Context(handle)
            }
            Err(e) => Self::cuda_err(e),
        }
    }

    pub(crate) fn handle_ctx_get_device(&self, session: &Session) -> CudaResponse {
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

    pub(crate) fn handle_ctx_set_cache_config(&self, config: i32) -> CudaResponse {
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

    pub(crate) fn handle_ctx_get_cache_config(&self) -> CudaResponse {
        let d = match self.driver() {
            Ok(d) => d,
            Err(e) => return e,
        };
        match d.ctx_get_cache_config() {
            Ok(config) => CudaResponse::CacheConfig(config),
            Err(e) => Self::cuda_err(e),
        }
    }

    pub(crate) fn handle_ctx_set_limit(&self, limit: i32, value: u64) -> CudaResponse {
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

    pub(crate) fn handle_ctx_get_limit(&self, limit: i32) -> CudaResponse {
        let d = match self.driver() {
            Ok(d) => d,
            Err(e) => return e,
        };
        match d.ctx_get_limit(limit) {
            Ok(value) => CudaResponse::ContextLimit(value),
            Err(e) => Self::cuda_err(e),
        }
    }

    pub(crate) fn handle_ctx_get_stream_priority_range(&self) -> CudaResponse {
        let d = match self.driver() {
            Ok(d) => d,
            Err(e) => return e,
        };
        match d.ctx_get_stream_priority_range() {
            Ok((least, greatest)) => CudaResponse::StreamPriorityRange { least, greatest },
            Err(e) => Self::cuda_err(e),
        }
    }

    pub(crate) fn handle_ctx_get_api_version(&self, ctx: NetworkHandle) -> CudaResponse {
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

    pub(crate) fn handle_ctx_get_flags(&self) -> CudaResponse {
        let d = match self.driver() {
            Ok(d) => d,
            Err(e) => return e,
        };
        match d.ctx_get_flags() {
            Ok(flags) => CudaResponse::ContextFlags(flags),
            Err(e) => Self::cuda_err(e),
        }
    }

    pub(crate) fn handle_ctx_set_flags(&self, flags: u32) -> CudaResponse {
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

    pub(crate) fn handle_ctx_reset_persisting_l2_cache(&self) -> CudaResponse {
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

    pub(crate) fn handle_ctx_enable_peer_access(&self, peer_ctx: NetworkHandle, flags: u32) -> CudaResponse {
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

    pub(crate) fn handle_ctx_disable_peer_access(&self, peer_ctx: NetworkHandle) -> CudaResponse {
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
}
