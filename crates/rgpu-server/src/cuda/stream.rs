use tracing::debug;

use rgpu_protocol::cuda_commands::CudaResponse;
use rgpu_protocol::handle::{NetworkHandle, ResourceType};

use crate::cuda_driver::CUDA_SUCCESS;
use crate::session::Session;

use super::CudaExecutor;

impl CudaExecutor {
    pub(crate) fn handle_stream_create(&self, session: &Session, flags: u32) -> CudaResponse {
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

    pub(crate) fn handle_stream_create_with_priority(&self, session: &Session, flags: u32, priority: i32) -> CudaResponse {
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

    pub(crate) fn handle_stream_destroy(&self, session: &Session, stream: NetworkHandle) -> CudaResponse {
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

    pub(crate) fn handle_stream_synchronize(&self, stream: NetworkHandle) -> CudaResponse {
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

    pub(crate) fn handle_stream_query(&self, stream: NetworkHandle) -> CudaResponse {
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

    pub(crate) fn handle_stream_wait_event(&self, stream: NetworkHandle, event: NetworkHandle, flags: u32) -> CudaResponse {
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

    pub(crate) fn handle_stream_get_priority(&self, stream: NetworkHandle) -> CudaResponse {
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

    pub(crate) fn handle_stream_get_flags(&self, stream: NetworkHandle) -> CudaResponse {
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

    pub(crate) fn handle_stream_get_ctx(&self, session: &Session, stream: NetworkHandle) -> CudaResponse {
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
}
