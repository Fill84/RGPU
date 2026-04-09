use rgpu_protocol::cuda_commands::CudaResponse;
use rgpu_protocol::handle::{NetworkHandle, ResourceType};

use crate::cuda_driver::CUDA_SUCCESS;
use crate::session::Session;

use super::CudaExecutor;

impl CudaExecutor {
    pub(crate) fn handle_event_create(&self, session: &Session, flags: u32) -> CudaResponse {
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

    pub(crate) fn handle_event_destroy(&self, session: &Session, event: NetworkHandle) -> CudaResponse {
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

    pub(crate) fn handle_event_record(&self, event: NetworkHandle, stream: NetworkHandle) -> CudaResponse {
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

    pub(crate) fn handle_event_record_with_flags(&self, event: NetworkHandle, stream: NetworkHandle, flags: u32) -> CudaResponse {
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

    pub(crate) fn handle_event_synchronize(&self, event: NetworkHandle) -> CudaResponse {
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

    pub(crate) fn handle_event_query(&self, event: NetworkHandle) -> CudaResponse {
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

    pub(crate) fn handle_event_elapsed_time(&self, start: NetworkHandle, end: NetworkHandle) -> CudaResponse {
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
}
