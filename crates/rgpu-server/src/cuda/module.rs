use tracing::debug;

use rgpu_protocol::cuda_commands::CudaResponse;
use rgpu_protocol::handle::{NetworkHandle, ResourceType};

use crate::cuda_driver::CUDA_SUCCESS;
use crate::session::Session;

use super::CudaExecutor;

impl CudaExecutor {
    pub(crate) fn handle_module_load_data(&self, session: &Session, image: Vec<u8>) -> CudaResponse {
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

    pub(crate) fn handle_module_unload(&self, session: &Session, module: NetworkHandle) -> CudaResponse {
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

    pub(crate) fn handle_module_get_function(&self, session: &Session, module: NetworkHandle, name: String) -> CudaResponse {
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

    pub(crate) fn handle_module_get_global(&self, session: &Session, module: NetworkHandle, name: String) -> CudaResponse {
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

    pub(crate) fn handle_module_load(&self, session: &Session, fname: String) -> CudaResponse {
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

    pub(crate) fn handle_module_load_data_ex(&self, session: &Session, image: Vec<u8>) -> CudaResponse {
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

    pub(crate) fn handle_module_load_fat_binary(&self, session: &Session, fat_cubin: Vec<u8>) -> CudaResponse {
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

    pub(crate) fn handle_link_create(&self, session: &Session) -> CudaResponse {
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

    pub(crate) fn handle_link_add_data(&self, link: NetworkHandle, jit_type: i32, data: Vec<u8>, name: String) -> CudaResponse {
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

    pub(crate) fn handle_link_add_file(&self, link: NetworkHandle, jit_type: i32, path: String) -> CudaResponse {
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

    pub(crate) fn handle_link_complete(&self, link: NetworkHandle) -> CudaResponse {
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

    pub(crate) fn handle_link_destroy(&self, session: &Session, link: NetworkHandle) -> CudaResponse {
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
}
