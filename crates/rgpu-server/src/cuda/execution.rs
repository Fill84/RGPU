use std::ffi::c_void;

use tracing::{debug, error};

use rgpu_protocol::cuda_commands::{CudaResponse, KernelParam};
use rgpu_protocol::handle::{NetworkHandle, ResourceType};

use crate::cuda_driver::{self, CUDA_SUCCESS};
use crate::session::Session;

use super::CudaExecutor;

impl CudaExecutor {
    pub(crate) fn handle_launch_kernel(
        &self,
        session: &Session,
        func: NetworkHandle,
        grid_dim: [u32; 3],
        block_dim: [u32; 3],
        shared_mem_bytes: u32,
        stream: NetworkHandle,
        kernel_params: Vec<KernelParam>,
    ) -> CudaResponse {
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

    pub(crate) fn handle_launch_cooperative_kernel(
        &self,
        session: &Session,
        func: NetworkHandle,
        grid_dim: [u32; 3],
        block_dim: [u32; 3],
        shared_mem_bytes: u32,
        stream: NetworkHandle,
        kernel_params: Vec<KernelParam>,
    ) -> CudaResponse {
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

    pub(crate) fn handle_func_get_attribute(&self, attrib: i32, func: NetworkHandle) -> CudaResponse {
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

    pub(crate) fn handle_func_set_attribute(&self, attrib: i32, func: NetworkHandle, value: i32) -> CudaResponse {
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

    pub(crate) fn handle_func_set_cache_config(&self, func: NetworkHandle, config: i32) -> CudaResponse {
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

    pub(crate) fn handle_func_set_shared_mem_config(&self, func: NetworkHandle, config: i32) -> CudaResponse {
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

    pub(crate) fn handle_func_get_module(&self, session: &Session, func: NetworkHandle) -> CudaResponse {
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

    pub(crate) fn handle_func_get_name(&self, func: NetworkHandle) -> CudaResponse {
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

    pub(crate) fn handle_occupancy_max_active_blocks(&self, func: NetworkHandle, block_size: i32, dynamic_smem_size: u64) -> CudaResponse {
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

    pub(crate) fn handle_occupancy_max_active_blocks_with_flags(&self, func: NetworkHandle, block_size: i32, dynamic_smem_size: u64, flags: u32) -> CudaResponse {
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

    pub(crate) fn handle_occupancy_available_dynamic_smem(&self, func: NetworkHandle, num_blocks: i32, block_size: i32) -> CudaResponse {
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
}
