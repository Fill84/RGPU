//! CUDA Execution Control API functions (kernel launch, occupancy, function attributes).

use std::ffi::{c_char, c_int, c_uint, c_void};
use tracing::{debug, error};
use rgpu_protocol::cuda_commands::{CudaCommand, CudaResponse, KernelParam};

use crate::{
    CUresult, CUfunction, CUmodule, CUstream,
    CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_UNKNOWN,
    send_cuda_command, null_stream_handle, handle_store,
};

// ── Execution Control ───────────────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn cuLaunchKernel_impl(
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
    _extra: *mut *mut c_void,
) -> CUresult {
    let local_func_id = f as u64;
    let net_func = match handle_store::get_func(local_func_id) {
        Some(h) => h,
        None => {
            error!("cuLaunchKernel: invalid function handle");
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    let stream_id = hstream as u64;
    let net_stream = if stream_id == 0 {
        null_stream_handle()
    } else {
        match handle_store::get_stream(stream_id) {
            Some(h) => h,
            None => null_stream_handle(),
        }
    };

    let mut params = Vec::new();
    if !kernel_params.is_null() {
        let mut i = 0;
        loop {
            let param_ptr = *kernel_params.add(i);
            if param_ptr.is_null() {
                break;
            }
            let data =
                std::slice::from_raw_parts(param_ptr as *const u8, std::mem::size_of::<u64>())
                    .to_vec();
            params.push(KernelParam { data });
            i += 1;
            if i >= 256 {
                break;
            }
        }
    }

    debug!(
        "cuLaunchKernel(grid=[{}x{}x{}], block=[{}x{}x{}], shared={}, params={})",
        grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y, block_dim_z,
        shared_mem_bytes, params.len()
    );

    match send_cuda_command(CudaCommand::LaunchKernel {
        func: net_func,
        grid_dim: [grid_dim_x, grid_dim_y, grid_dim_z],
        block_dim: [block_dim_x, block_dim_y, block_dim_z],
        shared_mem_bytes,
        stream: net_stream,
        kernel_params: params,
    }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuLaunchKernel(
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
    _extra: *mut *mut c_void,
) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuLaunchKernel_impl(f, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y, block_dim_z, shared_mem_bytes, hstream, kernel_params, _extra))
}

// ── Execution Control Extended ──────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn cuLaunchCooperativeKernel_impl(
    f: CUfunction,
    grid_dim_x: c_uint, grid_dim_y: c_uint, grid_dim_z: c_uint,
    block_dim_x: c_uint, block_dim_y: c_uint, block_dim_z: c_uint,
    shared_mem_bytes: c_uint, hstream: CUstream,
    kernel_params: *mut *mut c_void,
) -> CUresult {
    let net_func = match handle_store::get_func(f as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    let net_stream = if (hstream as u64) == 0 { null_stream_handle() } else { handle_store::get_stream(hstream as u64).unwrap_or_else(null_stream_handle) };

    let mut params = Vec::new();
    if !kernel_params.is_null() {
        let mut i = 0;
        loop {
            let param_ptr = *kernel_params.add(i);
            if param_ptr.is_null() { break; }
            let data = std::slice::from_raw_parts(param_ptr as *const u8, std::mem::size_of::<u64>()).to_vec();
            params.push(KernelParam { data });
            i += 1;
            if i >= 256 { break; }
        }
    }

    match send_cuda_command(CudaCommand::LaunchCooperativeKernel {
        func: net_func, grid_dim: [grid_dim_x, grid_dim_y, grid_dim_z],
        block_dim: [block_dim_x, block_dim_y, block_dim_z],
        shared_mem_bytes, stream: net_stream, kernel_params: params,
    }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuLaunchCooperativeKernel(
    f: CUfunction,
    grid_dim_x: c_uint, grid_dim_y: c_uint, grid_dim_z: c_uint,
    block_dim_x: c_uint, block_dim_y: c_uint, block_dim_z: c_uint,
    shared_mem_bytes: c_uint, hstream: CUstream,
    kernel_params: *mut *mut c_void,
) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuLaunchCooperativeKernel_impl(f, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y, block_dim_z, shared_mem_bytes, hstream, kernel_params))
}

#[allow(non_snake_case)]
unsafe fn cuFuncGetAttribute_impl(pi: *mut c_int, attrib: c_int, hfunc: CUfunction) -> CUresult {
    if pi.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let net_func = match handle_store::get_func(hfunc as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::FuncGetAttribute { attrib, func: net_func }) {
        CudaResponse::FuncAttribute(v) => { *pi = v; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuFuncGetAttribute(pi: *mut c_int, attrib: c_int, hfunc: CUfunction) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuFuncGetAttribute_impl(pi, attrib, hfunc))
}

#[allow(non_snake_case)]
unsafe fn cuFuncSetAttribute_impl(hfunc: CUfunction, attrib: c_int, value: c_int) -> CUresult {
    let net_func = match handle_store::get_func(hfunc as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::FuncSetAttribute { attrib, func: net_func, value }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuFuncSetAttribute(hfunc: CUfunction, attrib: c_int, value: c_int) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuFuncSetAttribute_impl(hfunc, attrib, value))
}

#[allow(non_snake_case)]
unsafe fn cuFuncSetCacheConfig_impl(hfunc: CUfunction, config: c_int) -> CUresult {
    let net_func = match handle_store::get_func(hfunc as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::FuncSetCacheConfig { func: net_func, config }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuFuncSetCacheConfig(hfunc: CUfunction, config: c_int) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuFuncSetCacheConfig_impl(hfunc, config))
}

#[allow(non_snake_case)]
unsafe fn cuFuncSetSharedMemConfig_impl(hfunc: CUfunction, config: c_int) -> CUresult {
    let net_func = match handle_store::get_func(hfunc as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::FuncSetSharedMemConfig { func: net_func, config }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuFuncSetSharedMemConfig(hfunc: CUfunction, config: c_int) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuFuncSetSharedMemConfig_impl(hfunc, config))
}

#[allow(non_snake_case)]
unsafe fn cuFuncGetModule_impl(hmod: *mut CUmodule, hfunc: CUfunction) -> CUresult {
    if hmod.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let net_func = match handle_store::get_func(hfunc as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::FuncGetModule { func: net_func }) {
        CudaResponse::FuncModule(handle) => { let id = handle_store::store_mod(handle); *hmod = id as CUmodule; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuFuncGetModule(hmod: *mut CUmodule, hfunc: CUfunction) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuFuncGetModule_impl(hmod, hfunc))
}

#[allow(non_snake_case)]
unsafe fn cuFuncGetName_impl(name: *mut *const c_char, hfunc: CUfunction) -> CUresult {
    if name.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let net_func = match handle_store::get_func(hfunc as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::FuncGetName { func: net_func }) {
        CudaResponse::FuncName(n) => {
            let c_str = std::ffi::CString::new(n).unwrap_or_default();
            *name = c_str.into_raw();
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuFuncGetName(name: *mut *const c_char, hfunc: CUfunction) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuFuncGetName_impl(name, hfunc))
}

#[allow(non_snake_case)]
unsafe fn cuOccupancyMaxActiveBlocksPerMultiprocessor_impl(num_blocks: *mut c_int, func: CUfunction, block_size: c_int, dynamic_smem_size: usize) -> CUresult {
    if num_blocks.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let net_func = match handle_store::get_func(func as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::OccupancyMaxActiveBlocksPerMultiprocessor { func: net_func, block_size, dynamic_smem_size: dynamic_smem_size as u64 }) {
        CudaResponse::OccupancyBlocks(b) => { *num_blocks = b; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuOccupancyMaxActiveBlocksPerMultiprocessor(num_blocks: *mut c_int, func: CUfunction, block_size: c_int, dynamic_smem_size: usize) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuOccupancyMaxActiveBlocksPerMultiprocessor_impl(num_blocks, func, block_size, dynamic_smem_size))
}

#[allow(non_snake_case)]
unsafe fn cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_impl(num_blocks: *mut c_int, func: CUfunction, block_size: c_int, dynamic_smem_size: usize, flags: c_uint) -> CUresult {
    if num_blocks.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let net_func = match handle_store::get_func(func as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::OccupancyMaxActiveBlocksPerMultiprocessorWithFlags { func: net_func, block_size, dynamic_smem_size: dynamic_smem_size as u64, flags: flags as u32 }) {
        CudaResponse::OccupancyBlocks(b) => { *num_blocks = b; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(num_blocks: *mut c_int, func: CUfunction, block_size: c_int, dynamic_smem_size: usize, flags: c_uint) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_impl(num_blocks, func, block_size, dynamic_smem_size, flags))
}

#[allow(non_snake_case)]
unsafe fn cuOccupancyAvailableDynamicSMemPerBlock_impl(dynamic_smem_size: *mut usize, func: CUfunction, num_blocks: c_int, block_size: c_int) -> CUresult {
    if dynamic_smem_size.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let net_func = match handle_store::get_func(func as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::OccupancyAvailableDynamicSMemPerBlock { func: net_func, num_blocks, block_size }) {
        CudaResponse::OccupancyDynamicSmem(s) => { *dynamic_smem_size = s as usize; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuOccupancyAvailableDynamicSMemPerBlock(dynamic_smem_size: *mut usize, func: CUfunction, num_blocks: c_int, block_size: c_int) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuOccupancyAvailableDynamicSMemPerBlock_impl(dynamic_smem_size, func, num_blocks, block_size))
}
