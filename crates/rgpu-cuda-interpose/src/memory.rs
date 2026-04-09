//! CUDA Memory Management API functions.

use std::ffi::{c_int, c_uint, c_void};
use tracing::{debug, error};
use rgpu_protocol::cuda_commands::{CudaCommand, CudaResponse};

use crate::{
    CUresult, CUdeviceptr, CUstream, CUmemoryPool,
    CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_UNKNOWN,
    send_cuda_command, null_stream_handle, handle_store,
};

// ── Memory Management ───────────────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn cuMemAlloc_v2_impl(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult {
    if dptr.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }

    debug!("cuMemAlloc_v2({} bytes)", bytesize);

    match send_cuda_command(CudaCommand::MemAlloc {
        byte_size: bytesize as u64,
    }) {
        CudaResponse::MemAllocated(handle) => {
            let local_id = handle_store::store_mem(handle);
            *dptr = local_id;
            debug!("cuMemAlloc_v2 -> local_id=0x{:x}", local_id);
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemAlloc_v2_impl(dptr, bytesize))
}

#[allow(non_snake_case)]
unsafe fn cuMemFree_v2_impl(dptr: CUdeviceptr) -> CUresult {
    let net_handle = match handle_store::get_mem_by_ptr(dptr) {
        Some(h) => h,
        None => {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    match send_cuda_command(CudaCommand::MemFree { dptr: net_handle }) {
        CudaResponse::Success => {
            handle_store::remove_mem(dptr);
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemFree_v2_impl(dptr))
}

#[allow(non_snake_case)]
unsafe fn cuMemcpyHtoD_v2_impl(
    dst_device: CUdeviceptr,
    src_host: *const c_void,
    byte_count: usize,
) -> CUresult {
    if src_host.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }

    let net_dst = match handle_store::get_mem_by_ptr(dst_device) {
        Some(h) => h,
        None => {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    let src_data = std::slice::from_raw_parts(src_host as *const u8, byte_count).to_vec();

    debug!("cuMemcpyHtoD_v2({} bytes)", byte_count);

    match send_cuda_command(CudaCommand::MemcpyHtoD {
        dst: net_dst,
        src_data,
        byte_count: byte_count as u64,
    }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemcpyHtoD_v2(
    dst_device: CUdeviceptr,
    src_host: *const c_void,
    byte_count: usize,
) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemcpyHtoD_v2_impl(dst_device, src_host, byte_count))
}

#[allow(non_snake_case)]
unsafe fn cuMemcpyDtoH_v2_impl(
    dst_host: *mut c_void,
    src_device: CUdeviceptr,
    byte_count: usize,
) -> CUresult {
    if dst_host.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }

    let net_src = match handle_store::get_mem_by_ptr(src_device) {
        Some(h) => h,
        None => {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    debug!("cuMemcpyDtoH_v2({} bytes)", byte_count);

    match send_cuda_command(CudaCommand::MemcpyDtoH {
        src: net_src,
        byte_count: byte_count as u64,
    }) {
        CudaResponse::MemoryData(data) => {
            let copy_len = std::cmp::min(data.len(), byte_count);
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst_host as *mut u8, copy_len);
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemcpyDtoH_v2(
    dst_host: *mut c_void,
    src_device: CUdeviceptr,
    byte_count: usize,
) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemcpyDtoH_v2_impl(dst_host, src_device, byte_count))
}

// ── Memory Management Extended ──────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn cuMemcpy_impl(dst: CUdeviceptr, src: CUdeviceptr, byte_count: usize) -> CUresult {
    let net_dst = match handle_store::get_mem_by_ptr(dst) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    let net_src = match handle_store::get_mem_by_ptr(src) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::MemcpyDtoD { dst: net_dst, src: net_src, byte_count: byte_count as u64 }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemcpy(dst: CUdeviceptr, src: CUdeviceptr, byte_count: usize) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemcpy_impl(dst, src, byte_count))
}

#[allow(non_snake_case)]
unsafe fn cuMemcpyDtoD_v2_impl(dst: CUdeviceptr, src: CUdeviceptr, byte_count: usize) -> CUresult {
    let net_dst = match handle_store::get_mem_by_ptr(dst) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    let net_src = match handle_store::get_mem_by_ptr(src) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::MemcpyDtoD { dst: net_dst, src: net_src, byte_count: byte_count as u64 }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemcpyDtoD_v2(dst: CUdeviceptr, src: CUdeviceptr, byte_count: usize) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemcpyDtoD_v2_impl(dst, src, byte_count))
}

#[allow(non_snake_case)]
unsafe fn cuMemcpyHtoDAsync_v2_impl(dst: CUdeviceptr, src: *const c_void, byte_count: usize, hstream: CUstream) -> CUresult {
    if src.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let net_dst = match handle_store::get_mem_by_ptr(dst) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    let net_stream = if (hstream as u64) == 0 { null_stream_handle() } else { handle_store::get_stream(hstream as u64).unwrap_or_else(null_stream_handle) };
    let src_data = std::slice::from_raw_parts(src as *const u8, byte_count).to_vec();
    match send_cuda_command(CudaCommand::MemcpyHtoDAsync { dst: net_dst, src_data, byte_count: byte_count as u64, stream: net_stream }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemcpyHtoDAsync_v2(dst: CUdeviceptr, src: *const c_void, byte_count: usize, hstream: CUstream) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemcpyHtoDAsync_v2_impl(dst, src, byte_count, hstream))
}

#[allow(non_snake_case)]
unsafe fn cuMemcpyDtoHAsync_v2_impl(dst: *mut c_void, src: CUdeviceptr, byte_count: usize, hstream: CUstream) -> CUresult {
    if dst.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let net_src = match handle_store::get_mem_by_ptr(src) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    let net_stream = if (hstream as u64) == 0 { null_stream_handle() } else { handle_store::get_stream(hstream as u64).unwrap_or_else(null_stream_handle) };
    match send_cuda_command(CudaCommand::MemcpyDtoHAsync { src: net_src, byte_count: byte_count as u64, stream: net_stream }) {
        CudaResponse::MemoryData(data) => {
            let copy_len = std::cmp::min(data.len(), byte_count);
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst as *mut u8, copy_len);
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemcpyDtoHAsync_v2(dst: *mut c_void, src: CUdeviceptr, byte_count: usize, hstream: CUstream) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemcpyDtoHAsync_v2_impl(dst, src, byte_count, hstream))
}

#[allow(non_snake_case)]
unsafe fn cuMemcpyAsync_impl(dst: CUdeviceptr, src: CUdeviceptr, byte_count: usize, hstream: CUstream) -> CUresult {
    let net_dst = match handle_store::get_mem_by_ptr(dst) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    let net_src = match handle_store::get_mem_by_ptr(src) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    let net_stream = if (hstream as u64) == 0 { null_stream_handle() } else { handle_store::get_stream(hstream as u64).unwrap_or_else(null_stream_handle) };
    match send_cuda_command(CudaCommand::MemcpyDtoDAsync { dst: net_dst, src: net_src, byte_count: byte_count as u64, stream: net_stream }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemcpyAsync(dst: CUdeviceptr, src: CUdeviceptr, byte_count: usize, hstream: CUstream) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemcpyAsync_impl(dst, src, byte_count, hstream))
}

#[allow(non_snake_case)]
unsafe fn cuMemcpyDtoDAsync_v2_impl(dst: CUdeviceptr, src: CUdeviceptr, byte_count: usize, hstream: CUstream) -> CUresult {
    let net_dst = match handle_store::get_mem_by_ptr(dst) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    let net_src = match handle_store::get_mem_by_ptr(src) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    let net_stream = if (hstream as u64) == 0 { null_stream_handle() } else { handle_store::get_stream(hstream as u64).unwrap_or_else(null_stream_handle) };
    match send_cuda_command(CudaCommand::MemcpyDtoDAsync { dst: net_dst, src: net_src, byte_count: byte_count as u64, stream: net_stream }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemcpyDtoDAsync_v2(dst: CUdeviceptr, src: CUdeviceptr, byte_count: usize, hstream: CUstream) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemcpyDtoDAsync_v2_impl(dst, src, byte_count, hstream))
}

// ── 2D Memory Copy ─────────────────────────────────────────────────

/// CUDA_MEMCPY2D struct for 2D memory copy operations.
#[repr(C)]
pub struct CUDA_MEMCPY2D {
    src_x_in_bytes: usize,
    src_y: usize,
    src_memory_type: u32, // CU_MEMORYTYPE_*
    src_host: *const c_void,
    src_device: CUdeviceptr,
    src_array: *mut c_void,
    src_pitch: usize,
    dst_x_in_bytes: usize,
    dst_y: usize,
    dst_memory_type: u32,
    dst_host: *mut c_void,
    dst_device: CUdeviceptr,
    dst_array: *mut c_void,
    dst_pitch: usize,
    width_in_bytes: usize,
    height: usize,
}

const CU_MEMORYTYPE_HOST: u32 = 1;
const CU_MEMORYTYPE_DEVICE: u32 = 2;

#[allow(non_snake_case)]
unsafe fn cuMemcpy2D_v2_impl(p_copy: *const CUDA_MEMCPY2D) -> CUresult {
    if p_copy.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let c = &*p_copy;

    match (c.src_memory_type, c.dst_memory_type) {
        (CU_MEMORYTYPE_HOST, CU_MEMORYTYPE_DEVICE) => {
            // Host to Device: pack all row data and send as single HtoD
            let net_dst = match handle_store::get_mem_by_ptr(c.dst_device) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
            let mut packed = Vec::with_capacity(c.width_in_bytes * c.height);
            for row in 0..c.height {
                let src_ptr = (c.src_host as *const u8).add(
                    (c.src_y + row) * c.src_pitch + c.src_x_in_bytes,
                );
                packed.extend_from_slice(std::slice::from_raw_parts(src_ptr, c.width_in_bytes));
            }
            // Send as a pitched HtoD copy: we send packed data + metadata
            // The server will unpack based on dst_pitch
            match send_cuda_command(CudaCommand::Memcpy2DHtoD {
                dst: net_dst,
                dst_x_in_bytes: c.dst_x_in_bytes as u64,
                dst_y: c.dst_y as u64,
                dst_pitch: c.dst_pitch as u64,
                src_data: packed,
                width_in_bytes: c.width_in_bytes as u64,
                height: c.height as u64,
            }) {
                CudaResponse::Success => CUDA_SUCCESS,
                CudaResponse::Error { code, .. } => code,
                _ => CUDA_ERROR_UNKNOWN,
            }
        }
        (CU_MEMORYTYPE_DEVICE, CU_MEMORYTYPE_HOST) => {
            // Device to Host: fetch all data and unpack row by row
            let net_src = match handle_store::get_mem_by_ptr(c.src_device) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
            match send_cuda_command(CudaCommand::Memcpy2DDtoH {
                src: net_src,
                src_x_in_bytes: c.src_x_in_bytes as u64,
                src_y: c.src_y as u64,
                src_pitch: c.src_pitch as u64,
                width_in_bytes: c.width_in_bytes as u64,
                height: c.height as u64,
            }) {
                CudaResponse::MemoryData(data) => {
                    // data is packed row-major (width_in_bytes * height)
                    for row in 0..c.height {
                        let dst_ptr = (c.dst_host as *mut u8).add(
                            (c.dst_y + row) * c.dst_pitch + c.dst_x_in_bytes,
                        );
                        let src_offset = row * c.width_in_bytes;
                        std::ptr::copy_nonoverlapping(
                            data.as_ptr().add(src_offset),
                            dst_ptr,
                            c.width_in_bytes,
                        );
                    }
                    CUDA_SUCCESS
                }
                CudaResponse::Error { code, .. } => code,
                _ => CUDA_ERROR_UNKNOWN,
            }
        }
        (CU_MEMORYTYPE_DEVICE, CU_MEMORYTYPE_DEVICE) => {
            // Device to Device: use existing DtoD
            let net_dst = match handle_store::get_mem_by_ptr(c.dst_device) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
            let net_src = match handle_store::get_mem_by_ptr(c.src_device) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
            match send_cuda_command(CudaCommand::Memcpy2DDtoD {
                dst: net_dst,
                dst_x_in_bytes: c.dst_x_in_bytes as u64,
                dst_y: c.dst_y as u64,
                dst_pitch: c.dst_pitch as u64,
                src: net_src,
                src_x_in_bytes: c.src_x_in_bytes as u64,
                src_y: c.src_y as u64,
                src_pitch: c.src_pitch as u64,
                width_in_bytes: c.width_in_bytes as u64,
                height: c.height as u64,
            }) {
                CudaResponse::Success => CUDA_SUCCESS,
                CudaResponse::Error { code, .. } => code,
                _ => CUDA_ERROR_UNKNOWN,
            }
        }
        _ => {
            error!("cuMemcpy2D: unsupported memory type combination src={} dst={}", c.src_memory_type, c.dst_memory_type);
            CUDA_ERROR_INVALID_VALUE
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemcpy2D_v2(p_copy: *const CUDA_MEMCPY2D) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemcpy2D_v2_impl(p_copy))
}

#[allow(non_snake_case)]
unsafe fn cuMemcpy2DAsync_v2_impl(p_copy: *const CUDA_MEMCPY2D, _hstream: CUstream) -> CUresult {
    // For now, implement as synchronous — the server will handle it synchronously anyway
    cuMemcpy2D_v2(p_copy)
}

#[no_mangle]
pub unsafe extern "C" fn cuMemcpy2DAsync_v2(p_copy: *const CUDA_MEMCPY2D, _hstream: CUstream) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemcpy2DAsync_v2_impl(p_copy, _hstream))
}

#[allow(non_snake_case)]
unsafe fn cuMemsetD8_v2_impl(dst: CUdeviceptr, value: u8, count: usize) -> CUresult {
    let net_dst = match handle_store::get_mem_by_ptr(dst) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::MemsetD8 { dst: net_dst, value, count: count as u64 }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemsetD8_v2(dst: CUdeviceptr, value: u8, count: usize) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemsetD8_v2_impl(dst, value, count))
}

#[allow(non_snake_case)]
unsafe fn cuMemsetD16_v2_impl(dst: CUdeviceptr, value: u16, count: usize) -> CUresult {
    let net_dst = match handle_store::get_mem_by_ptr(dst) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::MemsetD16 { dst: net_dst, value, count: count as u64 }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemsetD16_v2(dst: CUdeviceptr, value: u16, count: usize) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemsetD16_v2_impl(dst, value, count))
}

#[allow(non_snake_case)]
unsafe fn cuMemsetD32_v2_impl(dst: CUdeviceptr, value: u32, count: usize) -> CUresult {
    let net_dst = match handle_store::get_mem_by_ptr(dst) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::MemsetD32 { dst: net_dst, value, count: count as u64 }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemsetD32_v2(dst: CUdeviceptr, value: u32, count: usize) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemsetD32_v2_impl(dst, value, count))
}

#[allow(non_snake_case)]
unsafe fn cuMemsetD8Async_impl(dst: CUdeviceptr, value: u8, count: usize, hstream: CUstream) -> CUresult {
    let net_dst = match handle_store::get_mem_by_ptr(dst) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    let net_stream = if (hstream as u64) == 0 { null_stream_handle() } else { handle_store::get_stream(hstream as u64).unwrap_or_else(null_stream_handle) };
    match send_cuda_command(CudaCommand::MemsetD8Async { dst: net_dst, value, count: count as u64, stream: net_stream }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemsetD8Async(dst: CUdeviceptr, value: u8, count: usize, hstream: CUstream) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemsetD8Async_impl(dst, value, count, hstream))
}

#[allow(non_snake_case)]
unsafe fn cuMemsetD16Async_impl(dst: CUdeviceptr, value: u16, count: usize, hstream: CUstream) -> CUresult {
    let net_dst = match handle_store::get_mem_by_ptr(dst) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    let net_stream = if (hstream as u64) == 0 { null_stream_handle() } else { handle_store::get_stream(hstream as u64).unwrap_or_else(null_stream_handle) };
    match send_cuda_command(CudaCommand::MemsetD16Async { dst: net_dst, value, count: count as u64, stream: net_stream }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemsetD16Async(dst: CUdeviceptr, value: u16, count: usize, hstream: CUstream) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemsetD16Async_impl(dst, value, count, hstream))
}

#[allow(non_snake_case)]
unsafe fn cuMemsetD32Async_impl(dst: CUdeviceptr, value: u32, count: usize, hstream: CUstream) -> CUresult {
    let net_dst = match handle_store::get_mem_by_ptr(dst) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    let net_stream = if (hstream as u64) == 0 { null_stream_handle() } else { handle_store::get_stream(hstream as u64).unwrap_or_else(null_stream_handle) };
    match send_cuda_command(CudaCommand::MemsetD32Async { dst: net_dst, value, count: count as u64, stream: net_stream }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemsetD32Async(dst: CUdeviceptr, value: u32, count: usize, hstream: CUstream) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemsetD32Async_impl(dst, value, count, hstream))
}

#[allow(non_snake_case)]
unsafe fn cuMemGetInfo_v2_impl(free: *mut usize, total: *mut usize) -> CUresult {
    match send_cuda_command(CudaCommand::MemGetInfo) {
        CudaResponse::MemInfo { free: f, total: t } => {
            if !free.is_null() { *free = f as usize; }
            if !total.is_null() { *total = t as usize; }
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemGetInfo_v2_impl(free, total))
}

#[allow(non_snake_case)]
unsafe fn cuMemGetAddressRange_v2_impl(pbase: *mut CUdeviceptr, psize: *mut usize, dptr: CUdeviceptr) -> CUresult {
    let net_ptr = match handle_store::get_mem_by_ptr(dptr) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::MemGetAddressRange { dptr: net_ptr }) {
        CudaResponse::MemAddressRange { base, size } => {
            if !pbase.is_null() { let id = handle_store::store_mem(base); *pbase = id; }
            if !psize.is_null() { *psize = size as usize; }
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemGetAddressRange_v2(pbase: *mut CUdeviceptr, psize: *mut usize, dptr: CUdeviceptr) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemGetAddressRange_v2_impl(pbase, psize, dptr))
}

#[allow(non_snake_case)]
unsafe fn cuMemAllocHost_v2_impl(pp: *mut *mut c_void, bytesize: usize) -> CUresult {
    if pp.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    match send_cuda_command(CudaCommand::MemAllocHost { byte_size: bytesize as u64 }) {
        CudaResponse::HostPtr(handle) => {
            let id = handle_store::store_host_mem(handle);
            *pp = id as *mut c_void;
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemAllocHost_v2(pp: *mut *mut c_void, bytesize: usize) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemAllocHost_v2_impl(pp, bytesize))
}

#[allow(non_snake_case)]
unsafe fn cuMemFreeHost_impl(p: *mut c_void) -> CUresult {
    let net_h = match handle_store::get_host_mem(p as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::MemFreeHost { ptr: net_h }) {
        CudaResponse::Success => { handle_store::remove_host_mem(p as u64); CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemFreeHost(p: *mut c_void) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemFreeHost_impl(p))
}

#[allow(non_snake_case)]
unsafe fn cuMemHostAlloc_impl(pp: *mut *mut c_void, bytesize: usize, flags: c_uint) -> CUresult {
    if pp.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    match send_cuda_command(CudaCommand::MemHostAlloc { byte_size: bytesize as u64, flags: flags as u32 }) {
        CudaResponse::HostPtr(handle) => {
            let id = handle_store::store_host_mem(handle);
            *pp = id as *mut c_void;
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemHostAlloc(pp: *mut *mut c_void, bytesize: usize, flags: c_uint) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemHostAlloc_impl(pp, bytesize, flags))
}

#[allow(non_snake_case)]
unsafe fn cuMemHostGetDevicePointer_v2_impl(pdptr: *mut CUdeviceptr, p: *mut c_void, flags: c_uint) -> CUresult {
    if pdptr.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let net_h = match handle_store::get_host_mem(p as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::MemHostGetDevicePointer { host_ptr: net_h, flags: flags as u32 }) {
        CudaResponse::HostDevicePtr(handle) => { let id = handle_store::store_mem(handle); *pdptr = id; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemHostGetDevicePointer_v2(pdptr: *mut CUdeviceptr, p: *mut c_void, flags: c_uint) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemHostGetDevicePointer_v2_impl(pdptr, p, flags))
}

#[allow(non_snake_case)]
unsafe fn cuMemHostGetFlags_impl(pflags: *mut c_uint, p: *mut c_void) -> CUresult {
    if pflags.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let net_h = match handle_store::get_host_mem(p as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::MemHostGetFlags { host_ptr: net_h }) {
        CudaResponse::HostFlags(f) => { *pflags = f; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemHostGetFlags(pflags: *mut c_uint, p: *mut c_void) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemHostGetFlags_impl(pflags, p))
}

#[allow(non_snake_case)]
unsafe fn cuMemAllocManaged_impl(dptr: *mut CUdeviceptr, bytesize: usize, flags: c_uint) -> CUresult {
    if dptr.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    match send_cuda_command(CudaCommand::MemAllocManaged { byte_size: bytesize as u64, flags: flags as u32 }) {
        CudaResponse::MemAllocated(handle) => { let id = handle_store::store_mem(handle); *dptr = id; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemAllocManaged(dptr: *mut CUdeviceptr, bytesize: usize, flags: c_uint) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemAllocManaged_impl(dptr, bytesize, flags))
}

#[allow(non_snake_case)]
unsafe fn cuMemAllocPitch_v2_impl(dptr: *mut CUdeviceptr, ppitch: *mut usize, width: usize, height: usize, element_size: c_uint) -> CUresult {
    if dptr.is_null() || ppitch.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    match send_cuda_command(CudaCommand::MemAllocPitch { width: width as u64, height: height as u64, element_size }) {
        CudaResponse::MemAllocPitch { dptr: handle, pitch } => {
            let id = handle_store::store_mem(handle);
            *dptr = id;
            *ppitch = pitch as usize;
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemAllocPitch_v2(dptr: *mut CUdeviceptr, ppitch: *mut usize, width: usize, height: usize, element_size: c_uint) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemAllocPitch_v2_impl(dptr, ppitch, width, height, element_size))
}

// ── Pointer Queries ─────────────────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn cuPointerGetAttribute_impl(data: *mut c_void, attribute: c_int, ptr: CUdeviceptr) -> CUresult {
    if data.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let net_ptr = match handle_store::get_mem_by_ptr(ptr) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::PointerGetAttribute { attribute, ptr: net_ptr }) {
        CudaResponse::PointerAttribute(v) => { *(data as *mut u64) = v; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuPointerGetAttribute(data: *mut c_void, attribute: c_int, ptr: CUdeviceptr) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuPointerGetAttribute_impl(data, attribute, ptr))
}

#[allow(non_snake_case)]
unsafe fn cuPointerSetAttribute_impl(value: *const c_void, attribute: c_int, ptr: CUdeviceptr) -> CUresult {
    let net_ptr = match handle_store::get_mem_by_ptr(ptr) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    let val = if !value.is_null() { *(value as *const u64) } else { 0 };
    match send_cuda_command(CudaCommand::PointerSetAttribute { attribute, ptr: net_ptr, value: val }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuPointerSetAttribute(value: *const c_void, attribute: c_int, ptr: CUdeviceptr) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuPointerSetAttribute_impl(value, attribute, ptr))
}

// ── Memory Pools ────────────────────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn cuMemPoolDestroy_impl(pool: CUmemoryPool) -> CUresult {
    let net_pool = match handle_store::get_mempool(pool as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::MemPoolDestroy { pool: net_pool }) {
        CudaResponse::Success => { handle_store::remove_mempool(pool as u64); CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemPoolDestroy(pool: CUmemoryPool) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemPoolDestroy_impl(pool))
}

#[allow(non_snake_case)]
unsafe fn cuMemPoolTrimTo_impl(pool: CUmemoryPool, min_bytes_to_keep: usize) -> CUresult {
    let net_pool = match handle_store::get_mempool(pool as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::MemPoolTrimTo { pool: net_pool, min_bytes_to_keep: min_bytes_to_keep as u64 }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemPoolTrimTo(pool: CUmemoryPool, min_bytes_to_keep: usize) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemPoolTrimTo_impl(pool, min_bytes_to_keep))
}

#[allow(non_snake_case)]
unsafe fn cuMemAllocAsync_impl(dptr: *mut CUdeviceptr, bytesize: usize, hstream: CUstream) -> CUresult {
    if dptr.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let net_stream = if (hstream as u64) == 0 { null_stream_handle() } else { handle_store::get_stream(hstream as u64).unwrap_or_else(null_stream_handle) };
    match send_cuda_command(CudaCommand::MemAllocAsync { byte_size: bytesize as u64, stream: net_stream }) {
        CudaResponse::MemAllocated(handle) => { let id = handle_store::store_mem(handle); *dptr = id; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemAllocAsync(dptr: *mut CUdeviceptr, bytesize: usize, hstream: CUstream) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemAllocAsync_impl(dptr, bytesize, hstream))
}

#[allow(non_snake_case)]
unsafe fn cuMemFreeAsync_impl(dptr: CUdeviceptr, hstream: CUstream) -> CUresult {
    let net_ptr = match handle_store::get_mem_by_ptr(dptr) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    let net_stream = if (hstream as u64) == 0 { null_stream_handle() } else { handle_store::get_stream(hstream as u64).unwrap_or_else(null_stream_handle) };
    match send_cuda_command(CudaCommand::MemFreeAsync { dptr: net_ptr, stream: net_stream }) {
        CudaResponse::Success => { handle_store::remove_mem(dptr); CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemFreeAsync(dptr: CUdeviceptr, hstream: CUstream) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemFreeAsync_impl(dptr, hstream))
}

#[allow(non_snake_case)]
unsafe fn cuMemAllocFromPoolAsync_impl(dptr: *mut CUdeviceptr, bytesize: usize, pool: CUmemoryPool, hstream: CUstream) -> CUresult {
    if dptr.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let net_pool = match handle_store::get_mempool(pool as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    let net_stream = if (hstream as u64) == 0 { null_stream_handle() } else { handle_store::get_stream(hstream as u64).unwrap_or_else(null_stream_handle) };
    match send_cuda_command(CudaCommand::MemAllocFromPoolAsync { byte_size: bytesize as u64, pool: net_pool, stream: net_stream }) {
        CudaResponse::MemAllocated(handle) => { let id = handle_store::store_mem(handle); *dptr = id; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuMemAllocFromPoolAsync(dptr: *mut CUdeviceptr, bytesize: usize, pool: CUmemoryPool, hstream: CUstream) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemAllocFromPoolAsync_impl(dptr, bytesize, pool, hstream))
}

// ── Unversioned Export Aliases ──────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn cuMemAlloc_impl(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult {
    cuMemAlloc_v2(dptr, bytesize)
}

#[no_mangle]
pub unsafe extern "C" fn cuMemAlloc(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemAlloc_impl(dptr, bytesize))
}

#[allow(non_snake_case)]
unsafe fn cuMemFree_impl(dptr: CUdeviceptr) -> CUresult {
    cuMemFree_v2(dptr)
}

#[no_mangle]
pub unsafe extern "C" fn cuMemFree(dptr: CUdeviceptr) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemFree_impl(dptr))
}

#[allow(non_snake_case)]
unsafe fn cuMemcpyHtoD_impl(dst: CUdeviceptr, src: *const c_void, byte_count: usize) -> CUresult {
    cuMemcpyHtoD_v2(dst, src, byte_count)
}

#[no_mangle]
pub unsafe extern "C" fn cuMemcpyHtoD(dst: CUdeviceptr, src: *const c_void, byte_count: usize) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemcpyHtoD_impl(dst, src, byte_count))
}

#[allow(non_snake_case)]
unsafe fn cuMemcpyDtoH_impl(dst: *mut c_void, src: CUdeviceptr, byte_count: usize) -> CUresult {
    cuMemcpyDtoH_v2(dst, src, byte_count)
}

#[no_mangle]
pub unsafe extern "C" fn cuMemcpyDtoH(dst: *mut c_void, src: CUdeviceptr, byte_count: usize) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemcpyDtoH_impl(dst, src, byte_count))
}

#[allow(non_snake_case)]
unsafe fn cuMemcpyDtoD_impl(dst: CUdeviceptr, src: CUdeviceptr, byte_count: usize) -> CUresult {
    cuMemcpyDtoD_v2(dst, src, byte_count)
}

#[no_mangle]
pub unsafe extern "C" fn cuMemcpyDtoD(dst: CUdeviceptr, src: CUdeviceptr, byte_count: usize) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemcpyDtoD_impl(dst, src, byte_count))
}

#[allow(non_snake_case)]
unsafe fn cuMemcpyHtoDAsync_impl(dst: CUdeviceptr, src: *const c_void, byte_count: usize, hstream: CUstream) -> CUresult {
    cuMemcpyHtoDAsync_v2(dst, src, byte_count, hstream)
}

#[no_mangle]
pub unsafe extern "C" fn cuMemcpyHtoDAsync(dst: CUdeviceptr, src: *const c_void, byte_count: usize, hstream: CUstream) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemcpyHtoDAsync_impl(dst, src, byte_count, hstream))
}

#[allow(non_snake_case)]
unsafe fn cuMemcpyDtoHAsync_impl(dst: *mut c_void, src: CUdeviceptr, byte_count: usize, hstream: CUstream) -> CUresult {
    cuMemcpyDtoHAsync_v2(dst, src, byte_count, hstream)
}

#[no_mangle]
pub unsafe extern "C" fn cuMemcpyDtoHAsync(dst: *mut c_void, src: CUdeviceptr, byte_count: usize, hstream: CUstream) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemcpyDtoHAsync_impl(dst, src, byte_count, hstream))
}

#[allow(non_snake_case)]
unsafe fn cuMemcpyDtoDAsync_impl(dst: CUdeviceptr, src: CUdeviceptr, byte_count: usize, hstream: CUstream) -> CUresult {
    cuMemcpyDtoDAsync_v2(dst, src, byte_count, hstream)
}

#[no_mangle]
pub unsafe extern "C" fn cuMemcpyDtoDAsync(dst: CUdeviceptr, src: CUdeviceptr, byte_count: usize, hstream: CUstream) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemcpyDtoDAsync_impl(dst, src, byte_count, hstream))
}

#[allow(non_snake_case)]
unsafe fn cuMemcpy2D_impl(p_copy: *const CUDA_MEMCPY2D) -> CUresult {
    cuMemcpy2D_v2(p_copy)
}

#[no_mangle]
pub unsafe extern "C" fn cuMemcpy2D(p_copy: *const CUDA_MEMCPY2D) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemcpy2D_impl(p_copy))
}

#[allow(non_snake_case)]
unsafe fn cuMemcpy2DAsync_impl(p_copy: *const CUDA_MEMCPY2D, hstream: CUstream) -> CUresult {
    cuMemcpy2DAsync_v2(p_copy, hstream)
}

#[no_mangle]
pub unsafe extern "C" fn cuMemcpy2DAsync(p_copy: *const CUDA_MEMCPY2D, hstream: CUstream) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemcpy2DAsync_impl(p_copy, hstream))
}

#[allow(non_snake_case)]
unsafe fn cuMemsetD8_impl(dst: CUdeviceptr, value: u8, count: usize) -> CUresult {
    cuMemsetD8_v2(dst, value, count)
}

#[no_mangle]
pub unsafe extern "C" fn cuMemsetD8(dst: CUdeviceptr, value: u8, count: usize) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemsetD8_impl(dst, value, count))
}

#[allow(non_snake_case)]
unsafe fn cuMemsetD16_impl(dst: CUdeviceptr, value: u16, count: usize) -> CUresult {
    cuMemsetD16_v2(dst, value, count)
}

#[no_mangle]
pub unsafe extern "C" fn cuMemsetD16(dst: CUdeviceptr, value: u16, count: usize) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemsetD16_impl(dst, value, count))
}

#[allow(non_snake_case)]
unsafe fn cuMemsetD32_impl(dst: CUdeviceptr, value: u32, count: usize) -> CUresult {
    cuMemsetD32_v2(dst, value, count)
}

#[no_mangle]
pub unsafe extern "C" fn cuMemsetD32(dst: CUdeviceptr, value: u32, count: usize) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemsetD32_impl(dst, value, count))
}

#[allow(non_snake_case)]
unsafe fn cuMemGetInfo_impl(free: *mut usize, total: *mut usize) -> CUresult {
    cuMemGetInfo_v2(free, total)
}

#[no_mangle]
pub unsafe extern "C" fn cuMemGetInfo(free: *mut usize, total: *mut usize) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemGetInfo_impl(free, total))
}

#[allow(non_snake_case)]
unsafe fn cuMemGetAddressRange_impl(pbase: *mut CUdeviceptr, psize: *mut usize, dptr: CUdeviceptr) -> CUresult {
    cuMemGetAddressRange_v2(pbase, psize, dptr)
}

#[no_mangle]
pub unsafe extern "C" fn cuMemGetAddressRange(pbase: *mut CUdeviceptr, psize: *mut usize, dptr: CUdeviceptr) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemGetAddressRange_impl(pbase, psize, dptr))
}

#[allow(non_snake_case)]
unsafe fn cuMemAllocHost_impl(pp: *mut *mut c_void, bytesize: usize) -> CUresult {
    cuMemAllocHost_v2(pp, bytesize)
}

#[no_mangle]
pub unsafe extern "C" fn cuMemAllocHost(pp: *mut *mut c_void, bytesize: usize) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemAllocHost_impl(pp, bytesize))
}

#[allow(non_snake_case)]
unsafe fn cuMemHostGetDevicePointer_impl(pdptr: *mut CUdeviceptr, p: *mut c_void, flags: c_uint) -> CUresult {
    cuMemHostGetDevicePointer_v2(pdptr, p, flags)
}

#[no_mangle]
pub unsafe extern "C" fn cuMemHostGetDevicePointer(pdptr: *mut CUdeviceptr, p: *mut c_void, flags: c_uint) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemHostGetDevicePointer_impl(pdptr, p, flags))
}

#[allow(non_snake_case)]
unsafe fn cuMemAllocPitch_impl(dptr: *mut CUdeviceptr, ppitch: *mut usize, width: usize, height: usize, element_size: c_uint) -> CUresult {
    cuMemAllocPitch_v2(dptr, ppitch, width, height, element_size)
}

#[no_mangle]
pub unsafe extern "C" fn cuMemAllocPitch(dptr: *mut CUdeviceptr, ppitch: *mut usize, width: usize, height: usize, element_size: c_uint) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuMemAllocPitch_impl(dptr, ppitch, width, height, element_size))
}
