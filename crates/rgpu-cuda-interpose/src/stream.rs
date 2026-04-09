//! CUDA Stream Management API functions.

use std::ffi::{c_int, c_uint};
use tracing::debug;
use rgpu_protocol::cuda_commands::{CudaCommand, CudaResponse};

use crate::{
    CUresult, CUstream, CUevent, CUcontext,
    CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_READY, CUDA_ERROR_UNKNOWN,
    send_cuda_command, null_stream_handle, handle_store,
};

// ── Stream Management ───────────────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn cuStreamCreate_impl(phstream: *mut CUstream, flags: c_uint) -> CUresult {
    if phstream.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }

    match send_cuda_command(CudaCommand::StreamCreate {
        flags: flags as u32,
    }) {
        CudaResponse::Stream(handle) => {
            let local_id = handle_store::store_stream(handle);
            *phstream = local_id as CUstream;
            debug!("cuStreamCreate -> local_id=0x{:x}", local_id);
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuStreamCreate(phstream: *mut CUstream, flags: c_uint) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuStreamCreate_impl(phstream, flags))
}

#[allow(non_snake_case)]
unsafe fn cuStreamDestroy_v2_impl(hstream: CUstream) -> CUresult {
    let local_id = hstream as u64;
    let net_handle = match handle_store::get_stream(local_id) {
        Some(h) => h,
        None => {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    match send_cuda_command(CudaCommand::StreamDestroy {
        stream: net_handle,
    }) {
        CudaResponse::Success => {
            handle_store::remove_stream(local_id);
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuStreamDestroy_v2(hstream: CUstream) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuStreamDestroy_v2_impl(hstream))
}

#[allow(non_snake_case)]
unsafe fn cuStreamSynchronize_impl(hstream: CUstream) -> CUresult {
    let local_id = hstream as u64;

    let net_handle = if local_id == 0 {
        null_stream_handle()
    } else {
        match handle_store::get_stream(local_id) {
            Some(h) => h,
            None => {
                return CUDA_ERROR_INVALID_VALUE;
            }
        }
    };

    match send_cuda_command(CudaCommand::StreamSynchronize {
        stream: net_handle,
    }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuStreamSynchronize(hstream: CUstream) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuStreamSynchronize_impl(hstream))
}

#[allow(non_snake_case)]
unsafe fn cuStreamQuery_impl(hstream: CUstream) -> CUresult {
    let local_id = hstream as u64;
    let net_handle = if local_id == 0 {
        null_stream_handle()
    } else {
        match handle_store::get_stream(local_id) {
            Some(h) => h,
            None => {
                return CUDA_ERROR_INVALID_VALUE;
            }
        }
    };

    match send_cuda_command(CudaCommand::StreamQuery {
        stream: net_handle,
    }) {
        CudaResponse::StreamStatus(true) => CUDA_SUCCESS,
        CudaResponse::StreamStatus(false) => CUDA_ERROR_NOT_READY,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuStreamQuery(hstream: CUstream) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuStreamQuery_impl(hstream))
}

// ── Stream Management Extended ──────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn cuStreamCreateWithPriority_impl(phstream: *mut CUstream, flags: c_uint, priority: c_int) -> CUresult {
    if phstream.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    match send_cuda_command(CudaCommand::StreamCreateWithPriority { flags: flags as u32, priority }) {
        CudaResponse::Stream(handle) => { let id = handle_store::store_stream(handle); *phstream = id as CUstream; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuStreamCreateWithPriority(phstream: *mut CUstream, flags: c_uint, priority: c_int) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuStreamCreateWithPriority_impl(phstream, flags, priority))
}

#[allow(non_snake_case)]
unsafe fn cuStreamWaitEvent_impl(hstream: CUstream, hevent: CUevent, flags: c_uint) -> CUresult {
    let net_stream = if (hstream as u64) == 0 { null_stream_handle() } else { match handle_store::get_stream(hstream as u64) { Some(h) => h, None => null_stream_handle() } };
    let net_event = match handle_store::get_event(hevent as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::StreamWaitEvent { stream: net_stream, event: net_event, flags: flags as u32 }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuStreamWaitEvent(hstream: CUstream, hevent: CUevent, flags: c_uint) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuStreamWaitEvent_impl(hstream, hevent, flags))
}

#[allow(non_snake_case)]
unsafe fn cuStreamGetPriority_impl(hstream: CUstream, priority: *mut c_int) -> CUresult {
    if priority.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let net_stream = match handle_store::get_stream(hstream as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::StreamGetPriority { stream: net_stream }) {
        CudaResponse::StreamPriority(p) => { *priority = p; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuStreamGetPriority(hstream: CUstream, priority: *mut c_int) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuStreamGetPriority_impl(hstream, priority))
}

#[allow(non_snake_case)]
unsafe fn cuStreamGetFlags_impl(hstream: CUstream, flags: *mut c_uint) -> CUresult {
    if flags.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let net_stream = match handle_store::get_stream(hstream as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::StreamGetFlags { stream: net_stream }) {
        CudaResponse::StreamFlags(f) => { *flags = f; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuStreamGetFlags(hstream: CUstream, flags: *mut c_uint) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuStreamGetFlags_impl(hstream, flags))
}

#[allow(non_snake_case)]
unsafe fn cuStreamGetCtx_v2_impl(hstream: CUstream, pctx: *mut CUcontext) -> CUresult {
    if pctx.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let net_stream = match handle_store::get_stream(hstream as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::StreamGetCtx { stream: net_stream }) {
        CudaResponse::StreamCtx(handle) => { let id = handle_store::store_ctx(handle); *pctx = id as CUcontext; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuStreamGetCtx_v2(hstream: CUstream, pctx: *mut CUcontext) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuStreamGetCtx_v2_impl(hstream, pctx))
}

// ── Unversioned Export Aliases ──────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn cuStreamDestroy_impl(hstream: CUstream) -> CUresult {
    cuStreamDestroy_v2(hstream)
}

#[no_mangle]
pub unsafe extern "C" fn cuStreamDestroy(hstream: CUstream) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuStreamDestroy_impl(hstream))
}

#[allow(non_snake_case)]
unsafe fn cuStreamGetCtx_impl(hstream: CUstream, pctx: *mut CUcontext) -> CUresult {
    cuStreamGetCtx_v2(hstream, pctx)
}

#[no_mangle]
pub unsafe extern "C" fn cuStreamGetCtx(hstream: CUstream, pctx: *mut CUcontext) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuStreamGetCtx_impl(hstream, pctx))
}
