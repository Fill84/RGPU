//! CUDA Event Management API functions.

use std::ffi::c_uint;
use rgpu_protocol::cuda_commands::{CudaCommand, CudaResponse};

use crate::{
    CUresult, CUevent, CUstream,
    CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_READY, CUDA_ERROR_UNKNOWN,
    send_cuda_command, null_stream_handle, handle_store,
};

// ── Event Management ────────────────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn cuEventCreate_impl(phevent: *mut CUevent, flags: c_uint) -> CUresult {
    if phevent.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }

    match send_cuda_command(CudaCommand::EventCreate {
        flags: flags as u32,
    }) {
        CudaResponse::Event(handle) => {
            let local_id = handle_store::store_event(handle);
            *phevent = local_id as CUevent;
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuEventCreate(phevent: *mut CUevent, flags: c_uint) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuEventCreate_impl(phevent, flags))
}

#[allow(non_snake_case)]
unsafe fn cuEventDestroy_v2_impl(hevent: CUevent) -> CUresult {
    let local_id = hevent as u64;
    let net_handle = match handle_store::get_event(local_id) {
        Some(h) => h,
        None => return CUDA_ERROR_INVALID_VALUE,
    };

    match send_cuda_command(CudaCommand::EventDestroy { event: net_handle }) {
        CudaResponse::Success => {
            handle_store::remove_event(local_id);
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuEventDestroy_v2(hevent: CUevent) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuEventDestroy_v2_impl(hevent))
}

#[allow(non_snake_case)]
unsafe fn cuEventRecord_impl(hevent: CUevent, hstream: CUstream) -> CUresult {
    let local_event_id = hevent as u64;
    let net_event = match handle_store::get_event(local_event_id) {
        Some(h) => h,
        None => return CUDA_ERROR_INVALID_VALUE,
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

    match send_cuda_command(CudaCommand::EventRecord {
        event: net_event,
        stream: net_stream,
    }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuEventRecord(hevent: CUevent, hstream: CUstream) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuEventRecord_impl(hevent, hstream))
}

#[allow(non_snake_case)]
unsafe fn cuEventSynchronize_impl(hevent: CUevent) -> CUresult {
    let local_id = hevent as u64;
    let net_handle = match handle_store::get_event(local_id) {
        Some(h) => h,
        None => return CUDA_ERROR_INVALID_VALUE,
    };

    match send_cuda_command(CudaCommand::EventSynchronize { event: net_handle }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuEventSynchronize(hevent: CUevent) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuEventSynchronize_impl(hevent))
}

#[allow(non_snake_case)]
unsafe fn cuEventQuery_impl(hevent: CUevent) -> CUresult {
    let local_id = hevent as u64;
    let net_handle = match handle_store::get_event(local_id) {
        Some(h) => h,
        None => return CUDA_ERROR_INVALID_VALUE,
    };

    match send_cuda_command(CudaCommand::EventQuery { event: net_handle }) {
        CudaResponse::EventStatus(true) => CUDA_SUCCESS,
        CudaResponse::EventStatus(false) => CUDA_ERROR_NOT_READY,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuEventQuery(hevent: CUevent) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuEventQuery_impl(hevent))
}

#[allow(non_snake_case)]
unsafe fn cuEventElapsedTime_impl(
    ms: *mut f32,
    hstart: CUevent,
    hend: CUevent,
) -> CUresult {
    if ms.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }

    let net_start = match handle_store::get_event(hstart as u64) {
        Some(h) => h,
        None => return CUDA_ERROR_INVALID_VALUE,
    };
    let net_end = match handle_store::get_event(hend as u64) {
        Some(h) => h,
        None => return CUDA_ERROR_INVALID_VALUE,
    };

    match send_cuda_command(CudaCommand::EventElapsedTime {
        start: net_start,
        end: net_end,
    }) {
        CudaResponse::ElapsedTime(elapsed) => {
            *ms = elapsed;
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuEventElapsedTime(
    ms: *mut f32,
    hstart: CUevent,
    hend: CUevent,
) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuEventElapsedTime_impl(ms, hstart, hend))
}

// ── Event Management Extended ───────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn cuEventRecordWithFlags_impl(hevent: CUevent, hstream: CUstream, flags: c_uint) -> CUresult {
    let net_event = match handle_store::get_event(hevent as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    let net_stream = if (hstream as u64) == 0 { null_stream_handle() } else { handle_store::get_stream(hstream as u64).unwrap_or_else(null_stream_handle) };
    match send_cuda_command(CudaCommand::EventRecordWithFlags { event: net_event, stream: net_stream, flags: flags as u32 }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuEventRecordWithFlags(hevent: CUevent, hstream: CUstream, flags: c_uint) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuEventRecordWithFlags_impl(hevent, hstream, flags))
}

// ── Unversioned Export Alias ───────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn cuEventDestroy_impl(hevent: CUevent) -> CUresult {
    cuEventDestroy_v2(hevent)
}

#[no_mangle]
pub unsafe extern "C" fn cuEventDestroy(hevent: CUevent) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuEventDestroy_impl(hevent))
}
