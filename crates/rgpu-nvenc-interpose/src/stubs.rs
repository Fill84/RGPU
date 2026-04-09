//! Stub and passthrough interpose functions.
//!
//! Contains: get_last_error_string, set_io_cuda_streams, destroy_encoder,
//! invalidate_ref_frames, reconfigure_encoder, get_sequence_params,
//! register/unregister_async_event, get_sequence_param_payload,
//! get_encode_stats, run_motion_estimation_only, create/destroy_mv_buffer,
//! lookahead_picture, collect_sequence_stats.

use std::ffi::c_void;

use tracing::debug;

use rgpu_protocol::nvenc_commands::{NvencCommand, NvencResponse};

use crate::handle_store;
use crate::types::NvEncSequenceParamPayload;
use crate::{
    encoder_handle_from_ptr, response_to_status, send_nvenc_command, EMPTY_ERROR_STRING,
    NVENCSTATUS, NV_ENC_ERR_GENERIC, NV_ENC_ERR_INVALID_PARAM, NV_ENC_ERR_INVALID_PTR,
    NV_ENC_ERR_UNIMPLEMENTED, NV_ENC_SUCCESS,
};

// ── [37] nvEncGetLastErrorString ────────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn interpose_get_last_error_string_impl(
    _encoder: *mut c_void,
) -> *const u8 {
    EMPTY_ERROR_STRING.as_ptr()
}

pub(crate) unsafe extern "C" fn interpose_get_last_error_string(
    _encoder: *mut c_void,
) -> *const u8 {
    rgpu_common::ffi::catch_panic(EMPTY_ERROR_STRING.as_ptr(), || interpose_get_last_error_string_impl(_encoder))
}

// ── [38] nvEncSetIOCudaStreams ──────────────────────────────────────────

/// Stub for nvEncSetIOCudaStreams. The server manages CUDA streams internally,
/// so we accept and ignore the client-side stream settings.
#[allow(non_snake_case)]
unsafe fn interpose_set_io_cuda_streams_impl(
    encoder: *mut c_void,
    _input_stream: *mut c_void,
    _output_stream: *mut c_void,
) -> NVENCSTATUS {
    debug!("NVENC SetIOCudaStreams: CALLED encoder={:?} (stub, returning SUCCESS)", encoder);
    NV_ENC_SUCCESS
}

pub(crate) unsafe extern "C" fn interpose_set_io_cuda_streams(
    encoder: *mut c_void,
    _input_stream: *mut c_void,
    _output_stream: *mut c_void,
) -> NVENCSTATUS {
    rgpu_common::ffi::catch_panic(NV_ENC_ERR_GENERIC, || interpose_set_io_cuda_streams_impl(encoder, _input_stream, _output_stream))
}

// ── [27] nvEncDestroyEncoder ───────────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn interpose_destroy_encoder_impl(
    encoder: *mut c_void,
) -> NVENCSTATUS {
    let enc_id = encoder as u64;
    let enc_handle = match handle_store::get_encoder(enc_id) {
        Some(h) => h,
        None => return NV_ENC_ERR_INVALID_PARAM,
    };

    let resp = send_nvenc_command(NvencCommand::DestroyEncoder {
        encoder: enc_handle,
    });
    if response_to_status(&resp) == NV_ENC_SUCCESS {
        handle_store::remove_encoder(enc_id);
    }
    response_to_status(&resp)
}

pub(crate) unsafe extern "C" fn interpose_destroy_encoder(
    encoder: *mut c_void,
) -> NVENCSTATUS {
    rgpu_common::ffi::catch_panic(NV_ENC_ERR_GENERIC, || interpose_destroy_encoder_impl(encoder))
}

// ── [28] nvEncInvalidateRefFrames ──────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn interpose_invalidate_ref_frames_impl(
    encoder: *mut c_void,
    invalid_ref_frame_timestamp: u64,
) -> NVENCSTATUS {
    let enc_handle = match encoder_handle_from_ptr(encoder) {
        Some(h) => h,
        None => return NV_ENC_ERR_INVALID_PARAM,
    };

    let resp = send_nvenc_command(NvencCommand::InvalidateRefFrames {
        encoder: enc_handle,
        invalid_ref_frame_timestamp,
    });
    response_to_status(&resp)
}

pub(crate) unsafe extern "C" fn interpose_invalidate_ref_frames(
    encoder: *mut c_void,
    invalid_ref_frame_timestamp: u64,
) -> NVENCSTATUS {
    rgpu_common::ffi::catch_panic(NV_ENC_ERR_GENERIC, || interpose_invalidate_ref_frames_impl(encoder, invalid_ref_frame_timestamp))
}

// ── [32] nvEncReconfigureEncoder ───────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn interpose_reconfigure_encoder_impl(
    encoder: *mut c_void,
    reconfig_params: *mut c_void,
) -> NVENCSTATUS {
    if reconfig_params.is_null() {
        return NV_ENC_ERR_INVALID_PTR;
    }
    let enc_handle = match encoder_handle_from_ptr(encoder) {
        Some(h) => h,
        None => return NV_ENC_ERR_INVALID_PARAM,
    };

    // NV_ENC_RECONFIGURE_PARAMS is a large struct - use safe fixed size
    const RECONFIG_PARAMS_SAFE_SIZE: usize = 8192;
    let params = std::slice::from_raw_parts(reconfig_params as *const u8, RECONFIG_PARAMS_SAFE_SIZE).to_vec();

    let resp = send_nvenc_command(NvencCommand::ReconfigureEncoder {
        encoder: enc_handle,
        params,
    });
    response_to_status(&resp)
}

pub(crate) unsafe extern "C" fn interpose_reconfigure_encoder(
    encoder: *mut c_void,
    reconfig_params: *mut c_void,
) -> NVENCSTATUS {
    rgpu_common::ffi::catch_panic(NV_ENC_ERR_GENERIC, || interpose_reconfigure_encoder_impl(encoder, reconfig_params))
}

// ── [22] nvEncGetSequenceParams ────────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn interpose_get_sequence_params_impl(
    encoder: *mut c_void,
    payload: *mut NvEncSequenceParamPayload,
) -> NVENCSTATUS {
    debug!("NVENC GetSequenceParams: CALLED encoder={:?}", encoder);
    if payload.is_null() {
        return NV_ENC_ERR_INVALID_PTR;
    }
    let enc_handle = match encoder_handle_from_ptr(encoder) {
        Some(h) => h,
        None => return NV_ENC_ERR_INVALID_PARAM,
    };

    let resp = send_nvenc_command(NvencCommand::GetSequenceParams {
        encoder: enc_handle,
    });
    match resp {
        NvencResponse::SequenceParams { data } => {
            let p = &mut *payload;
            let buf = p.sp_spps_buffer as *mut u8;
            if buf.is_null() {
                return NV_ENC_ERR_INVALID_PTR;
            }
            let copy_len = std::cmp::min(data.len(), p.in_buffer_size as usize);
            std::ptr::copy_nonoverlapping(data.as_ptr(), buf, copy_len);
            if !p.out_sps_pps_payload_size.is_null() {
                *p.out_sps_pps_payload_size = copy_len as u32;
            }
            debug!("NVENC GetSequenceParams: SUCCESS {} bytes", copy_len);
            NV_ENC_SUCCESS
        }
        NvencResponse::Error { code, .. } => {
            debug!("NVENC GetSequenceParams: ERROR code={}", code);
            code
        }
        _ => NV_ENC_ERR_GENERIC,
    }
}

pub(crate) unsafe extern "C" fn interpose_get_sequence_params(
    encoder: *mut c_void,
    payload: *mut NvEncSequenceParamPayload,
) -> NVENCSTATUS {
    rgpu_common::ffi::catch_panic(NV_ENC_ERR_GENERIC, || interpose_get_sequence_params_impl(encoder, payload))
}

// ── [23] nvEncRegisterAsyncEvent ───────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn interpose_register_async_event_impl(
    _encoder: *mut c_void,
    _event_params: *mut c_void,
) -> NVENCSTATUS {
    // Async events are not usable over the network
    NV_ENC_ERR_UNIMPLEMENTED
}

pub(crate) unsafe extern "C" fn interpose_register_async_event(
    _encoder: *mut c_void,
    _event_params: *mut c_void,
) -> NVENCSTATUS {
    rgpu_common::ffi::catch_panic(NV_ENC_ERR_GENERIC, || interpose_register_async_event_impl(_encoder, _event_params))
}

// ── [24] nvEncUnregisterAsyncEvent ─────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn interpose_unregister_async_event_impl(
    _encoder: *mut c_void,
    _event_params: *mut c_void,
) -> NVENCSTATUS {
    // Async events are not usable over the network
    NV_ENC_ERR_UNIMPLEMENTED
}

pub(crate) unsafe extern "C" fn interpose_unregister_async_event(
    _encoder: *mut c_void,
    _event_params: *mut c_void,
) -> NVENCSTATUS {
    rgpu_common::ffi::catch_panic(NV_ENC_ERR_GENERIC, || interpose_unregister_async_event_impl(_encoder, _event_params))
}

// ── [40] nvEncGetSequenceParamEx (stub) ────────────────────────────────

#[allow(non_snake_case)]
unsafe fn interpose_stub_get_sequence_param_ex_impl(
    _encoder: *mut c_void,
    _params: *mut c_void,
) -> NVENCSTATUS {
    debug!("NVENC GetSequenceParamEx: CALLED (stub, returning UNIMPLEMENTED)");
    NV_ENC_ERR_UNIMPLEMENTED
}

pub(crate) unsafe extern "C" fn interpose_stub_get_sequence_param_ex(
    _encoder: *mut c_void,
    _params: *mut c_void,
) -> NVENCSTATUS {
    rgpu_common::ffi::catch_panic(NV_ENC_ERR_GENERIC, || interpose_stub_get_sequence_param_ex_impl(_encoder, _params))
}

// ── [21] nvEncGetEncodeStats ───────────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn interpose_get_encode_stats_impl(
    encoder: *mut c_void,
    encode_stats: *mut c_void,
) -> NVENCSTATUS {
    if encode_stats.is_null() {
        return NV_ENC_ERR_INVALID_PTR;
    }
    let enc_handle = match encoder_handle_from_ptr(encoder) {
        Some(h) => h,
        None => return NV_ENC_ERR_INVALID_PARAM,
    };

    let resp = send_nvenc_command(NvencCommand::GetEncodeStats {
        encoder: enc_handle,
    });
    match resp {
        NvencResponse::EncodeStats { data } => {
            let dst = encode_stats as *mut u8;
            let copy_len = data.len();
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, copy_len);
            NV_ENC_SUCCESS
        }
        NvencResponse::Error { code, .. } => code,
        _ => NV_ENC_ERR_GENERIC,
    }
}

pub(crate) unsafe extern "C" fn interpose_get_encode_stats(
    encoder: *mut c_void,
    encode_stats: *mut c_void,
) -> NVENCSTATUS {
    rgpu_common::ffi::catch_panic(NV_ENC_ERR_GENERIC, || interpose_get_encode_stats_impl(encoder, encode_stats))
}

// ── [36] nvEncRunMotionEstimationOnly (stub) ───────────────────────────

#[allow(non_snake_case)]
unsafe fn interpose_stub_run_motion_estimation_only_impl(
    _encoder: *mut c_void,
    _params: *mut c_void,
) -> NVENCSTATUS {
    debug!("NVENC RunMotionEstimationOnly: CALLED (stub, returning UNIMPLEMENTED)");
    NV_ENC_ERR_UNIMPLEMENTED
}

pub(crate) unsafe extern "C" fn interpose_stub_run_motion_estimation_only(
    _encoder: *mut c_void,
    _params: *mut c_void,
) -> NVENCSTATUS {
    rgpu_common::ffi::catch_panic(NV_ENC_ERR_GENERIC, || interpose_stub_run_motion_estimation_only_impl(_encoder, _params))
}

// ── [34] nvEncCreateMVBuffer (stub) ────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn interpose_stub_create_mv_buffer_impl(
    _encoder: *mut c_void,
    _params: *mut c_void,
) -> NVENCSTATUS {
    debug!("NVENC CreateMVBuffer: CALLED (stub, returning UNIMPLEMENTED)");
    NV_ENC_ERR_UNIMPLEMENTED
}

pub(crate) unsafe extern "C" fn interpose_stub_create_mv_buffer(
    _encoder: *mut c_void,
    _params: *mut c_void,
) -> NVENCSTATUS {
    rgpu_common::ffi::catch_panic(NV_ENC_ERR_GENERIC, || interpose_stub_create_mv_buffer_impl(_encoder, _params))
}

// ── [35] nvEncDestroyMVBuffer (stub) ───────────────────────────────────

#[allow(non_snake_case)]
unsafe fn interpose_stub_destroy_mv_buffer_impl(
    _encoder: *mut c_void,
    _buffer: *mut c_void,
) -> NVENCSTATUS {
    debug!("NVENC DestroyMVBuffer: CALLED (stub, returning UNIMPLEMENTED)");
    NV_ENC_ERR_UNIMPLEMENTED
}

pub(crate) unsafe extern "C" fn interpose_stub_destroy_mv_buffer(
    _encoder: *mut c_void,
    _buffer: *mut c_void,
) -> NVENCSTATUS {
    rgpu_common::ffi::catch_panic(NV_ENC_ERR_GENERIC, || interpose_stub_destroy_mv_buffer_impl(_encoder, _buffer))
}
