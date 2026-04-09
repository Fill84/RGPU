//! Encoder session and encoding parameter interpose functions.
//!
//! Contains: open_encode_session, open_encode_session_ex, initialize_encoder,
//! get_encode_guid_count, get_encode_guids, get_encode_profile_guid_count,
//! get_encode_profile_guids, get_input_format_count, get_input_formats,
//! get_encode_caps, get_encode_preset_count, get_encode_preset_guids,
//! get_encode_preset_config, get_encode_preset_config_ex, encode_picture.

use std::ffi::{c_int, c_void};

use tracing::debug;

use rgpu_protocol::handle::NetworkHandle;
use rgpu_protocol::nvenc_commands::{NvencCommand, NvencResponse};

use crate::handle_store;
use crate::types::{
    NvEncCapsParam, NvEncInitializeParamsLayout, NvEncOpenEncodeSessionExParams, NvEncPicParams,
};
use crate::{
    encoder_handle_from_ptr, input_to_output_map, resolve_cuda_ctx_handle, response_to_status,
    send_nvenc_command, NvEncGuid, NVENCSTATUS, NV_ENC_ERR_GENERIC, NV_ENC_ERR_INVALID_DEVICE,
    NV_ENC_ERR_INVALID_PARAM, NV_ENC_ERR_INVALID_PTR, NV_ENC_ERR_UNIMPLEMENTED, NV_ENC_SUCCESS,
};

// ── [0] nvEncOpenEncodeSession (legacy) ────────────────────────────────

#[allow(non_snake_case)]
unsafe fn interpose_open_encode_session_impl(
    _device: *mut c_void,
    _device_type: u32,
    _encoder: *mut *mut c_void,
) -> NVENCSTATUS {
    // Legacy function, applications should use nvEncOpenEncodeSessionEx
    NV_ENC_ERR_UNIMPLEMENTED
}

pub(crate) unsafe extern "C" fn interpose_open_encode_session(
    _device: *mut c_void,
    _device_type: u32,
    _encoder: *mut *mut c_void,
) -> NVENCSTATUS {
    rgpu_common::ffi::catch_panic(NV_ENC_ERR_GENERIC, || interpose_open_encode_session_impl(_device, _device_type, _encoder))
}

// ── [29] nvEncOpenEncodeSessionEx ──────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn interpose_open_encode_session_ex_impl(
    open_session_ex_params: *mut NvEncOpenEncodeSessionExParams,
    encoder: *mut *mut c_void,
) -> NVENCSTATUS {
    if open_session_ex_params.is_null() || encoder.is_null() {
        return NV_ENC_ERR_INVALID_PTR;
    }

    let params = &*open_session_ex_params;
    let device_type = params.device_type;

    // The device pointer is a CUcontext from our CUDA interpose.
    // Resolve it to a proper NetworkHandle via the CUDA interpose's exported function.
    let cuda_ctx_id = params.device as u64;

    let cuda_context = match resolve_cuda_ctx_handle(cuda_ctx_id) {
        Some(h) => {
            debug!("NVENC: resolved CUDA ctx local_id={:#x} -> {:?}", cuda_ctx_id, h);
            h
        }
        None => {
            debug!("NVENC: could not resolve CUDA context local_id={:#x}", cuda_ctx_id);
            return NV_ENC_ERR_INVALID_DEVICE;
        }
    };

    debug!("NVENC OpenEncodeSessionEx: ctx_id={:#x} -> network_handle={:?} device_type={}", cuda_ctx_id, cuda_context, device_type);

    let resp = send_nvenc_command(NvencCommand::OpenEncodeSession {
        cuda_context,
        device_type,
    });
    match resp {
        NvencResponse::EncoderOpened { handle } => {
            let id = handle_store::store_encoder(handle);
            debug!("NVENC OpenEncodeSessionEx: SUCCESS encoder_ptr={:#x} handle={:?}", id, handle);
            *encoder = id as *mut c_void;
            NV_ENC_SUCCESS
        }
        NvencResponse::Error { code, message } => {
            debug!("NVENC OpenEncodeSession error: {} ({})", message, code);
            code
        }
        _ => NV_ENC_ERR_GENERIC,
    }
}

pub(crate) unsafe extern "C" fn interpose_open_encode_session_ex(
    open_session_ex_params: *mut NvEncOpenEncodeSessionExParams,
    encoder: *mut *mut c_void,
) -> NVENCSTATUS {
    rgpu_common::ffi::catch_panic(NV_ENC_ERR_GENERIC, || interpose_open_encode_session_ex_impl(open_session_ex_params, encoder))
}

// ── [11] nvEncInitializeEncoder ────────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn interpose_initialize_encoder_impl(
    encoder: *mut c_void,
    create_encode_params: *mut c_void,
) -> NVENCSTATUS {
    if create_encode_params.is_null() {
        return NV_ENC_ERR_INVALID_PTR;
    }
    let enc_handle = match encoder_handle_from_ptr(encoder) {
        Some(h) => h,
        None => return NV_ENC_ERR_INVALID_PARAM,
    };

    let params_size = std::mem::size_of::<NvEncInitializeParamsLayout>();
    let params_ptr = create_encode_params as *const u8;

    // Read the encodeConfig pointer at the known offset
    let config_ptr_offset = std::mem::offset_of!(NvEncInitializeParamsLayout, encode_config);
    let encode_config_ptr = *(params_ptr.add(config_ptr_offset) as *const *mut c_void);

    // Also read the privData pointer offset for zeroing
    let priv_data_offset = std::mem::offset_of!(NvEncInitializeParamsLayout, priv_data);

    debug!(
        "NVENC InitializeEncoder: encoder={:p}, params_size={}, config_ptr_offset={}, encodeConfig={:p}",
        encoder, params_size, config_ptr_offset, encode_config_ptr
    );

    // Read the full NV_ENC_INITIALIZE_PARAMS as raw bytes
    let mut params = std::slice::from_raw_parts(params_ptr, params_size).to_vec();

    // Zero out client-side pointers that are meaningless on the server
    let ptr_size = std::mem::size_of::<*mut c_void>();
    // encodeConfig pointer
    for b in &mut params[config_ptr_offset..config_ptr_offset + ptr_size] {
        *b = 0;
    }
    // privData pointer
    for b in &mut params[priv_data_offset..priv_data_offset + ptr_size] {
        *b = 0;
    }

    // If encodeConfig is non-null, read the NV_ENC_CONFIG data too.
    // NV_ENC_CONFIG is ~4608 bytes on 64-bit. Read 4096 bytes (safe, within allocation).
    // The server will pad it to a larger buffer for the real driver.
    const CONFIG_READ_SIZE: usize = 4096;
    let encode_config = if !encode_config_ptr.is_null() {
        let config_bytes = std::slice::from_raw_parts(
            encode_config_ptr as *const u8,
            CONFIG_READ_SIZE,
        ).to_vec();
        debug!("NVENC InitializeEncoder: encodeConfig version={:#010x}, reading {} bytes",
            u32::from_le_bytes([config_bytes[0], config_bytes[1], config_bytes[2], config_bytes[3]]),
            CONFIG_READ_SIZE);
        Some(config_bytes)
    } else {
        None
    };

    let resp = send_nvenc_command(NvencCommand::InitializeEncoder {
        encoder: enc_handle,
        params,
        encode_config,
    });
    let status = response_to_status(&resp);
    debug!("NVENC InitializeEncoder: response status={} — about to return to caller (BUILD=2026-02-27T2)", status);
    // Force flush stderr to ensure the log appears before potential crash
    #[cfg(target_os = "windows")]
    {
        use std::io::Write;
        let _ = std::io::stderr().flush();
    }
    status
}

pub(crate) unsafe extern "C" fn interpose_initialize_encoder(
    encoder: *mut c_void,
    create_encode_params: *mut c_void,
) -> NVENCSTATUS {
    rgpu_common::ffi::catch_panic(NV_ENC_ERR_GENERIC, || interpose_initialize_encoder_impl(encoder, create_encode_params))
}

// ── [1] nvEncGetEncodeGUIDCount ────────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn interpose_get_encode_guid_count_impl(
    encoder: *mut c_void,
    count: *mut u32,
) -> NVENCSTATUS {
    if count.is_null() {
        return NV_ENC_ERR_INVALID_PTR;
    }
    let enc_handle = match encoder_handle_from_ptr(encoder) {
        Some(h) => h,
        None => {
            debug!("NVENC GetEncodeGUIDCount: encoder handle not found for ptr {:?}", encoder);
            return NV_ENC_ERR_INVALID_PARAM;
        }
    };

    let resp = send_nvenc_command(NvencCommand::GetEncodeGUIDCount { encoder: enc_handle });
    match resp {
        NvencResponse::GUIDCount(c) => {
            *count = c;
            debug!("NVENC GetEncodeGUIDCount: encoder={:?} -> count={}", enc_handle, c);
            NV_ENC_SUCCESS
        }
        NvencResponse::Error { code, message } => {
            debug!("NVENC GetEncodeGUIDCount error: {} ({})", message, code);
            code
        }
        other => {
            debug!("NVENC GetEncodeGUIDCount: unexpected response: {:?}", other);
            NV_ENC_ERR_GENERIC
        }
    }
}

pub(crate) unsafe extern "C" fn interpose_get_encode_guid_count(
    encoder: *mut c_void,
    count: *mut u32,
) -> NVENCSTATUS {
    rgpu_common::ffi::catch_panic(NV_ENC_ERR_GENERIC, || interpose_get_encode_guid_count_impl(encoder, count))
}

// ── [2] nvEncGetEncodeGUIDs ────────────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn interpose_get_encode_guids_impl(
    encoder: *mut c_void,
    guids: *mut NvEncGuid,
    guid_array_size: u32,
    guid_count: *mut u32,
) -> NVENCSTATUS {
    debug!("NVENC GetEncodeGUIDs: CALLED encoder={:?} array_size={}", encoder, guid_array_size);
    if guids.is_null() || guid_count.is_null() {
        debug!("NVENC GetEncodeGUIDs: null pointer");
        return NV_ENC_ERR_INVALID_PTR;
    }
    let enc_handle = match encoder_handle_from_ptr(encoder) {
        Some(h) => h,
        None => {
            debug!("NVENC GetEncodeGUIDs: encoder handle not found for ptr {:?}", encoder);
            return NV_ENC_ERR_INVALID_PARAM;
        }
    };

    debug!("NVENC GetEncodeGUIDs: sending to daemon, encoder={:?}", enc_handle);
    let resp = send_nvenc_command(NvencCommand::GetEncodeGUIDs { encoder: enc_handle });
    match resp {
        NvencResponse::GUIDs(nv_guids) => {
            let write_count = std::cmp::min(nv_guids.len(), guid_array_size as usize);
            for i in 0..write_count {
                let g = NvEncGuid::from_nv_guid(&nv_guids[i]);
                debug!("NVENC GetEncodeGUIDs: guid[{}] = {:08x}-{:04x}-{:04x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
                    i, g.data1, g.data2, g.data3,
                    g.data4[0], g.data4[1], g.data4[2], g.data4[3],
                    g.data4[4], g.data4[5], g.data4[6], g.data4[7]);
                *guids.add(i) = g;
            }
            *guid_count = write_count as u32;
            debug!("NVENC GetEncodeGUIDs: returning {} guid(s)", write_count);
            NV_ENC_SUCCESS
        }
        NvencResponse::Error { code, message } => {
            debug!("NVENC GetEncodeGUIDs error: {} ({})", message, code);
            code
        }
        other => {
            debug!("NVENC GetEncodeGUIDs: unexpected response: {:?}", other);
            NV_ENC_ERR_GENERIC
        }
    }
}

pub(crate) unsafe extern "C" fn interpose_get_encode_guids(
    encoder: *mut c_void,
    guids: *mut NvEncGuid,
    guid_array_size: u32,
    guid_count: *mut u32,
) -> NVENCSTATUS {
    rgpu_common::ffi::catch_panic(NV_ENC_ERR_GENERIC, || interpose_get_encode_guids_impl(encoder, guids, guid_array_size, guid_count))
}

// ── [3] nvEncGetEncodeProfileGUIDCount ─────────────────────────────────

#[allow(non_snake_case)]
unsafe fn interpose_get_encode_profile_guid_count_impl(
    _encoder: *mut c_void,
    _encode_guid: NvEncGuid,
    count: *mut u32,
) -> NVENCSTATUS {
    if count.is_null() {
        return NV_ENC_ERR_INVALID_PTR;
    }
    // Profile GUIDs not yet implemented — return count=0
    *count = 0;
    NV_ENC_SUCCESS
}

pub(crate) unsafe extern "C" fn interpose_get_encode_profile_guid_count(
    _encoder: *mut c_void,
    _encode_guid: NvEncGuid,
    count: *mut u32,
) -> NVENCSTATUS {
    rgpu_common::ffi::catch_panic(NV_ENC_ERR_GENERIC, || interpose_get_encode_profile_guid_count_impl(_encoder, _encode_guid, count))
}

// ── [4] nvEncGetEncodeProfileGUIDs ─────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn interpose_get_encode_profile_guids_impl(
    _encoder: *mut c_void,
    _encode_guid: NvEncGuid,
    _profile_guids: *mut NvEncGuid,
    _guid_array_size: u32,
    guid_count: *mut u32,
) -> NVENCSTATUS {
    if guid_count.is_null() {
        return NV_ENC_ERR_INVALID_PTR;
    }
    // Profile GUIDs not yet implemented — return empty list
    *guid_count = 0;
    NV_ENC_SUCCESS
}

pub(crate) unsafe extern "C" fn interpose_get_encode_profile_guids(
    _encoder: *mut c_void,
    _encode_guid: NvEncGuid,
    _profile_guids: *mut NvEncGuid,
    _guid_array_size: u32,
    guid_count: *mut u32,
) -> NVENCSTATUS {
    rgpu_common::ffi::catch_panic(NV_ENC_ERR_GENERIC, || interpose_get_encode_profile_guids_impl(_encoder, _encode_guid, _profile_guids, _guid_array_size, guid_count))
}

// ── [5] nvEncGetInputFormatCount ───────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn interpose_get_input_format_count_impl(
    encoder: *mut c_void,
    encode_guid: NvEncGuid,
    count: *mut u32,
) -> NVENCSTATUS {
    if count.is_null() {
        return NV_ENC_ERR_INVALID_PTR;
    }
    let enc_handle = match encoder_handle_from_ptr(encoder) {
        Some(h) => h,
        None => return NV_ENC_ERR_INVALID_PARAM,
    };

    let resp = send_nvenc_command(NvencCommand::GetInputFormatCount {
        encoder: enc_handle,
        encode_guid: encode_guid.to_nv_guid(),
    });
    match resp {
        NvencResponse::InputFormatCount(c) => {
            *count = c;
            NV_ENC_SUCCESS
        }
        NvencResponse::Error { code, .. } => code,
        _ => NV_ENC_ERR_GENERIC,
    }
}

pub(crate) unsafe extern "C" fn interpose_get_input_format_count(
    encoder: *mut c_void,
    encode_guid: NvEncGuid,
    count: *mut u32,
) -> NVENCSTATUS {
    rgpu_common::ffi::catch_panic(NV_ENC_ERR_GENERIC, || interpose_get_input_format_count_impl(encoder, encode_guid, count))
}

// ── [6] nvEncGetInputFormats ───────────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn interpose_get_input_formats_impl(
    encoder: *mut c_void,
    encode_guid: NvEncGuid,
    input_fmts: *mut u32,
    input_fmt_array_size: u32,
    input_fmt_count: *mut u32,
) -> NVENCSTATUS {
    if input_fmts.is_null() || input_fmt_count.is_null() {
        return NV_ENC_ERR_INVALID_PTR;
    }
    let enc_handle = match encoder_handle_from_ptr(encoder) {
        Some(h) => h,
        None => return NV_ENC_ERR_INVALID_PARAM,
    };

    let resp = send_nvenc_command(NvencCommand::GetInputFormats {
        encoder: enc_handle,
        encode_guid: encode_guid.to_nv_guid(),
    });
    match resp {
        NvencResponse::InputFormats(fmts) => {
            let write_count = std::cmp::min(fmts.len(), input_fmt_array_size as usize);
            for i in 0..write_count {
                *input_fmts.add(i) = fmts[i];
            }
            *input_fmt_count = write_count as u32;
            NV_ENC_SUCCESS
        }
        NvencResponse::Error { code, .. } => code,
        _ => NV_ENC_ERR_GENERIC,
    }
}

pub(crate) unsafe extern "C" fn interpose_get_input_formats(
    encoder: *mut c_void,
    encode_guid: NvEncGuid,
    input_fmts: *mut u32,
    input_fmt_array_size: u32,
    input_fmt_count: *mut u32,
) -> NVENCSTATUS {
    rgpu_common::ffi::catch_panic(NV_ENC_ERR_GENERIC, || interpose_get_input_formats_impl(encoder, encode_guid, input_fmts, input_fmt_array_size, input_fmt_count))
}

// ── [7] nvEncGetEncodeCaps ─────────────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn interpose_get_encode_caps_impl(
    encoder: *mut c_void,
    encode_guid: NvEncGuid,
    caps_param: *mut NvEncCapsParam,
    caps_val: *mut c_int,
) -> NVENCSTATUS {
    if caps_param.is_null() || caps_val.is_null() {
        return NV_ENC_ERR_INVALID_PTR;
    }
    let enc_handle = match encoder_handle_from_ptr(encoder) {
        Some(h) => h,
        None => return NV_ENC_ERR_INVALID_PARAM,
    };

    let query = (*caps_param).caps_to_query;
    let resp = send_nvenc_command(NvencCommand::GetEncodeCaps {
        encoder: enc_handle,
        encode_guid: encode_guid.to_nv_guid(),
        caps_param: query as i32,
    });
    match resp {
        NvencResponse::CapsValue(val) => {
            debug!("NVENC GetEncodeCaps: query={} -> val={}", query, val);
            *caps_val = val;
            NV_ENC_SUCCESS
        }
        NvencResponse::Error { code, message } => {
            debug!("NVENC GetEncodeCaps: query={} -> ERROR code={} msg={}", query, code, message);
            code
        }
        _ => NV_ENC_ERR_GENERIC,
    }
}

pub(crate) unsafe extern "C" fn interpose_get_encode_caps(
    encoder: *mut c_void,
    encode_guid: NvEncGuid,
    caps_param: *mut NvEncCapsParam,
    caps_val: *mut c_int,
) -> NVENCSTATUS {
    rgpu_common::ffi::catch_panic(NV_ENC_ERR_GENERIC, || interpose_get_encode_caps_impl(encoder, encode_guid, caps_param, caps_val))
}

// ── [8] nvEncGetEncodePresetCount ──────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn interpose_get_encode_preset_count_impl(
    encoder: *mut c_void,
    encode_guid: NvEncGuid,
    count: *mut u32,
) -> NVENCSTATUS {
    if count.is_null() {
        return NV_ENC_ERR_INVALID_PTR;
    }
    let enc_handle = match encoder_handle_from_ptr(encoder) {
        Some(h) => h,
        None => return NV_ENC_ERR_INVALID_PARAM,
    };

    let resp = send_nvenc_command(NvencCommand::GetEncodePresetCount {
        encoder: enc_handle,
        encode_guid: encode_guid.to_nv_guid(),
    });
    match resp {
        NvencResponse::GUIDCount(c) => {
            *count = c;
            NV_ENC_SUCCESS
        }
        NvencResponse::Error { code, .. } => code,
        _ => NV_ENC_ERR_GENERIC,
    }
}

pub(crate) unsafe extern "C" fn interpose_get_encode_preset_count(
    encoder: *mut c_void,
    encode_guid: NvEncGuid,
    count: *mut u32,
) -> NVENCSTATUS {
    rgpu_common::ffi::catch_panic(NV_ENC_ERR_GENERIC, || interpose_get_encode_preset_count_impl(encoder, encode_guid, count))
}

// ── [9] nvEncGetEncodePresetGUIDs ──────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn interpose_get_encode_preset_guids_impl(
    encoder: *mut c_void,
    encode_guid: NvEncGuid,
    preset_guids: *mut NvEncGuid,
    guid_array_size: u32,
    guid_count: *mut u32,
) -> NVENCSTATUS {
    if preset_guids.is_null() || guid_count.is_null() {
        return NV_ENC_ERR_INVALID_PTR;
    }
    let enc_handle = match encoder_handle_from_ptr(encoder) {
        Some(h) => h,
        None => return NV_ENC_ERR_INVALID_PARAM,
    };

    let resp = send_nvenc_command(NvencCommand::GetEncodePresetGUIDs {
        encoder: enc_handle,
        encode_guid: encode_guid.to_nv_guid(),
    });
    match resp {
        NvencResponse::GUIDs(nv_guids) => {
            let write_count = std::cmp::min(nv_guids.len(), guid_array_size as usize);
            for i in 0..write_count {
                *preset_guids.add(i) = NvEncGuid::from_nv_guid(&nv_guids[i]);
            }
            *guid_count = write_count as u32;
            NV_ENC_SUCCESS
        }
        NvencResponse::Error { code, .. } => code,
        _ => NV_ENC_ERR_GENERIC,
    }
}

pub(crate) unsafe extern "C" fn interpose_get_encode_preset_guids(
    encoder: *mut c_void,
    encode_guid: NvEncGuid,
    preset_guids: *mut NvEncGuid,
    guid_array_size: u32,
    guid_count: *mut u32,
) -> NVENCSTATUS {
    rgpu_common::ffi::catch_panic(NV_ENC_ERR_GENERIC, || interpose_get_encode_preset_guids_impl(encoder, encode_guid, preset_guids, guid_array_size, guid_count))
}

// ── [10] nvEncGetEncodePresetConfig (legacy) ───────────────────────────

#[allow(non_snake_case)]
unsafe fn interpose_get_encode_preset_config_impl(
    encoder: *mut c_void,
    encode_guid: NvEncGuid,
    preset_guid: NvEncGuid,
    preset_config: *mut c_void,
) -> NVENCSTATUS {
    debug!("NVENC GetEncodePresetConfig (legacy): CALLED encoder={:p}", encoder);
    if preset_config.is_null() {
        return NV_ENC_ERR_INVALID_PTR;
    }
    let enc_handle = match encoder_handle_from_ptr(encoder) {
        Some(h) => h,
        None => return NV_ENC_ERR_INVALID_PARAM,
    };

    let resp = send_nvenc_command(NvencCommand::GetEncodePresetConfig {
        encoder: enc_handle,
        encode_guid: encode_guid.to_nv_guid(),
        preset_guid: preset_guid.to_nv_guid(),
    });
    match resp {
        NvencResponse::PresetConfig(data) => {
            // Copy raw preset config data into the caller's buffer.
            // The caller provides a versioned struct (NV_ENC_PRESET_CONFIG) which
            // starts with a version field; the server serializes the full struct.
            let dst = preset_config as *mut u8;
            let copy_len = data.len();
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, copy_len);
            NV_ENC_SUCCESS
        }
        NvencResponse::Error { code, .. } => code,
        _ => NV_ENC_ERR_GENERIC,
    }
}

pub(crate) unsafe extern "C" fn interpose_get_encode_preset_config(
    encoder: *mut c_void,
    encode_guid: NvEncGuid,
    preset_guid: NvEncGuid,
    preset_config: *mut c_void,
) -> NVENCSTATUS {
    rgpu_common::ffi::catch_panic(NV_ENC_ERR_GENERIC, || interpose_get_encode_preset_config_impl(encoder, encode_guid, preset_guid, preset_config))
}

// ── [39] nvEncGetEncodePresetConfigEx ──────────────────────────────────

#[allow(non_snake_case)]
unsafe fn interpose_get_encode_preset_config_ex_impl(
    encoder: *mut c_void,
    encode_guid: NvEncGuid,
    preset_guid: NvEncGuid,
    tuning_info: u32,
    preset_config: *mut c_void,
) -> NVENCSTATUS {
    debug!("NVENC GetEncodePresetConfigEx: CALLED encoder={:p} tuning={}", encoder, tuning_info);
    if preset_config.is_null() {
        debug!("NVENC GetEncodePresetConfigEx: NULL preset_config ptr");
        return NV_ENC_ERR_INVALID_PTR;
    }
    let enc_handle = match encoder_handle_from_ptr(encoder) {
        Some(h) => h,
        None => {
            debug!("NVENC GetEncodePresetConfigEx: failed to resolve encoder handle");
            return NV_ENC_ERR_INVALID_PARAM;
        }
    };
    debug!("NVENC GetEncodePresetConfigEx: sending IPC command");

    let resp = send_nvenc_command(NvencCommand::GetEncodePresetConfigEx {
        encoder: enc_handle,
        encode_guid: encode_guid.to_nv_guid(),
        preset_guid: preset_guid.to_nv_guid(),
        tuning_info,
    });
    debug!("NVENC GetEncodePresetConfigEx: got response");
    match resp {
        NvencResponse::PresetConfig(data) => {
            debug!("NVENC GetEncodePresetConfigEx: SUCCESS {} bytes", data.len());
            let dst = preset_config as *mut u8;
            let copy_len = data.len();
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, copy_len);
            NV_ENC_SUCCESS
        }
        NvencResponse::Error { code, message } => {
            debug!("NVENC GetEncodePresetConfigEx: error code={} msg={}", code, message);
            code
        }
        _ => {
            debug!("NVENC GetEncodePresetConfigEx: unexpected response");
            NV_ENC_ERR_GENERIC
        }
    }
}

pub(crate) unsafe extern "C" fn interpose_get_encode_preset_config_ex(
    encoder: *mut c_void,
    encode_guid: NvEncGuid,
    preset_guid: NvEncGuid,
    tuning_info: u32,
    preset_config: *mut c_void,
) -> NVENCSTATUS {
    rgpu_common::ffi::catch_panic(NV_ENC_ERR_GENERIC, || interpose_get_encode_preset_config_ex_impl(encoder, encode_guid, preset_guid, tuning_info, preset_config))
}

// ── [16] nvEncEncodePicture ────────────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn interpose_encode_picture_impl(
    encoder: *mut c_void,
    encode_pic_params: *mut c_void,
) -> NVENCSTATUS {
    debug!("NVENC EncodePicture: CALLED encoder={:?}", encoder);
    if encode_pic_params.is_null() {
        return NV_ENC_ERR_INVALID_PTR;
    }
    let enc_handle = match encoder_handle_from_ptr(encoder) {
        Some(h) => h,
        None => return NV_ENC_ERR_INVALID_PARAM,
    };

    let pic = encode_pic_params as *const NvEncPicParams;

    // NV_ENC_PIC_PARAMS is a large struct. Use a safe fixed size for serialization.
    // The version field does NOT contain the struct size (unlike some NVIDIA APIs).
    const PIC_PARAMS_SAFE_SIZE: usize = 4096;

    // Resolve input/output handles
    let input_id = (*pic).input_buffer as u64;
    let mut output_id = (*pic).output_bitstream as u64;

    // If outputBitstream is NULL, look up the paired bitstream buffer
    if output_id == 0 && input_id != 0 {
        if let Some(&paired_output) = input_to_output_map().lock().get(&input_id) {
            debug!("NVENC EncodePicture: outputBitstream was NULL, using paired buffer {:#x} for input {:#x}",
                paired_output, input_id);
            output_id = paired_output;
        }
    }

    debug!("NVENC EncodePicture: input={:#x} output={:#x} type={} {}x{} pitch={}",
        input_id, output_id, (*pic).picture_type,
        (*pic).input_width, (*pic).input_height, (*pic).input_pitch);

    // Input can be either an input buffer or a mapped resource
    let input_handle = handle_store::get_input_buffer(input_id)
        .or_else(|| handle_store::get_mapped_resource(input_id))
        .unwrap_or(NetworkHandle::null());

    let output_handle = handle_store::get_bitstream_buffer(output_id)
        .unwrap_or(NetworkHandle::null());

    // Serialize the raw params
    let params_bytes = std::slice::from_raw_parts(
        encode_pic_params as *const u8,
        PIC_PARAMS_SAFE_SIZE,
    ).to_vec();

    let resp = send_nvenc_command(NvencCommand::EncodePicture {
        encoder: enc_handle,
        params: params_bytes,
        input: input_handle,
        output: output_handle,
        picture_type: (*pic).picture_type,
    });
    let status = response_to_status(&resp);
    debug!("NVENC EncodePicture: response status={}", status);
    status
}

pub(crate) unsafe extern "C" fn interpose_encode_picture(
    encoder: *mut c_void,
    encode_pic_params: *mut c_void,
) -> NVENCSTATUS {
    rgpu_common::ffi::catch_panic(NV_ENC_ERR_GENERIC, || interpose_encode_picture_impl(encoder, encode_pic_params))
}
