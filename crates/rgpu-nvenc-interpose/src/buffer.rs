//! Buffer management interpose functions.
//!
//! Contains: create/destroy input buffers, create/destroy bitstream buffers,
//! lock/unlock bitstream, lock/unlock input buffer, register/unregister resource,
//! map/unmap input resource.

use std::ffi::c_void;

use tracing::debug;

use rgpu_protocol::nvenc_commands::{NvencCommand, NvencResponse};

use crate::handle_store;
use crate::types::{
    calc_buffer_size, input_buffer_dims, locked_bitstreams, locked_input_buffers,
    InputBufferDims, LockedBitstreamInfo, LockedInputBufferInfo,
    NvEncCreateBitstreamBufferParams, NvEncCreateInputBufferParams,
    NvEncLockBitstreamParams, NvEncLockInputBufferParams,
    NvEncMapInputResourceParams, NvEncRegisterResourceParams,
};
use crate::{
    encoder_handle_from_ptr, input_to_output_map, resolve_cuda_mem_handle, response_to_status,
    send_nvenc_command, LAST_INPUT_BUFFER_ID, NVENCSTATUS, NV_ENC_ERR_GENERIC,
    NV_ENC_ERR_INVALID_PARAM, NV_ENC_ERR_INVALID_PTR, NV_ENC_INPUT_PTR, NV_ENC_OUTPUT_PTR,
    NV_ENC_REGISTERED_PTR, NV_ENC_SUCCESS,
};

// ── [12] nvEncCreateInputBuffer ────────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn interpose_create_input_buffer_impl(
    encoder: *mut c_void,
    create_input_buffer_params: *mut NvEncCreateInputBufferParams,
) -> NVENCSTATUS {
    debug!("NVENC CreateInputBuffer: CALLED encoder={:?}", encoder);
    if create_input_buffer_params.is_null() {
        return NV_ENC_ERR_INVALID_PTR;
    }
    let enc_handle = match encoder_handle_from_ptr(encoder) {
        Some(h) => h,
        None => return NV_ENC_ERR_INVALID_PARAM,
    };

    let params = &mut *create_input_buffer_params;
    debug!("NVENC CreateInputBuffer: {}x{} fmt={}", params.width, params.height, params.buffer_fmt);

    let resp = send_nvenc_command(NvencCommand::CreateInputBuffer {
        encoder: enc_handle,
        width: params.width,
        height: params.height,
        buffer_fmt: params.buffer_fmt,
    });
    match resp {
        NvencResponse::InputBufferCreated { handle } => {
            let id = handle_store::store_input_buffer(handle);
            params.input_buffer = id as NV_ENC_INPUT_PTR;
            // Track dimensions for LockInputBuffer shadow buffer allocation
            {
                let mut dims = input_buffer_dims().lock();
                dims.insert(id, InputBufferDims {
                    height: params.height,
                    buffer_fmt: params.buffer_fmt,
                });
            }
            // Record last input buffer ID for pairing with next bitstream buffer
            LAST_INPUT_BUFFER_ID.store(id, std::sync::atomic::Ordering::Relaxed);
            debug!("NVENC CreateInputBuffer: SUCCESS id={:#x}", id);
            NV_ENC_SUCCESS
        }
        NvencResponse::Error { code, .. } => {
            debug!("NVENC CreateInputBuffer: ERROR code={}", code);
            code
        }
        _ => NV_ENC_ERR_GENERIC,
    }
}

pub(crate) unsafe extern "C" fn interpose_create_input_buffer(
    encoder: *mut c_void,
    create_input_buffer_params: *mut NvEncCreateInputBufferParams,
) -> NVENCSTATUS {
    rgpu_common::ffi::catch_panic(NV_ENC_ERR_GENERIC, || interpose_create_input_buffer_impl(encoder, create_input_buffer_params))
}

// ── [13] nvEncDestroyInputBuffer ───────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn interpose_destroy_input_buffer_impl(
    encoder: *mut c_void,
    input_buffer: NV_ENC_INPUT_PTR,
) -> NVENCSTATUS {
    let enc_handle = match encoder_handle_from_ptr(encoder) {
        Some(h) => h,
        None => return NV_ENC_ERR_INVALID_PARAM,
    };
    let buf_id = input_buffer as u64;
    let buf_handle = match handle_store::get_input_buffer(buf_id) {
        Some(h) => h,
        None => return NV_ENC_ERR_INVALID_PARAM,
    };

    let resp = send_nvenc_command(NvencCommand::DestroyInputBuffer {
        encoder: enc_handle,
        input_buffer: buf_handle,
    });
    if response_to_status(&resp) == NV_ENC_SUCCESS {
        handle_store::remove_input_buffer(buf_id);
        input_buffer_dims().lock().remove(&buf_id);
    }
    response_to_status(&resp)
}

pub(crate) unsafe extern "C" fn interpose_destroy_input_buffer(
    encoder: *mut c_void,
    input_buffer: NV_ENC_INPUT_PTR,
) -> NVENCSTATUS {
    rgpu_common::ffi::catch_panic(NV_ENC_ERR_GENERIC, || interpose_destroy_input_buffer_impl(encoder, input_buffer))
}

// ── [14] nvEncCreateBitstreamBuffer ────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn interpose_create_bitstream_buffer_impl(
    encoder: *mut c_void,
    create_bitstream_buffer_params: *mut NvEncCreateBitstreamBufferParams,
) -> NVENCSTATUS {
    debug!("NVENC CreateBitstreamBuffer: CALLED encoder={:?}", encoder);
    if create_bitstream_buffer_params.is_null() {
        return NV_ENC_ERR_INVALID_PTR;
    }
    let enc_handle = match encoder_handle_from_ptr(encoder) {
        Some(h) => h,
        None => return NV_ENC_ERR_INVALID_PARAM,
    };

    let params = &mut *create_bitstream_buffer_params;

    let resp = send_nvenc_command(NvencCommand::CreateBitstreamBuffer {
        encoder: enc_handle,
    });
    match resp {
        NvencResponse::BitstreamBufferCreated { handle } => {
            let id = handle_store::store_bitstream_buffer(handle);
            params.bitstream_buffer = id as NV_ENC_OUTPUT_PTR;
            // Pair this bitstream buffer with the last input buffer
            let last_input = LAST_INPUT_BUFFER_ID.load(std::sync::atomic::Ordering::Relaxed);
            if last_input != 0 {
                input_to_output_map().lock().insert(last_input, id);
                LAST_INPUT_BUFFER_ID.store(0, std::sync::atomic::Ordering::Relaxed);
            }
            debug!("NVENC CreateBitstreamBuffer: SUCCESS id={:#x} paired_with_input={:#x}", id, last_input);
            NV_ENC_SUCCESS
        }
        NvencResponse::Error { code, .. } => {
            debug!("NVENC CreateBitstreamBuffer: ERROR code={}", code);
            code
        }
        _ => NV_ENC_ERR_GENERIC,
    }
}

pub(crate) unsafe extern "C" fn interpose_create_bitstream_buffer(
    encoder: *mut c_void,
    create_bitstream_buffer_params: *mut NvEncCreateBitstreamBufferParams,
) -> NVENCSTATUS {
    rgpu_common::ffi::catch_panic(NV_ENC_ERR_GENERIC, || interpose_create_bitstream_buffer_impl(encoder, create_bitstream_buffer_params))
}

// ── [15] nvEncDestroyBitstreamBuffer ───────────────────────────────────

#[allow(non_snake_case)]
unsafe fn interpose_destroy_bitstream_buffer_impl(
    encoder: *mut c_void,
    bitstream_buffer: NV_ENC_OUTPUT_PTR,
) -> NVENCSTATUS {
    let enc_handle = match encoder_handle_from_ptr(encoder) {
        Some(h) => h,
        None => return NV_ENC_ERR_INVALID_PARAM,
    };
    let buf_id = bitstream_buffer as u64;
    let buf_handle = match handle_store::get_bitstream_buffer(buf_id) {
        Some(h) => h,
        None => return NV_ENC_ERR_INVALID_PARAM,
    };

    let resp = send_nvenc_command(NvencCommand::DestroyBitstreamBuffer {
        encoder: enc_handle,
        bitstream_buffer: buf_handle,
    });
    if response_to_status(&resp) == NV_ENC_SUCCESS {
        handle_store::remove_bitstream_buffer(buf_id);
    }
    response_to_status(&resp)
}

pub(crate) unsafe extern "C" fn interpose_destroy_bitstream_buffer(
    encoder: *mut c_void,
    bitstream_buffer: NV_ENC_OUTPUT_PTR,
) -> NVENCSTATUS {
    rgpu_common::ffi::catch_panic(NV_ENC_ERR_GENERIC, || interpose_destroy_bitstream_buffer_impl(encoder, bitstream_buffer))
}

// ── [17] nvEncLockBitstream ────────────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn interpose_lock_bitstream_impl(
    encoder: *mut c_void,
    lock_bitstream_params: *mut NvEncLockBitstreamParams,
) -> NVENCSTATUS {
    if lock_bitstream_params.is_null() {
        return NV_ENC_ERR_INVALID_PTR;
    }
    let enc_handle = match encoder_handle_from_ptr(encoder) {
        Some(h) => h,
        None => return NV_ENC_ERR_INVALID_PARAM,
    };

    let params = &mut *lock_bitstream_params;
    let buf_id = params.output_bitstream as u64;
    debug!("NVENC LockBitstream: buf_id={:#x}", buf_id);
    let buf_handle = match handle_store::get_bitstream_buffer(buf_id) {
        Some(h) => h,
        None => {
            debug!("NVENC LockBitstream: FAILED to find handle for buf_id={:#x}", buf_id);
            return NV_ENC_ERR_INVALID_PARAM;
        }
    };

    let resp = send_nvenc_command(NvencCommand::LockBitstream {
        encoder: enc_handle,
        bitstream_buffer: buf_handle,
    });
    match resp {
        NvencResponse::BitstreamData {
            data,
            picture_type,
            frame_idx,
            output_timestamp,
        } => {
            // Store the data locally so the app can read from the pointer
            let data_len = data.len();
            let mut locked = locked_bitstreams().lock();
            locked.insert(buf_id, LockedBitstreamInfo {
                data,
                picture_type,
                frame_idx,
                output_timestamp,
            });
            let info = match locked.get(&buf_id) {
                Some(i) => i,
                None => return NV_ENC_ERR_GENERIC,
            };
            params.bitstream_buffer_ptr = info.data.as_ptr() as *mut c_void;
            params.bitstream_size_in_bytes = data_len as u32;
            params.picture_type = picture_type;
            params.frame_idx = frame_idx;
            params.output_timestamp = output_timestamp;
            NV_ENC_SUCCESS
        }
        NvencResponse::Error { code, .. } => code,
        _ => NV_ENC_ERR_GENERIC,
    }
}

pub(crate) unsafe extern "C" fn interpose_lock_bitstream(
    encoder: *mut c_void,
    lock_bitstream_params: *mut NvEncLockBitstreamParams,
) -> NVENCSTATUS {
    rgpu_common::ffi::catch_panic(NV_ENC_ERR_GENERIC, || interpose_lock_bitstream_impl(encoder, lock_bitstream_params))
}

// ── [18] nvEncUnlockBitstream ──────────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn interpose_unlock_bitstream_impl(
    encoder: *mut c_void,
    bitstream_buffer: NV_ENC_OUTPUT_PTR,
) -> NVENCSTATUS {
    let enc_handle = match encoder_handle_from_ptr(encoder) {
        Some(h) => h,
        None => return NV_ENC_ERR_INVALID_PARAM,
    };
    let buf_id = bitstream_buffer as u64;
    let buf_handle = match handle_store::get_bitstream_buffer(buf_id) {
        Some(h) => h,
        None => return NV_ENC_ERR_INVALID_PARAM,
    };

    // Release local shadow buffer
    {
        let mut locked = locked_bitstreams().lock();
        locked.remove(&buf_id);
    }

    let resp = send_nvenc_command(NvencCommand::UnlockBitstream {
        encoder: enc_handle,
        bitstream_buffer: buf_handle,
    });
    response_to_status(&resp)
}

pub(crate) unsafe extern "C" fn interpose_unlock_bitstream(
    encoder: *mut c_void,
    bitstream_buffer: NV_ENC_OUTPUT_PTR,
) -> NVENCSTATUS {
    rgpu_common::ffi::catch_panic(NV_ENC_ERR_GENERIC, || interpose_unlock_bitstream_impl(encoder, bitstream_buffer))
}

// ── [19] nvEncLockInputBuffer ──────────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn interpose_lock_input_buffer_impl(
    encoder: *mut c_void,
    lock_input_buffer_params: *mut NvEncLockInputBufferParams,
) -> NVENCSTATUS {
    if lock_input_buffer_params.is_null() {
        return NV_ENC_ERR_INVALID_PTR;
    }
    let enc_handle = match encoder_handle_from_ptr(encoder) {
        Some(h) => h,
        None => return NV_ENC_ERR_INVALID_PARAM,
    };

    let params = &mut *lock_input_buffer_params;
    let buf_id = params.input_buffer as u64;
    let buf_handle = match handle_store::get_input_buffer(buf_id) {
        Some(h) => h,
        None => return NV_ENC_ERR_INVALID_PARAM,
    };

    let resp = send_nvenc_command(NvencCommand::LockInputBuffer {
        encoder: enc_handle,
        input_buffer: buf_handle,
    });
    match resp {
        NvencResponse::InputBufferLocked { pitch, buffer_size: _ } => {
            // Calculate shadow buffer size from pitch + tracked dimensions
            let size = {
                let dims = input_buffer_dims().lock();
                if let Some(d) = dims.get(&buf_id) {
                    calc_buffer_size(pitch, d.height, d.buffer_fmt)
                } else {
                    // Fallback: generous estimate (4K * pitch)
                    debug!("NVENC LockInputBuffer: no dims for buf {:#x}, using fallback size", buf_id);
                    (pitch as usize) * 4096
                }
            };
            debug!("NVENC LockInputBuffer: buf={:#x} pitch={} shadow_size={}", buf_id, pitch, size);
            // Allocate a shadow buffer for the app to write pixel data into
            let data = vec![0u8; size];
            let mut locked = locked_input_buffers().lock();
            locked.insert(buf_id, LockedInputBufferInfo {
                data,
                pitch,
            });
            // The pointer must come from the stored data (not the local `data` which moved)
            let info = match locked.get(&buf_id) {
                Some(i) => i,
                None => return NV_ENC_ERR_GENERIC,
            };
            params.buffer_data_ptr = info.data.as_ptr() as *mut c_void;
            params.pitch = pitch;
            NV_ENC_SUCCESS
        }
        NvencResponse::Error { code, .. } => code,
        _ => NV_ENC_ERR_GENERIC,
    }
}

pub(crate) unsafe extern "C" fn interpose_lock_input_buffer(
    encoder: *mut c_void,
    lock_input_buffer_params: *mut NvEncLockInputBufferParams,
) -> NVENCSTATUS {
    rgpu_common::ffi::catch_panic(NV_ENC_ERR_GENERIC, || interpose_lock_input_buffer_impl(encoder, lock_input_buffer_params))
}

// ── [20] nvEncUnlockInputBuffer ────────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn interpose_unlock_input_buffer_impl(
    encoder: *mut c_void,
    input_buffer: NV_ENC_INPUT_PTR,
) -> NVENCSTATUS {
    let enc_handle = match encoder_handle_from_ptr(encoder) {
        Some(h) => h,
        None => return NV_ENC_ERR_INVALID_PARAM,
    };
    let buf_id = input_buffer as u64;
    let buf_handle = match handle_store::get_input_buffer(buf_id) {
        Some(h) => h,
        None => return NV_ENC_ERR_INVALID_PARAM,
    };

    // Get the shadow buffer data and send it to the server
    let (data, pitch) = {
        let mut locked = locked_input_buffers().lock();
        match locked.remove(&buf_id) {
            Some(info) => (info.data, info.pitch),
            None => return NV_ENC_ERR_INVALID_PARAM,
        }
    };

    let resp = send_nvenc_command(NvencCommand::UnlockInputBuffer {
        encoder: enc_handle,
        input_buffer: buf_handle,
        data,
        pitch,
    });
    response_to_status(&resp)
}

pub(crate) unsafe extern "C" fn interpose_unlock_input_buffer(
    encoder: *mut c_void,
    input_buffer: NV_ENC_INPUT_PTR,
) -> NVENCSTATUS {
    rgpu_common::ffi::catch_panic(NV_ENC_ERR_GENERIC, || interpose_unlock_input_buffer_impl(encoder, input_buffer))
}

// ── [30] nvEncRegisterResource ─────────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn interpose_register_resource_impl(
    encoder: *mut c_void,
    register_resource_params: *mut NvEncRegisterResourceParams,
) -> NVENCSTATUS {
    debug!("NVENC RegisterResource: CALLED encoder={:?}", encoder);
    if register_resource_params.is_null() {
        return NV_ENC_ERR_INVALID_PTR;
    }
    let enc_handle = match encoder_handle_from_ptr(encoder) {
        Some(h) => h,
        None => return NV_ENC_ERR_INVALID_PARAM,
    };

    let params = &mut *register_resource_params;
    debug!("NVENC RegisterResource: type={} resource={:?} {}x{} pitch={} fmt={}",
        params.resource_type, params.resource_to_register,
        params.width, params.height, params.pitch, params.buffer_format);

    // The resource_to_register is a CUDA device pointer (CUdeviceptr) from our interpose.
    // Resolve it to a proper NetworkHandle via the CUDA interpose's export.
    let resource_id = params.resource_to_register as u64;
    let resource = match resolve_cuda_mem_handle(resource_id) {
        Some(h) => {
            debug!("NVENC RegisterResource: resolved mem local_id={:#x} -> {:?}", resource_id, h);
            h
        }
        None => {
            debug!("NVENC RegisterResource: could not resolve resource local_id={:#x}", resource_id);
            return NV_ENC_ERR_INVALID_PARAM;
        }
    };

    let resp = send_nvenc_command(NvencCommand::RegisterResource {
        encoder: enc_handle,
        resource_type: params.resource_type,
        resource,
        width: params.width,
        height: params.height,
        pitch: params.pitch,
        buffer_fmt: params.buffer_format,
    });
    match resp {
        NvencResponse::ResourceRegistered { handle } => {
            let id = handle_store::store_registered_resource(handle);
            params.registered_resource = id as NV_ENC_REGISTERED_PTR;
            debug!("NVENC RegisterResource: SUCCESS id={:#x}", id);
            NV_ENC_SUCCESS
        }
        NvencResponse::Error { code, .. } => {
            debug!("NVENC RegisterResource: ERROR code={}", code);
            code
        }
        _ => NV_ENC_ERR_GENERIC,
    }
}

pub(crate) unsafe extern "C" fn interpose_register_resource(
    encoder: *mut c_void,
    register_resource_params: *mut NvEncRegisterResourceParams,
) -> NVENCSTATUS {
    rgpu_common::ffi::catch_panic(NV_ENC_ERR_GENERIC, || interpose_register_resource_impl(encoder, register_resource_params))
}

// ── [31] nvEncUnregisterResource ───────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn interpose_unregister_resource_impl(
    encoder: *mut c_void,
    registered_resource: NV_ENC_REGISTERED_PTR,
) -> NVENCSTATUS {
    let enc_handle = match encoder_handle_from_ptr(encoder) {
        Some(h) => h,
        None => return NV_ENC_ERR_INVALID_PARAM,
    };
    let reg_id = registered_resource as u64;
    let reg_handle = match handle_store::get_registered_resource(reg_id) {
        Some(h) => h,
        None => return NV_ENC_ERR_INVALID_PARAM,
    };

    let resp = send_nvenc_command(NvencCommand::UnregisterResource {
        encoder: enc_handle,
        registered_resource: reg_handle,
    });
    if response_to_status(&resp) == NV_ENC_SUCCESS {
        handle_store::remove_registered_resource(reg_id);
    }
    response_to_status(&resp)
}

pub(crate) unsafe extern "C" fn interpose_unregister_resource(
    encoder: *mut c_void,
    registered_resource: NV_ENC_REGISTERED_PTR,
) -> NVENCSTATUS {
    rgpu_common::ffi::catch_panic(NV_ENC_ERR_GENERIC, || interpose_unregister_resource_impl(encoder, registered_resource))
}

// ── [25] nvEncMapInputResource ─────────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn interpose_map_input_resource_impl(
    encoder: *mut c_void,
    map_input_resource_params: *mut NvEncMapInputResourceParams,
) -> NVENCSTATUS {
    debug!("NVENC MapInputResource: CALLED encoder={:?}", encoder);
    if map_input_resource_params.is_null() {
        return NV_ENC_ERR_INVALID_PTR;
    }
    let enc_handle = match encoder_handle_from_ptr(encoder) {
        Some(h) => h,
        None => return NV_ENC_ERR_INVALID_PARAM,
    };

    let params = &mut *map_input_resource_params;
    let reg_id = params.input_resource as u64;
    debug!("NVENC MapInputResource: input_resource={:#x}", reg_id);
    let reg_handle = match handle_store::get_registered_resource(reg_id) {
        Some(h) => h,
        None => {
            debug!("NVENC MapInputResource: could not find registered resource {:#x}", reg_id);
            return NV_ENC_ERR_INVALID_PARAM;
        }
    };

    let resp = send_nvenc_command(NvencCommand::MapInputResource {
        encoder: enc_handle,
        registered_resource: reg_handle,
    });
    match resp {
        NvencResponse::ResourceMapped { handle } => {
            let id = handle_store::store_mapped_resource(handle);
            params.mapped_resource = id as NV_ENC_INPUT_PTR;
            debug!("NVENC MapInputResource: SUCCESS id={:#x}", id);
            NV_ENC_SUCCESS
        }
        NvencResponse::Error { code, .. } => {
            debug!("NVENC MapInputResource: ERROR code={}", code);
            code
        }
        _ => NV_ENC_ERR_GENERIC,
    }
}

pub(crate) unsafe extern "C" fn interpose_map_input_resource(
    encoder: *mut c_void,
    map_input_resource_params: *mut NvEncMapInputResourceParams,
) -> NVENCSTATUS {
    rgpu_common::ffi::catch_panic(NV_ENC_ERR_GENERIC, || interpose_map_input_resource_impl(encoder, map_input_resource_params))
}

// ── [26] nvEncUnmapInputResource ───────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn interpose_unmap_input_resource_impl(
    encoder: *mut c_void,
    mapped_resource: NV_ENC_INPUT_PTR,
) -> NVENCSTATUS {
    let enc_handle = match encoder_handle_from_ptr(encoder) {
        Some(h) => h,
        None => return NV_ENC_ERR_INVALID_PARAM,
    };
    let map_id = mapped_resource as u64;
    let map_handle = match handle_store::get_mapped_resource(map_id) {
        Some(h) => h,
        None => return NV_ENC_ERR_INVALID_PARAM,
    };

    let resp = send_nvenc_command(NvencCommand::UnmapInputResource {
        encoder: enc_handle,
        mapped_resource: map_handle,
    });
    if response_to_status(&resp) == NV_ENC_SUCCESS {
        handle_store::remove_mapped_resource(map_id);
    }
    response_to_status(&resp)
}

pub(crate) unsafe extern "C" fn interpose_unmap_input_resource(
    encoder: *mut c_void,
    mapped_resource: NV_ENC_INPUT_PTR,
) -> NVENCSTATUS {
    rgpu_common::ffi::catch_panic(NV_ENC_ERR_GENERIC, || interpose_unmap_input_resource_impl(encoder, mapped_resource))
}
