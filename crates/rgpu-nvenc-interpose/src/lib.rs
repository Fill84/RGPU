//! NVENC Video Encoder API interception library.
//!
//! This cdylib replaces the standard NVENC library (nvEncodeAPI64.dll on Windows,
//! libnvidia-encode.so on Linux). It intercepts NVENC encoding calls and forwards
//! them to the RGPU client daemon via IPC.
//!
//! NVENC only exports 2 functions:
//! 1. `NvEncodeAPIGetMaxSupportedVersion(uint32_t* version)` - returns API version
//! 2. `NvEncodeAPICreateInstance(NV_ENCODE_API_FUNCTION_LIST* functionList)` - fills a vtable
//!
//! All actual encoding operations happen through function pointers in the vtable.
//! Our interpose fills the vtable with our own functions that send IPC commands.

#![allow(non_camel_case_types)]

mod ipc_client;
pub mod handle_store;

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

use tracing::error;

use rgpu_protocol::handle::NetworkHandle;
use rgpu_protocol::nvenc_commands::{NvGuid, NvencCommand, NvencResponse};

use ipc_client::NvencIpcClient;

// ── NVENC types ────────────────────────────────────────────────────────

type NVENCSTATUS = i32;
type NV_ENC_INPUT_PTR = *mut c_void;
type NV_ENC_OUTPUT_PTR = *mut c_void;
type NV_ENC_REGISTERED_PTR = *mut c_void;

const NV_ENC_SUCCESS: NVENCSTATUS = 0;
const NV_ENC_ERR_INVALID_PTR: NVENCSTATUS = 6;
const NV_ENC_ERR_INVALID_PARAM: NVENCSTATUS = 8;
const NV_ENC_ERR_GENERIC: NVENCSTATUS = 20;
const NV_ENC_ERR_UNIMPLEMENTED: NVENCSTATUS = 22;

/// NVENC API version: (major << 4) | minor. We advertise 12.2.
const NVENC_API_VERSION: u32 = (12 << 4) | 2;

// ── GUID type matching NVENC's GUID layout ─────────────────────────────

/// NVENC GUID structure (matches Windows GUID layout).
#[repr(C)]
#[derive(Clone, Copy)]
struct NvEncGuid {
    data1: u32,
    data2: u16,
    data3: u16,
    data4: [u8; 8],
}

impl NvEncGuid {
    fn to_nv_guid(&self) -> NvGuid {
        let mut bytes = [0u8; 16];
        bytes[0..4].copy_from_slice(&self.data1.to_le_bytes());
        bytes[4..6].copy_from_slice(&self.data2.to_le_bytes());
        bytes[6..8].copy_from_slice(&self.data3.to_le_bytes());
        bytes[8..16].copy_from_slice(&self.data4);
        NvGuid(bytes)
    }

    fn from_nv_guid(guid: &NvGuid) -> Self {
        NvEncGuid {
            data1: u32::from_le_bytes([guid.0[0], guid.0[1], guid.0[2], guid.0[3]]),
            data2: u16::from_le_bytes([guid.0[4], guid.0[5]]),
            data3: u16::from_le_bytes([guid.0[6], guid.0[7]]),
            data4: [
                guid.0[8], guid.0[9], guid.0[10], guid.0[11],
                guid.0[12], guid.0[13], guid.0[14], guid.0[15],
            ],
        }
    }
}

// ── NVENC FFI structs ──────────────────────────────────────────────────

/// NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS
#[repr(C)]
struct NvEncOpenEncodeSessionExParams {
    version: u32,
    device_type: u32,
    device: *mut c_void,
    reserved: *mut c_void,
    api_version: u32,
    reserved1: [u32; 253],
    reserved2: [*mut c_void; 64],
}

/// NV_ENC_CAPS_PARAM
#[repr(C)]
struct NvEncCapsParam {
    version: u32,
    caps_to_query: u32,
    reserved: [u32; 62],
}

/// NV_ENC_CREATE_INPUT_BUFFER
#[repr(C)]
struct NvEncCreateInputBufferParams {
    version: u32,
    width: u32,
    height: u32,
    memory_heap: u32,
    buffer_fmt: u32,
    reserved: u32,
    input_buffer: NV_ENC_INPUT_PTR,
    p_sys_mem: *mut c_void,
    reserved1: [u32; 57],
    reserved2: [*mut c_void; 63],
}

/// NV_ENC_CREATE_BITSTREAM_BUFFER
#[repr(C)]
struct NvEncCreateBitstreamBufferParams {
    version: u32,
    reserved: u32,
    bitstream_buffer: NV_ENC_OUTPUT_PTR,
    reserved_mem_buf: *mut c_void,
    reserved_size: u32,
    reserved1: [u32; 57],
    reserved2: [*mut c_void; 63],
}

/// NV_ENC_LOCK_INPUT_BUFFER
#[repr(C)]
struct NvEncLockInputBufferParams {
    version: u32,
    reserved_flags: u32,
    input_buffer: NV_ENC_INPUT_PTR,
    buffer_data_ptr: *mut c_void,
    pitch: u32,
    reserved1: [u32; 251],
    reserved2: [*mut c_void; 64],
}

/// NV_ENC_LOCK_BITSTREAM
#[repr(C)]
struct NvEncLockBitstreamParams {
    version: u32,
    do_not_wait: u32,
    output_bitstream: NV_ENC_OUTPUT_PTR,
    slice_offsets: *mut u32,
    frame_idx: u32,
    hw_encode_status: u32,
    num_slices: u32,
    bitstream_size_in_bytes: u32,
    output_timestamp: u64,
    output_duration: u64,
    bitstream_buffer_ptr: *mut c_void,
    picture_type: u32,
    picture_struct: u32,
    frame_avg_qp: u32,
    frame_satd: u32,
    ltr_frame_idx: u32,
    ltr_frame_bitmap: u32,
    reserved: [u32; 13],
    intra_refresh_cnt: u32,
    reserved1: [u32; 219],
    reserved2: [*mut c_void; 64],
}

/// NV_ENC_REGISTER_RESOURCE
#[repr(C)]
struct NvEncRegisterResourceParams {
    version: u32,
    resource_type: u32,
    width: u32,
    height: u32,
    pitch: u32,
    sub_resource_index: u32,
    resource_to_register: *mut c_void,
    registered_resource: NV_ENC_REGISTERED_PTR,
    buffer_format: u32,
    buffer_usage: u32,
    reserved1: [u32; 247],
    reserved2: [*mut c_void; 62],
}

/// NV_ENC_MAP_INPUT_RESOURCE
#[repr(C)]
struct NvEncMapInputResourceParams {
    version: u32,
    sub_resource_index: u32,
    input_resource: NV_ENC_REGISTERED_PTR,
    mapped_resource: NV_ENC_INPUT_PTR,
    mapped_buffer_fmt: u32,
    reserved1: [u32; 255],
    reserved2: [*mut c_void; 63],
}

/// NV_ENC_PIC_PARAMS (simplified — we send the raw bytes over the wire)
#[repr(C)]
struct NvEncPicParams {
    version: u32,
    input_width: u32,
    input_height: u32,
    input_pitch: u32,
    encode_params_flags: u32,
    frame_idx: u32,
    input_timestamp: u64,
    input_duration: u64,
    input_buffer: NV_ENC_INPUT_PTR,
    output_bitstream: NV_ENC_OUTPUT_PTR,
    completion_event: *mut c_void,
    buffer_fmt: u32,
    picture_struct: u32,
    picture_type: u32,
    // Remaining fields not needed for our interpose — raw bytes sent
}

/// NV_ENC_SEQUENCE_PARAM_PAYLOAD
#[repr(C)]
struct NvEncSequenceParamPayload {
    version: u32,
    in_buffer_size: u32,
    sps_id: u32,
    pps_id: u32,
    sp_spps_buffer: *mut c_void,
    out_sps_pps_payload_size: *mut u32,
    reserved1: [u32; 250],
    reserved2: [*mut c_void; 64],
}

// ── NV_ENCODE_API_FUNCTION_LIST (the vtable) ───────────────────────────

/// The NVENC function table that applications receive from NvEncodeAPICreateInstance.
/// This matches the NV_ENCODE_API_FUNCTION_LIST layout from nvEncodeAPI.h.
#[repr(C)]
pub struct NvEncApiFunctionList {
    version: u32,
    reserved: u32,
    // [0]  nvEncOpenEncodeSession (legacy)
    nv_enc_open_encode_session: Option<unsafe extern "C" fn(*mut c_void, u32, *mut *mut c_void) -> NVENCSTATUS>,
    // [1]  nvEncGetEncodeGUIDCount
    nv_enc_get_encode_guid_count: Option<unsafe extern "C" fn(*mut c_void, *mut u32) -> NVENCSTATUS>,
    // [2]  nvEncGetEncodeGUIDs
    nv_enc_get_encode_guids: Option<unsafe extern "C" fn(*mut c_void, *mut NvEncGuid, u32, *mut u32) -> NVENCSTATUS>,
    // [3]  nvEncGetEncodeProfileGUIDCount
    nv_enc_get_encode_profile_guid_count: Option<unsafe extern "C" fn(*mut c_void, NvEncGuid, *mut u32) -> NVENCSTATUS>,
    // [4]  nvEncGetEncodeProfileGUIDs
    nv_enc_get_encode_profile_guids: Option<unsafe extern "C" fn(*mut c_void, NvEncGuid, *mut NvEncGuid, u32, *mut u32) -> NVENCSTATUS>,
    // [5]  nvEncGetInputFormatCount
    nv_enc_get_input_format_count: Option<unsafe extern "C" fn(*mut c_void, NvEncGuid, *mut u32) -> NVENCSTATUS>,
    // [6]  nvEncGetInputFormats
    nv_enc_get_input_formats: Option<unsafe extern "C" fn(*mut c_void, NvEncGuid, *mut u32, u32, *mut u32) -> NVENCSTATUS>,
    // [7]  nvEncGetEncodeCaps
    nv_enc_get_encode_caps: Option<unsafe extern "C" fn(*mut c_void, NvEncGuid, *mut NvEncCapsParam, *mut c_int) -> NVENCSTATUS>,
    // [8]  nvEncGetEncodePresetCount
    nv_enc_get_encode_preset_count: Option<unsafe extern "C" fn(*mut c_void, NvEncGuid, *mut u32) -> NVENCSTATUS>,
    // [9]  nvEncGetEncodePresetGUIDs
    nv_enc_get_encode_preset_guids: Option<unsafe extern "C" fn(*mut c_void, NvEncGuid, *mut NvEncGuid, u32, *mut u32) -> NVENCSTATUS>,
    // [10] nvEncGetEncodePresetConfig (legacy)
    nv_enc_get_encode_preset_config: Option<unsafe extern "C" fn(*mut c_void, NvEncGuid, NvEncGuid, *mut c_void) -> NVENCSTATUS>,
    // [11] nvEncInitializeEncoder
    nv_enc_initialize_encoder: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NVENCSTATUS>,
    // [12] nvEncCreateInputBuffer
    nv_enc_create_input_buffer: Option<unsafe extern "C" fn(*mut c_void, *mut NvEncCreateInputBufferParams) -> NVENCSTATUS>,
    // [13] nvEncDestroyInputBuffer
    nv_enc_destroy_input_buffer: Option<unsafe extern "C" fn(*mut c_void, NV_ENC_INPUT_PTR) -> NVENCSTATUS>,
    // [14] nvEncCreateBitstreamBuffer
    nv_enc_create_bitstream_buffer: Option<unsafe extern "C" fn(*mut c_void, *mut NvEncCreateBitstreamBufferParams) -> NVENCSTATUS>,
    // [15] nvEncDestroyBitstreamBuffer
    nv_enc_destroy_bitstream_buffer: Option<unsafe extern "C" fn(*mut c_void, NV_ENC_OUTPUT_PTR) -> NVENCSTATUS>,
    // [16] nvEncEncodePicture
    nv_enc_encode_picture: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NVENCSTATUS>,
    // [17] nvEncLockBitstream
    nv_enc_lock_bitstream: Option<unsafe extern "C" fn(*mut c_void, *mut NvEncLockBitstreamParams) -> NVENCSTATUS>,
    // [18] nvEncUnlockBitstream
    nv_enc_unlock_bitstream: Option<unsafe extern "C" fn(*mut c_void, NV_ENC_OUTPUT_PTR) -> NVENCSTATUS>,
    // [19] nvEncLockInputBuffer
    nv_enc_lock_input_buffer: Option<unsafe extern "C" fn(*mut c_void, *mut NvEncLockInputBufferParams) -> NVENCSTATUS>,
    // [20] nvEncUnlockInputBuffer
    nv_enc_unlock_input_buffer: Option<unsafe extern "C" fn(*mut c_void, NV_ENC_INPUT_PTR) -> NVENCSTATUS>,
    // [21] nvEncGetEncodeStats
    nv_enc_get_encode_stats: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NVENCSTATUS>,
    // [22] nvEncGetSequenceParams
    nv_enc_get_sequence_params: Option<unsafe extern "C" fn(*mut c_void, *mut NvEncSequenceParamPayload) -> NVENCSTATUS>,
    // [23] nvEncRegisterAsyncEvent
    nv_enc_register_async_event: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NVENCSTATUS>,
    // [24] nvEncUnregisterAsyncEvent
    nv_enc_unregister_async_event: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NVENCSTATUS>,
    // [25] nvEncMapInputResource
    nv_enc_map_input_resource: Option<unsafe extern "C" fn(*mut c_void, *mut NvEncMapInputResourceParams) -> NVENCSTATUS>,
    // [26] nvEncUnmapInputResource
    nv_enc_unmap_input_resource: Option<unsafe extern "C" fn(*mut c_void, NV_ENC_INPUT_PTR) -> NVENCSTATUS>,
    // [27] nvEncDestroyEncoder
    nv_enc_destroy_encoder: Option<unsafe extern "C" fn(*mut c_void) -> NVENCSTATUS>,
    // [28] nvEncInvalidateRefFrames
    nv_enc_invalidate_ref_frames: Option<unsafe extern "C" fn(*mut c_void, u64) -> NVENCSTATUS>,
    // [29] nvEncOpenEncodeSessionEx
    nv_enc_open_encode_session_ex: Option<unsafe extern "C" fn(*mut NvEncOpenEncodeSessionExParams, *mut *mut c_void) -> NVENCSTATUS>,
    // [30] nvEncRegisterResource
    nv_enc_register_resource: Option<unsafe extern "C" fn(*mut c_void, *mut NvEncRegisterResourceParams) -> NVENCSTATUS>,
    // [31] nvEncUnregisterResource
    nv_enc_unregister_resource: Option<unsafe extern "C" fn(*mut c_void, NV_ENC_REGISTERED_PTR) -> NVENCSTATUS>,
    // [32] nvEncReconfigureEncoder
    nv_enc_reconfigure_encoder: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NVENCSTATUS>,
    // [33] reserved1
    reserved1: *mut c_void,
    // [34] nvEncGetEncodePresetConfigEx
    nv_enc_get_encode_preset_config_ex: Option<unsafe extern "C" fn(*mut c_void, NvEncGuid, NvEncGuid, u32, *mut c_void) -> NVENCSTATUS>,
}

// ── IPC client singleton ───────────────────────────────────────────────

static IPC_CLIENT: OnceLock<NvencIpcClient> = OnceLock::new();

fn get_client() -> &'static NvencIpcClient {
    IPC_CLIENT.get_or_init(|| {
        let path = rgpu_common::platform::default_ipc_path();
        NvencIpcClient::new(&path)
    })
}

fn send_nvenc_command(cmd: NvencCommand) -> NvencResponse {
    let client = get_client();
    match client.send_command(cmd) {
        Ok(resp) => resp,
        Err(e) => {
            error!("NVENC IPC error: {}", e);
            NvencResponse::Error {
                code: NV_ENC_ERR_GENERIC,
                message: e.to_string(),
            }
        }
    }
}

/// Extract the NVENCSTATUS error code from an NvencResponse::Error, or 0 for success.
fn response_to_status(resp: &NvencResponse) -> NVENCSTATUS {
    match resp {
        NvencResponse::Success => NV_ENC_SUCCESS,
        NvencResponse::Error { code, .. } => *code,
        _ => NV_ENC_SUCCESS,
    }
}

/// Extract the encoder NetworkHandle from a local opaque encoder pointer.
fn encoder_handle_from_ptr(encoder: *mut c_void) -> Option<NetworkHandle> {
    let id = encoder as u64;
    handle_store::get_encoder(id)
}

// ── Shadow buffers for LockInputBuffer ─────────────────────────────────
// When we lock an input buffer, the app writes pixel data into it. We need a
// local buffer to store that data, then send it via IPC on unlock.

use parking_lot::Mutex;
use std::collections::HashMap;

struct LockedInputBufferInfo {
    data: Vec<u8>,
    pitch: u32,
}

static LOCKED_INPUT_BUFFERS: OnceLock<Mutex<HashMap<u64, LockedInputBufferInfo>>> = OnceLock::new();

fn locked_input_buffers() -> &'static Mutex<HashMap<u64, LockedInputBufferInfo>> {
    LOCKED_INPUT_BUFFERS.get_or_init(|| Mutex::new(HashMap::new()))
}

// ── Shadow buffers for LockBitstream ───────────────────────────────────
// When we lock a bitstream buffer, the server sends us encoded data. We keep
// it in a local buffer until the app calls unlock.

struct LockedBitstreamInfo {
    data: Vec<u8>,
    #[allow(dead_code)]
    picture_type: u32,
    #[allow(dead_code)]
    frame_idx: u32,
    #[allow(dead_code)]
    output_timestamp: u64,
}

static LOCKED_BITSTREAMS: OnceLock<Mutex<HashMap<u64, LockedBitstreamInfo>>> = OnceLock::new();

fn locked_bitstreams() -> &'static Mutex<HashMap<u64, LockedBitstreamInfo>> {
    LOCKED_BITSTREAMS.get_or_init(|| Mutex::new(HashMap::new()))
}

// ── Marker function for RGPU interpose DLL detection ───────────────────

/// Detection marker function for installer/daemon verification.
/// Returns 1 to indicate this is the RGPU interpose library.
#[no_mangle]
pub extern "C" fn rgpu_interpose_marker() -> c_int {
    1
}

// ── Exported NVENC API Functions ───────────────────────────────────────

/// NvEncodeAPIGetMaxSupportedVersion - returns the maximum supported NVENC API version.
#[no_mangle]
pub unsafe extern "C" fn NvEncodeAPIGetMaxSupportedVersion(version: *mut u32) -> NVENCSTATUS {
    // Initialize logging on first call
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("RGPU_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .try_init();

    if version.is_null() {
        return NV_ENC_ERR_INVALID_PTR;
    }

    let resp = send_nvenc_command(NvencCommand::GetMaxSupportedVersion);
    match resp {
        NvencResponse::MaxSupportedVersion { version: v } => {
            *version = v;
            NV_ENC_SUCCESS
        }
        NvencResponse::Error { code, .. } => code,
        _ => {
            // Fallback: return our target API version
            *version = NVENC_API_VERSION;
            NV_ENC_SUCCESS
        }
    }
}

/// NvEncodeAPICreateInstance - fills the function table with our interceptor functions.
#[no_mangle]
pub unsafe extern "C" fn NvEncodeAPICreateInstance(
    function_list: *mut NvEncApiFunctionList,
) -> NVENCSTATUS {
    // Initialize logging on first call
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("RGPU_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .try_init();

    if function_list.is_null() {
        return NV_ENC_ERR_INVALID_PTR;
    }

    let fl = &mut *function_list;

    // Fill the function table with our interceptors
    fl.nv_enc_open_encode_session = Some(interpose_open_encode_session);
    fl.nv_enc_get_encode_guid_count = Some(interpose_get_encode_guid_count);
    fl.nv_enc_get_encode_guids = Some(interpose_get_encode_guids);
    fl.nv_enc_get_encode_profile_guid_count = Some(interpose_get_encode_profile_guid_count);
    fl.nv_enc_get_encode_profile_guids = Some(interpose_get_encode_profile_guids);
    fl.nv_enc_get_input_format_count = Some(interpose_get_input_format_count);
    fl.nv_enc_get_input_formats = Some(interpose_get_input_formats);
    fl.nv_enc_get_encode_caps = Some(interpose_get_encode_caps);
    fl.nv_enc_get_encode_preset_count = Some(interpose_get_encode_preset_count);
    fl.nv_enc_get_encode_preset_guids = Some(interpose_get_encode_preset_guids);
    fl.nv_enc_get_encode_preset_config = Some(interpose_get_encode_preset_config);
    fl.nv_enc_initialize_encoder = Some(interpose_initialize_encoder);
    fl.nv_enc_create_input_buffer = Some(interpose_create_input_buffer);
    fl.nv_enc_destroy_input_buffer = Some(interpose_destroy_input_buffer);
    fl.nv_enc_create_bitstream_buffer = Some(interpose_create_bitstream_buffer);
    fl.nv_enc_destroy_bitstream_buffer = Some(interpose_destroy_bitstream_buffer);
    fl.nv_enc_encode_picture = Some(interpose_encode_picture);
    fl.nv_enc_lock_bitstream = Some(interpose_lock_bitstream);
    fl.nv_enc_unlock_bitstream = Some(interpose_unlock_bitstream);
    fl.nv_enc_lock_input_buffer = Some(interpose_lock_input_buffer);
    fl.nv_enc_unlock_input_buffer = Some(interpose_unlock_input_buffer);
    fl.nv_enc_get_encode_stats = Some(interpose_get_encode_stats);
    fl.nv_enc_get_sequence_params = Some(interpose_get_sequence_params);
    fl.nv_enc_register_async_event = Some(interpose_register_async_event);
    fl.nv_enc_unregister_async_event = Some(interpose_unregister_async_event);
    fl.nv_enc_map_input_resource = Some(interpose_map_input_resource);
    fl.nv_enc_unmap_input_resource = Some(interpose_unmap_input_resource);
    fl.nv_enc_destroy_encoder = Some(interpose_destroy_encoder);
    fl.nv_enc_invalidate_ref_frames = Some(interpose_invalidate_ref_frames);
    fl.nv_enc_open_encode_session_ex = Some(interpose_open_encode_session_ex);
    fl.nv_enc_register_resource = Some(interpose_register_resource);
    fl.nv_enc_unregister_resource = Some(interpose_unregister_resource);
    fl.nv_enc_reconfigure_encoder = Some(interpose_reconfigure_encoder);
    fl.reserved1 = std::ptr::null_mut();
    fl.nv_enc_get_encode_preset_config_ex = Some(interpose_get_encode_preset_config_ex);

    NV_ENC_SUCCESS
}

// ═══════════════════════════════════════════════════════════════════════
// Vtable interceptor functions
// ═══════════════════════════════════════════════════════════════════════

// ── [0] nvEncOpenEncodeSession (legacy) ────────────────────────────────

unsafe extern "C" fn interpose_open_encode_session(
    _device: *mut c_void,
    _device_type: u32,
    _encoder: *mut *mut c_void,
) -> NVENCSTATUS {
    // Legacy function, applications should use nvEncOpenEncodeSessionEx
    NV_ENC_ERR_UNIMPLEMENTED
}

// ── [1] nvEncGetEncodeGUIDCount ────────────────────────────────────────

unsafe extern "C" fn interpose_get_encode_guid_count(
    encoder: *mut c_void,
    count: *mut u32,
) -> NVENCSTATUS {
    if count.is_null() {
        return NV_ENC_ERR_INVALID_PTR;
    }
    let enc_handle = match encoder_handle_from_ptr(encoder) {
        Some(h) => h,
        None => return NV_ENC_ERR_INVALID_PARAM,
    };

    let resp = send_nvenc_command(NvencCommand::GetEncodeGUIDCount { encoder: enc_handle });
    match resp {
        NvencResponse::GUIDCount(c) => {
            *count = c;
            NV_ENC_SUCCESS
        }
        NvencResponse::Error { code, .. } => code,
        _ => NV_ENC_ERR_GENERIC,
    }
}

// ── [2] nvEncGetEncodeGUIDs ────────────────────────────────────────────

unsafe extern "C" fn interpose_get_encode_guids(
    encoder: *mut c_void,
    guids: *mut NvEncGuid,
    guid_array_size: u32,
    guid_count: *mut u32,
) -> NVENCSTATUS {
    if guids.is_null() || guid_count.is_null() {
        return NV_ENC_ERR_INVALID_PTR;
    }
    let enc_handle = match encoder_handle_from_ptr(encoder) {
        Some(h) => h,
        None => return NV_ENC_ERR_INVALID_PARAM,
    };

    let resp = send_nvenc_command(NvencCommand::GetEncodeGUIDs { encoder: enc_handle });
    match resp {
        NvencResponse::GUIDs(nv_guids) => {
            let write_count = std::cmp::min(nv_guids.len(), guid_array_size as usize);
            for i in 0..write_count {
                *guids.add(i) = NvEncGuid::from_nv_guid(&nv_guids[i]);
            }
            *guid_count = write_count as u32;
            NV_ENC_SUCCESS
        }
        NvencResponse::Error { code, .. } => code,
        _ => NV_ENC_ERR_GENERIC,
    }
}

// ── [3] nvEncGetEncodeProfileGUIDCount ─────────────────────────────────

unsafe extern "C" fn interpose_get_encode_profile_guid_count(
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

// ── [4] nvEncGetEncodeProfileGUIDs ─────────────────────────────────────

unsafe extern "C" fn interpose_get_encode_profile_guids(
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

// ── [5] nvEncGetInputFormatCount ───────────────────────────────────────

unsafe extern "C" fn interpose_get_input_format_count(
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

// ── [6] nvEncGetInputFormats ───────────────────────────────────────────

unsafe extern "C" fn interpose_get_input_formats(
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

// ── [7] nvEncGetEncodeCaps ─────────────────────────────────────────────

unsafe extern "C" fn interpose_get_encode_caps(
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

    let resp = send_nvenc_command(NvencCommand::GetEncodeCaps {
        encoder: enc_handle,
        encode_guid: encode_guid.to_nv_guid(),
        caps_param: (*caps_param).caps_to_query as i32,
    });
    match resp {
        NvencResponse::CapsValue(val) => {
            *caps_val = val;
            NV_ENC_SUCCESS
        }
        NvencResponse::Error { code, .. } => code,
        _ => NV_ENC_ERR_GENERIC,
    }
}

// ── [8] nvEncGetEncodePresetCount ──────────────────────────────────────

unsafe extern "C" fn interpose_get_encode_preset_count(
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

// ── [9] nvEncGetEncodePresetGUIDs ──────────────────────────────────────

unsafe extern "C" fn interpose_get_encode_preset_guids(
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

// ── [10] nvEncGetEncodePresetConfig (legacy) ───────────────────────────

unsafe extern "C" fn interpose_get_encode_preset_config(
    encoder: *mut c_void,
    encode_guid: NvEncGuid,
    preset_guid: NvEncGuid,
    preset_config: *mut c_void,
) -> NVENCSTATUS {
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

// ── [11] nvEncInitializeEncoder ────────────────────────────────────────

unsafe extern "C" fn interpose_initialize_encoder(
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

    // Read the version field to determine the struct size, then send raw bytes.
    // NV_ENC_INITIALIZE_PARAMS starts with a version field: (struct_size | (api_ver << 16)).
    let version = *(create_encode_params as *const u32);
    let struct_size = (version & 0xFFFF) as usize;
    let struct_size = if struct_size == 0 { 4096 } else { struct_size }; // fallback

    let params = std::slice::from_raw_parts(create_encode_params as *const u8, struct_size).to_vec();

    let resp = send_nvenc_command(NvencCommand::InitializeEncoder {
        encoder: enc_handle,
        params,
    });
    response_to_status(&resp)
}

// ── [12] nvEncCreateInputBuffer ────────────────────────────────────────

unsafe extern "C" fn interpose_create_input_buffer(
    encoder: *mut c_void,
    create_input_buffer_params: *mut NvEncCreateInputBufferParams,
) -> NVENCSTATUS {
    if create_input_buffer_params.is_null() {
        return NV_ENC_ERR_INVALID_PTR;
    }
    let enc_handle = match encoder_handle_from_ptr(encoder) {
        Some(h) => h,
        None => return NV_ENC_ERR_INVALID_PARAM,
    };

    let params = &mut *create_input_buffer_params;

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
            NV_ENC_SUCCESS
        }
        NvencResponse::Error { code, .. } => code,
        _ => NV_ENC_ERR_GENERIC,
    }
}

// ── [13] nvEncDestroyInputBuffer ───────────────────────────────────────

unsafe extern "C" fn interpose_destroy_input_buffer(
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
    }
    response_to_status(&resp)
}

// ── [14] nvEncCreateBitstreamBuffer ────────────────────────────────────

unsafe extern "C" fn interpose_create_bitstream_buffer(
    encoder: *mut c_void,
    create_bitstream_buffer_params: *mut NvEncCreateBitstreamBufferParams,
) -> NVENCSTATUS {
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
            NV_ENC_SUCCESS
        }
        NvencResponse::Error { code, .. } => code,
        _ => NV_ENC_ERR_GENERIC,
    }
}

// ── [15] nvEncDestroyBitstreamBuffer ───────────────────────────────────

unsafe extern "C" fn interpose_destroy_bitstream_buffer(
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

// ── [16] nvEncEncodePicture ────────────────────────────────────────────

unsafe extern "C" fn interpose_encode_picture(
    encoder: *mut c_void,
    encode_pic_params: *mut c_void,
) -> NVENCSTATUS {
    if encode_pic_params.is_null() {
        return NV_ENC_ERR_INVALID_PTR;
    }
    let enc_handle = match encoder_handle_from_ptr(encoder) {
        Some(h) => h,
        None => return NV_ENC_ERR_INVALID_PARAM,
    };

    // Read the NV_ENC_PIC_PARAMS fields we need.
    // The struct starts with a version field; use it to determine size.
    let pic = encode_pic_params as *const NvEncPicParams;
    let version = (*pic).version;
    let struct_size = (version & 0xFFFF) as usize;
    let struct_size = if struct_size == 0 { 4096 } else { struct_size };

    // Resolve input/output handles
    let input_id = (*pic).input_buffer as u64;
    let output_id = (*pic).output_bitstream as u64;

    // Input can be either an input buffer or a mapped resource
    let input_handle = handle_store::get_input_buffer(input_id)
        .or_else(|| handle_store::get_mapped_resource(input_id))
        .unwrap_or(NetworkHandle::null());

    let output_handle = handle_store::get_bitstream_buffer(output_id)
        .unwrap_or(NetworkHandle::null());

    // Serialize the raw params (excluding pointers that we resolve separately)
    let params_bytes = std::slice::from_raw_parts(
        encode_pic_params as *const u8,
        struct_size,
    ).to_vec();

    let resp = send_nvenc_command(NvencCommand::EncodePicture {
        encoder: enc_handle,
        params: params_bytes,
        input: input_handle,
        output: output_handle,
        picture_type: (*pic).picture_type,
    });
    response_to_status(&resp)
}

// ── [17] nvEncLockBitstream ────────────────────────────────────────────

unsafe extern "C" fn interpose_lock_bitstream(
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
    let buf_handle = match handle_store::get_bitstream_buffer(buf_id) {
        Some(h) => h,
        None => return NV_ENC_ERR_INVALID_PARAM,
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
            let info = locked.get(&buf_id).unwrap();
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

// ── [18] nvEncUnlockBitstream ──────────────────────────────────────────

unsafe extern "C" fn interpose_unlock_bitstream(
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

// ── [19] nvEncLockInputBuffer ──────────────────────────────────────────

unsafe extern "C" fn interpose_lock_input_buffer(
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
        NvencResponse::InputBufferLocked { pitch, buffer_size } => {
            // Allocate a shadow buffer for the app to write pixel data into
            let data = vec![0u8; buffer_size as usize];
            let data_ptr = data.as_ptr() as *mut c_void;
            let mut locked = locked_input_buffers().lock();
            locked.insert(buf_id, LockedInputBufferInfo {
                data,
                pitch,
            });
            // The pointer must come from the stored data (not the local `data` which moved)
            let info = locked.get(&buf_id).unwrap();
            params.buffer_data_ptr = info.data.as_ptr() as *mut c_void;
            params.pitch = pitch;
            // Suppress unused variable warning
            let _ = data_ptr;
            NV_ENC_SUCCESS
        }
        NvencResponse::Error { code, .. } => code,
        _ => NV_ENC_ERR_GENERIC,
    }
}

// ── [20] nvEncUnlockInputBuffer ────────────────────────────────────────

unsafe extern "C" fn interpose_unlock_input_buffer(
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

// ── [21] nvEncGetEncodeStats ───────────────────────────────────────────

unsafe extern "C" fn interpose_get_encode_stats(
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

// ── [22] nvEncGetSequenceParams ────────────────────────────────────────

unsafe extern "C" fn interpose_get_sequence_params(
    encoder: *mut c_void,
    payload: *mut NvEncSequenceParamPayload,
) -> NVENCSTATUS {
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
            NV_ENC_SUCCESS
        }
        NvencResponse::Error { code, .. } => code,
        _ => NV_ENC_ERR_GENERIC,
    }
}

// ── [23] nvEncRegisterAsyncEvent ───────────────────────────────────────

unsafe extern "C" fn interpose_register_async_event(
    _encoder: *mut c_void,
    _event_params: *mut c_void,
) -> NVENCSTATUS {
    // Async events are not usable over the network
    NV_ENC_ERR_UNIMPLEMENTED
}

// ── [24] nvEncUnregisterAsyncEvent ─────────────────────────────────────

unsafe extern "C" fn interpose_unregister_async_event(
    _encoder: *mut c_void,
    _event_params: *mut c_void,
) -> NVENCSTATUS {
    // Async events are not usable over the network
    NV_ENC_ERR_UNIMPLEMENTED
}

// ── [25] nvEncMapInputResource ─────────────────────────────────────────

unsafe extern "C" fn interpose_map_input_resource(
    encoder: *mut c_void,
    map_input_resource_params: *mut NvEncMapInputResourceParams,
) -> NVENCSTATUS {
    if map_input_resource_params.is_null() {
        return NV_ENC_ERR_INVALID_PTR;
    }
    let enc_handle = match encoder_handle_from_ptr(encoder) {
        Some(h) => h,
        None => return NV_ENC_ERR_INVALID_PARAM,
    };

    let params = &mut *map_input_resource_params;
    let reg_id = params.input_resource as u64;
    let reg_handle = match handle_store::get_registered_resource(reg_id) {
        Some(h) => h,
        None => return NV_ENC_ERR_INVALID_PARAM,
    };

    let resp = send_nvenc_command(NvencCommand::MapInputResource {
        encoder: enc_handle,
        registered_resource: reg_handle,
    });
    match resp {
        NvencResponse::ResourceMapped { handle } => {
            let id = handle_store::store_mapped_resource(handle);
            params.mapped_resource = id as NV_ENC_INPUT_PTR;
            NV_ENC_SUCCESS
        }
        NvencResponse::Error { code, .. } => code,
        _ => NV_ENC_ERR_GENERIC,
    }
}

// ── [26] nvEncUnmapInputResource ───────────────────────────────────────

unsafe extern "C" fn interpose_unmap_input_resource(
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

// ── [27] nvEncDestroyEncoder ───────────────────────────────────────────

unsafe extern "C" fn interpose_destroy_encoder(
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

// ── [28] nvEncInvalidateRefFrames ──────────────────────────────────────

unsafe extern "C" fn interpose_invalidate_ref_frames(
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

// ── [29] nvEncOpenEncodeSessionEx ──────────────────────────────────────

unsafe extern "C" fn interpose_open_encode_session_ex(
    open_session_ex_params: *mut NvEncOpenEncodeSessionExParams,
    encoder: *mut *mut c_void,
) -> NVENCSTATUS {
    if open_session_ex_params.is_null() || encoder.is_null() {
        return NV_ENC_ERR_INVALID_PTR;
    }

    let params = &*open_session_ex_params;
    let device_type = params.device_type;

    // The device pointer is a CUcontext from our CUDA interpose.
    // Look it up in the CUDA interpose handle_store to get the NetworkHandle.
    // The device pointer was returned by cuCtxCreate as an opaque local ID.
    let cuda_ctx_id = params.device as u64;

    // We need the CUDA context's NetworkHandle. Since the NVENC interpose and
    // CUDA interpose are separate DLLs, we cannot directly access the CUDA
    // handle_store. Instead, we use a convention: the CUDA interpose returns
    // opaque IDs as pointers, and we pass this ID to the daemon which resolves
    // it server-side. We create a dummy NetworkHandle with the resource_id set
    // to the local CUDA context ID, and the daemon/server resolves it.
    //
    // Actually, the client daemon sees the CUDA context handle as stored in its
    // own handle maps from previous CUDA commands. We send the cuda_ctx_id
    // which the daemon can use to look up the real server-side handle.
    let cuda_context = NetworkHandle {
        server_id: 0,
        session_id: 0,
        resource_id: cuda_ctx_id,
        resource_type: rgpu_protocol::handle::ResourceType::CuContext,
    };

    let resp = send_nvenc_command(NvencCommand::OpenEncodeSession {
        cuda_context,
        device_type,
    });
    match resp {
        NvencResponse::EncoderOpened { handle } => {
            let id = handle_store::store_encoder(handle);
            *encoder = id as *mut c_void;
            NV_ENC_SUCCESS
        }
        NvencResponse::Error { code, .. } => code,
        _ => NV_ENC_ERR_GENERIC,
    }
}

// ── [30] nvEncRegisterResource ─────────────────────────────────────────

unsafe extern "C" fn interpose_register_resource(
    encoder: *mut c_void,
    register_resource_params: *mut NvEncRegisterResourceParams,
) -> NVENCSTATUS {
    if register_resource_params.is_null() {
        return NV_ENC_ERR_INVALID_PTR;
    }
    let enc_handle = match encoder_handle_from_ptr(encoder) {
        Some(h) => h,
        None => return NV_ENC_ERR_INVALID_PARAM,
    };

    let params = &mut *register_resource_params;

    // The resource_to_register is a CUDA device pointer (CUdeviceptr) or similar.
    // It was returned by our CUDA interpose as an opaque ID. Create a handle
    // that the daemon can resolve.
    let resource_id = params.resource_to_register as u64;
    let resource = NetworkHandle {
        server_id: 0,
        session_id: 0,
        resource_id,
        resource_type: rgpu_protocol::handle::ResourceType::CuDevicePtr,
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
            NV_ENC_SUCCESS
        }
        NvencResponse::Error { code, .. } => code,
        _ => NV_ENC_ERR_GENERIC,
    }
}

// ── [31] nvEncUnregisterResource ───────────────────────────────────────

unsafe extern "C" fn interpose_unregister_resource(
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

// ── [32] nvEncReconfigureEncoder ───────────────────────────────────────

unsafe extern "C" fn interpose_reconfigure_encoder(
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

    // Read struct size from version field
    let version = *(reconfig_params as *const u32);
    let struct_size = (version & 0xFFFF) as usize;
    let struct_size = if struct_size == 0 { 4096 } else { struct_size };

    let params = std::slice::from_raw_parts(reconfig_params as *const u8, struct_size).to_vec();

    let resp = send_nvenc_command(NvencCommand::ReconfigureEncoder {
        encoder: enc_handle,
        params,
    });
    response_to_status(&resp)
}

// ── [34] nvEncGetEncodePresetConfigEx ──────────────────────────────────

unsafe extern "C" fn interpose_get_encode_preset_config_ex(
    encoder: *mut c_void,
    encode_guid: NvEncGuid,
    preset_guid: NvEncGuid,
    _tuning_info: u32,
    preset_config: *mut c_void,
) -> NVENCSTATUS {
    if preset_config.is_null() {
        return NV_ENC_ERR_INVALID_PTR;
    }
    let enc_handle = match encoder_handle_from_ptr(encoder) {
        Some(h) => h,
        None => return NV_ENC_ERR_INVALID_PARAM,
    };

    // Reuse the same server-side command — the server handles both legacy and Ex variants
    let resp = send_nvenc_command(NvencCommand::GetEncodePresetConfig {
        encoder: enc_handle,
        encode_guid: encode_guid.to_nv_guid(),
        preset_guid: preset_guid.to_nv_guid(),
    });
    match resp {
        NvencResponse::PresetConfig(data) => {
            let dst = preset_config as *mut u8;
            let copy_len = data.len();
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, copy_len);
            NV_ENC_SUCCESS
        }
        NvencResponse::Error { code, .. } => code,
        _ => NV_ENC_ERR_GENERIC,
    }
}
