//! NVENC FFI type definitions and shadow buffer infrastructure.
//!
//! Contains all `#[repr(C)]` struct definitions used for NVENC FFI interception,
//! shadow buffer tracking for locked input/bitstream buffers, and input buffer
//! dimension tracking.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::OnceLock;

use parking_lot::Mutex;

use crate::{NV_ENC_INPUT_PTR, NV_ENC_OUTPUT_PTR, NV_ENC_REGISTERED_PTR};

// ── NVENC FFI structs ──────────────────────────────────────────────────

/// NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS
#[repr(C)]
pub(crate) struct NvEncOpenEncodeSessionExParams {
    pub(crate) version: u32,
    pub(crate) device_type: u32,
    pub(crate) device: *mut c_void,
    pub(crate) reserved: *mut c_void,
    pub(crate) api_version: u32,
    pub(crate) reserved1: [u32; 253],
    pub(crate) reserved2: [*mut c_void; 64],
}

/// NV_ENC_CAPS_PARAM
#[repr(C)]
pub(crate) struct NvEncCapsParam {
    pub(crate) version: u32,
    pub(crate) caps_to_query: u32,
    pub(crate) reserved: [u32; 62],
}

/// NV_ENC_CREATE_INPUT_BUFFER
#[repr(C)]
pub(crate) struct NvEncCreateInputBufferParams {
    pub(crate) version: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) memory_heap: u32,
    pub(crate) buffer_fmt: u32,
    pub(crate) reserved: u32,
    pub(crate) input_buffer: NV_ENC_INPUT_PTR,
    pub(crate) p_sys_mem: *mut c_void,
    pub(crate) reserved1: [u32; 57],
    pub(crate) reserved2: [*mut c_void; 63],
}

/// NV_ENC_CREATE_BITSTREAM_BUFFER
#[repr(C)]
pub(crate) struct NvEncCreateBitstreamBufferParams {
    pub(crate) version: u32,
    pub(crate) encoder_buffer: u32,                // deprecated
    pub(crate) reserved_pad: *mut c_void,          // padding (driver skips this, writes at offset 16)
    pub(crate) bitstream_buffer: NV_ENC_OUTPUT_PTR, // OUT: actual pointer at offset 16
    pub(crate) reserved_size: u32,
    pub(crate) reserved1: [u32; 57],
    pub(crate) reserved2: [*mut c_void; 63],
}

/// NV_ENC_LOCK_INPUT_BUFFER
#[repr(C)]
pub(crate) struct NvEncLockInputBufferParams {
    pub(crate) version: u32,
    pub(crate) reserved_flags: u32,
    pub(crate) input_buffer: NV_ENC_INPUT_PTR,
    pub(crate) buffer_data_ptr: *mut c_void,
    pub(crate) pitch: u32,
    pub(crate) reserved1: [u32; 251],
    pub(crate) reserved2: [*mut c_void; 64],
}

/// NV_ENC_LOCK_BITSTREAM
#[repr(C)]
pub(crate) struct NvEncLockBitstreamParams {
    pub(crate) version: u32,
    pub(crate) do_not_wait: u32,
    pub(crate) output_bitstream: NV_ENC_OUTPUT_PTR,
    pub(crate) slice_offsets: *mut u32,
    pub(crate) frame_idx: u32,
    pub(crate) hw_encode_status: u32,
    pub(crate) num_slices: u32,
    pub(crate) bitstream_size_in_bytes: u32,
    pub(crate) output_timestamp: u64,
    pub(crate) output_duration: u64,
    pub(crate) bitstream_buffer_ptr: *mut c_void,
    pub(crate) picture_type: u32,
    pub(crate) picture_struct: u32,
    pub(crate) frame_avg_qp: u32,
    pub(crate) frame_satd: u32,
    pub(crate) ltr_frame_idx: u32,
    pub(crate) ltr_frame_bitmap: u32,
    pub(crate) reserved: [u32; 13],
    pub(crate) intra_refresh_cnt: u32,
    pub(crate) reserved1: [u32; 219],
    pub(crate) reserved2: [*mut c_void; 64],
}

/// NV_ENC_REGISTER_RESOURCE
#[repr(C)]
pub(crate) struct NvEncRegisterResourceParams {
    pub(crate) version: u32,
    pub(crate) resource_type: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) pitch: u32,
    pub(crate) sub_resource_index: u32,
    pub(crate) resource_to_register: *mut c_void,
    pub(crate) registered_resource: NV_ENC_REGISTERED_PTR,
    pub(crate) buffer_format: u32,
    pub(crate) buffer_usage: u32,
    pub(crate) reserved1: [u32; 247],
    pub(crate) reserved2: [*mut c_void; 62],
}

/// NV_ENC_MAP_INPUT_RESOURCE
#[repr(C)]
pub(crate) struct NvEncMapInputResourceParams {
    pub(crate) version: u32,
    pub(crate) sub_resource_index: u32,
    pub(crate) input_resource: NV_ENC_REGISTERED_PTR,
    pub(crate) mapped_resource: NV_ENC_INPUT_PTR,
    pub(crate) mapped_buffer_fmt: u32,
    pub(crate) reserved1: [u32; 255],
    pub(crate) reserved2: [*mut c_void; 63],
}

/// NV_ENC_PIC_PARAMS (simplified — we send the raw bytes over the wire)
#[repr(C)]
pub(crate) struct NvEncPicParams {
    pub(crate) version: u32,
    pub(crate) input_width: u32,
    pub(crate) input_height: u32,
    pub(crate) input_pitch: u32,
    pub(crate) encode_params_flags: u32,
    pub(crate) frame_idx: u32,
    pub(crate) input_timestamp: u64,
    pub(crate) input_duration: u64,
    pub(crate) input_buffer: NV_ENC_INPUT_PTR,
    pub(crate) output_bitstream: NV_ENC_OUTPUT_PTR,
    pub(crate) completion_event: *mut c_void,
    pub(crate) buffer_fmt: u32,
    pub(crate) picture_struct: u32,
    pub(crate) picture_type: u32,
    // Remaining fields not needed for our interpose — raw bytes sent
}

/// NV_ENC_SEQUENCE_PARAM_PAYLOAD
#[repr(C)]
pub(crate) struct NvEncSequenceParamPayload {
    pub(crate) version: u32,
    pub(crate) in_buffer_size: u32,
    pub(crate) sps_id: u32,
    pub(crate) pps_id: u32,
    pub(crate) sp_spps_buffer: *mut c_void,
    pub(crate) out_sps_pps_payload_size: *mut u32,
    pub(crate) reserved1: [u32; 250],
    pub(crate) reserved2: [*mut c_void; 64],
}

/// NV_ENC_INITIALIZE_PARAMS layout on 64-bit to determine size and pointer offsets.
/// We only need this for sizeof and field offsets; fields themselves are sent as raw bytes.
#[repr(C)]
pub(crate) struct NvEncInitializeParamsLayout {
    pub(crate) version: u32,
    pub(crate) encode_guid: [u8; 16],
    pub(crate) preset_guid: [u8; 16],
    pub(crate) encode_width: u32,
    pub(crate) encode_height: u32,
    pub(crate) dar_width: u32,
    pub(crate) dar_height: u32,
    pub(crate) frame_rate_num: u32,
    pub(crate) frame_rate_den: u32,
    pub(crate) enable_encode_async: u32,
    pub(crate) enable_ptd: u32,
    pub(crate) bitfields: u32,
    pub(crate) priv_data_size: u32,
    pub(crate) _reserved_pad: u32,            // explicit padding / reserved (SDK 13.0)
    pub(crate) priv_data: *mut c_void,        // offset 80 on 64-bit
    pub(crate) encode_config: *mut c_void,    // offset 88 on 64-bit
    pub(crate) max_encode_width: u32,
    pub(crate) max_encode_height: u32,
    pub(crate) me_hints: [u64; 2],           // NVENC_EXTERNAL_ME_HINT_COUNTS_PER_BLOCKTYPE[2]
    pub(crate) tuning_info: u32,
    pub(crate) buffer_format: u32,
    pub(crate) num_state_buffers: u32,
    pub(crate) output_stats_level: u32,
    pub(crate) reserved1: [u32; 284],
    pub(crate) reserved2: [*mut c_void; 64],
}

// ── Shadow buffers for LockInputBuffer ─────────────────────────────────
// When we lock an input buffer, the app writes pixel data into it. We need a
// local buffer to store that data, then send it via IPC on unlock.

pub(crate) struct LockedInputBufferInfo {
    pub(crate) data: Vec<u8>,
    pub(crate) pitch: u32,
}

static LOCKED_INPUT_BUFFERS: OnceLock<Mutex<HashMap<u64, LockedInputBufferInfo>>> = OnceLock::new();

pub(crate) fn locked_input_buffers() -> &'static Mutex<HashMap<u64, LockedInputBufferInfo>> {
    LOCKED_INPUT_BUFFERS.get_or_init(|| Mutex::new(HashMap::new()))
}

// ── Input buffer dimension tracking ────────────────────────────────────
// We track height and format per input buffer so LockInputBuffer can
// calculate the correct shadow buffer size from pitch + height + format.

pub(crate) struct InputBufferDims {
    pub(crate) height: u32,
    pub(crate) buffer_fmt: u32,
}

static INPUT_BUFFER_DIMS: OnceLock<Mutex<HashMap<u64, InputBufferDims>>> = OnceLock::new();

pub(crate) fn input_buffer_dims() -> &'static Mutex<HashMap<u64, InputBufferDims>> {
    INPUT_BUFFER_DIMS.get_or_init(|| Mutex::new(HashMap::new()))
}

// ── Shadow buffers for LockBitstream ───────────────────────────────────
// When we lock a bitstream buffer, the server sends us encoded data. We keep
// it in a local buffer until the app calls unlock.

pub(crate) struct LockedBitstreamInfo {
    pub(crate) data: Vec<u8>,
    #[allow(dead_code)]
    pub(crate) picture_type: u32,
    #[allow(dead_code)]
    pub(crate) frame_idx: u32,
    #[allow(dead_code)]
    pub(crate) output_timestamp: u64,
}

static LOCKED_BITSTREAMS: OnceLock<Mutex<HashMap<u64, LockedBitstreamInfo>>> = OnceLock::new();

pub(crate) fn locked_bitstreams() -> &'static Mutex<HashMap<u64, LockedBitstreamInfo>> {
    LOCKED_BITSTREAMS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Calculate total buffer size from pitch, height, and NVENC buffer format.
pub(crate) fn calc_buffer_size(pitch: u32, height: u32, buffer_fmt: u32) -> usize {
    let p = pitch as usize;
    let h = height as usize;
    match buffer_fmt {
        // NV12, YV12, IYUV — 4:2:0 planar/semi-planar: 1.5x height
        0x1 | 0x10 | 0x100 => p * h * 3 / 2,
        // YUV420_10BIT — 4:2:0 10-bit
        0x10000 => p * h * 3 / 2,
        // YUV444 — 3 full planes
        0x1000 => p * h * 3,
        // YUV444_10BIT — 3 full 10-bit planes
        0x100000 => p * h * 3,
        // ARGB, ABGR, ARGB10, ABGR10, AYUV — pitch already includes 4 bytes/pixel
        0x1000000 | 0x10000000 | 0x2000000 | 0x20000000 | 0x4000000 => p * h,
        // Unknown — generous fallback
        _ => p * h * 4,
    }
}
