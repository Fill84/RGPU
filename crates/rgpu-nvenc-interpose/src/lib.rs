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
mod types;
mod encoder;
mod buffer;
mod stubs;

use std::collections::HashMap;
use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

use parking_lot::Mutex;
use tracing::{debug, error};

use rgpu_protocol::handle::{NetworkHandle, ResourceType};
use rgpu_protocol::nvenc_commands::{NvGuid, NvencCommand, NvencResponse};

use ipc_client::NvencIpcClient;

// ── Cross-DLL CUDA handle resolution ───────────────────────────────────
// The CUDA interpose (nvcuda.dll) exports rgpu_cuda_resolve_ctx/mem functions
// that let us look up NetworkHandles for CUDA local IDs.

type CudaResolveFn = unsafe extern "C" fn(u64, *mut u16, *mut u32, *mut u64) -> c_int;

static CUDA_RESOLVE_CTX: OnceLock<Option<CudaResolveFn>> = OnceLock::new();
static CUDA_RESOLVE_MEM: OnceLock<Option<CudaResolveFn>> = OnceLock::new();

fn load_cuda_resolve_fn(name: &[u8]) -> Option<CudaResolveFn> {
    unsafe {
        #[cfg(target_os = "windows")]
        let lib_name = "nvcuda.dll";
        #[cfg(not(target_os = "windows"))]
        let lib_name = "libcuda.so.1";

        // Use RTLD_NOLOAD equivalent - the CUDA interpose should already be loaded
        let lib = libloading::Library::new(lib_name).ok()?;
        let sym: libloading::Symbol<CudaResolveFn> = lib.get(name).ok()?;
        let func = *sym;
        // Leak the library to keep it loaded
        std::mem::forget(lib);
        Some(func)
    }
}

pub(crate) fn resolve_cuda_ctx_handle(local_id: u64) -> Option<NetworkHandle> {
    let func = CUDA_RESOLVE_CTX.get_or_init(|| load_cuda_resolve_fn(b"rgpu_cuda_resolve_ctx"));
    if let Some(func) = func {
        let mut server_id: u16 = 0;
        let mut session_id: u32 = 0;
        let mut resource_id: u64 = 0;
        let result = unsafe { func(local_id, &mut server_id, &mut session_id, &mut resource_id) };
        if result != 0 {
            Some(NetworkHandle {
                server_id,
                session_id,
                resource_id,
                resource_type: ResourceType::CuContext,
            })
        } else {
            None
        }
    } else {
        None
    }
}

pub(crate) fn resolve_cuda_mem_handle(local_id: u64) -> Option<NetworkHandle> {
    let func = CUDA_RESOLVE_MEM.get_or_init(|| load_cuda_resolve_fn(b"rgpu_cuda_resolve_mem"));
    if let Some(func) = func {
        let mut server_id: u16 = 0;
        let mut session_id: u32 = 0;
        let mut resource_id: u64 = 0;
        let result = unsafe { func(local_id, &mut server_id, &mut session_id, &mut resource_id) };
        if result != 0 {
            Some(NetworkHandle {
                server_id,
                session_id,
                resource_id,
                resource_type: ResourceType::CuDevicePtr,
            })
        } else {
            None
        }
    } else {
        None
    }
}

// ── NVENC types ────────────────────────────────────────────────────────

pub(crate) type NVENCSTATUS = i32;
pub(crate) type NV_ENC_INPUT_PTR = *mut c_void;
pub(crate) type NV_ENC_OUTPUT_PTR = *mut c_void;
pub(crate) type NV_ENC_REGISTERED_PTR = *mut c_void;

pub(crate) const NV_ENC_SUCCESS: NVENCSTATUS = 0;
pub(crate) const NV_ENC_ERR_INVALID_DEVICE: NVENCSTATUS = 4;
pub(crate) const NV_ENC_ERR_INVALID_PTR: NVENCSTATUS = 6;
pub(crate) const NV_ENC_ERR_INVALID_PARAM: NVENCSTATUS = 8;
pub(crate) const NV_ENC_ERR_GENERIC: NVENCSTATUS = 20;
pub(crate) const NV_ENC_ERR_UNIMPLEMENTED: NVENCSTATUS = 22;

/// NVENC API version: (major << 4) | minor. We advertise 12.2.
const NVENC_API_VERSION: u32 = (12 << 4) | 2;

// ── GUID type matching NVENC's GUID layout ─────────────────────────────

/// NVENC GUID structure (matches Windows GUID layout).
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct NvEncGuid {
    pub(crate) data1: u32,
    pub(crate) data2: u16,
    pub(crate) data3: u16,
    pub(crate) data4: [u8; 8],
}

impl NvEncGuid {
    pub(crate) fn to_nv_guid(&self) -> NvGuid {
        let mut bytes = [0u8; 16];
        bytes[0..4].copy_from_slice(&self.data1.to_le_bytes());
        bytes[4..6].copy_from_slice(&self.data2.to_le_bytes());
        bytes[6..8].copy_from_slice(&self.data3.to_le_bytes());
        bytes[8..16].copy_from_slice(&self.data4);
        NvGuid(bytes)
    }

    pub(crate) fn from_nv_guid(guid: &NvGuid) -> Self {
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

// ── NV_ENCODE_API_FUNCTION_LIST (the vtable) ───────────────────────────

/// The NVENC function table that applications receive from NvEncodeAPICreateInstance.
/// This matches the NV_ENCODE_API_FUNCTION_LIST layout from nvEncodeAPI.h (SDK 12.x/13.0).
///
/// CRITICAL: The field order MUST match the SDK header exactly. The SDK order is:
///   nvEncOpenEncodeSession, nvEncGetEncodeGUIDCount, nvEncGetEncodeProfileGUIDCount,
///   nvEncGetEncodeProfileGUIDs, nvEncGetEncodeGUIDs, nvEncGetInputFormatCount, ...
/// Note: ProfileGUID functions come BEFORE GetEncodeGUIDs in the SDK layout!
#[repr(C)]
pub struct NvEncApiFunctionList {
    version: u32,
    reserved: u32,
    // [0]  nvEncOpenEncodeSession (legacy)
    nv_enc_open_encode_session: Option<unsafe extern "C" fn(*mut c_void, u32, *mut *mut c_void) -> NVENCSTATUS>,
    // [1]  nvEncGetEncodeGUIDCount
    nv_enc_get_encode_guid_count: Option<unsafe extern "C" fn(*mut c_void, *mut u32) -> NVENCSTATUS>,
    // [2]  nvEncGetEncodeProfileGUIDCount
    nv_enc_get_encode_profile_guid_count: Option<unsafe extern "C" fn(*mut c_void, NvEncGuid, *mut u32) -> NVENCSTATUS>,
    // [3]  nvEncGetEncodeProfileGUIDs
    nv_enc_get_encode_profile_guids: Option<unsafe extern "C" fn(*mut c_void, NvEncGuid, *mut NvEncGuid, u32, *mut u32) -> NVENCSTATUS>,
    // [4]  nvEncGetEncodeGUIDs
    nv_enc_get_encode_guids: Option<unsafe extern "C" fn(*mut c_void, *mut NvEncGuid, u32, *mut u32) -> NVENCSTATUS>,
    // [5]  nvEncGetInputFormatCount
    nv_enc_get_input_format_count: Option<unsafe extern "C" fn(*mut c_void, NvEncGuid, *mut u32) -> NVENCSTATUS>,
    // [6]  nvEncGetInputFormats
    nv_enc_get_input_formats: Option<unsafe extern "C" fn(*mut c_void, NvEncGuid, *mut u32, u32, *mut u32) -> NVENCSTATUS>,
    // [7]  nvEncGetEncodeCaps
    nv_enc_get_encode_caps: Option<unsafe extern "C" fn(*mut c_void, NvEncGuid, *mut types::NvEncCapsParam, *mut c_int) -> NVENCSTATUS>,
    // [8]  nvEncGetEncodePresetCount
    nv_enc_get_encode_preset_count: Option<unsafe extern "C" fn(*mut c_void, NvEncGuid, *mut u32) -> NVENCSTATUS>,
    // [9]  nvEncGetEncodePresetGUIDs
    nv_enc_get_encode_preset_guids: Option<unsafe extern "C" fn(*mut c_void, NvEncGuid, *mut NvEncGuid, u32, *mut u32) -> NVENCSTATUS>,
    // [10] nvEncGetEncodePresetConfig (legacy)
    nv_enc_get_encode_preset_config: Option<unsafe extern "C" fn(*mut c_void, NvEncGuid, NvEncGuid, *mut c_void) -> NVENCSTATUS>,
    // [11] nvEncInitializeEncoder
    nv_enc_initialize_encoder: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NVENCSTATUS>,
    // [12] nvEncCreateInputBuffer
    nv_enc_create_input_buffer: Option<unsafe extern "C" fn(*mut c_void, *mut types::NvEncCreateInputBufferParams) -> NVENCSTATUS>,
    // [13] nvEncDestroyInputBuffer
    nv_enc_destroy_input_buffer: Option<unsafe extern "C" fn(*mut c_void, NV_ENC_INPUT_PTR) -> NVENCSTATUS>,
    // [14] nvEncCreateBitstreamBuffer
    nv_enc_create_bitstream_buffer: Option<unsafe extern "C" fn(*mut c_void, *mut types::NvEncCreateBitstreamBufferParams) -> NVENCSTATUS>,
    // [15] nvEncDestroyBitstreamBuffer
    nv_enc_destroy_bitstream_buffer: Option<unsafe extern "C" fn(*mut c_void, NV_ENC_OUTPUT_PTR) -> NVENCSTATUS>,
    // [16] nvEncEncodePicture
    nv_enc_encode_picture: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NVENCSTATUS>,
    // [17] nvEncLockBitstream
    nv_enc_lock_bitstream: Option<unsafe extern "C" fn(*mut c_void, *mut types::NvEncLockBitstreamParams) -> NVENCSTATUS>,
    // [18] nvEncUnlockBitstream
    nv_enc_unlock_bitstream: Option<unsafe extern "C" fn(*mut c_void, NV_ENC_OUTPUT_PTR) -> NVENCSTATUS>,
    // [19] nvEncLockInputBuffer
    nv_enc_lock_input_buffer: Option<unsafe extern "C" fn(*mut c_void, *mut types::NvEncLockInputBufferParams) -> NVENCSTATUS>,
    // [20] nvEncUnlockInputBuffer
    nv_enc_unlock_input_buffer: Option<unsafe extern "C" fn(*mut c_void, NV_ENC_INPUT_PTR) -> NVENCSTATUS>,
    // [21] nvEncGetEncodeStats
    nv_enc_get_encode_stats: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NVENCSTATUS>,
    // [22] nvEncGetSequenceParams
    nv_enc_get_sequence_params: Option<unsafe extern "C" fn(*mut c_void, *mut types::NvEncSequenceParamPayload) -> NVENCSTATUS>,
    // [23] nvEncRegisterAsyncEvent
    nv_enc_register_async_event: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NVENCSTATUS>,
    // [24] nvEncUnregisterAsyncEvent
    nv_enc_unregister_async_event: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NVENCSTATUS>,
    // [25] nvEncMapInputResource
    nv_enc_map_input_resource: Option<unsafe extern "C" fn(*mut c_void, *mut types::NvEncMapInputResourceParams) -> NVENCSTATUS>,
    // [26] nvEncUnmapInputResource
    nv_enc_unmap_input_resource: Option<unsafe extern "C" fn(*mut c_void, NV_ENC_INPUT_PTR) -> NVENCSTATUS>,
    // [27] nvEncDestroyEncoder
    nv_enc_destroy_encoder: Option<unsafe extern "C" fn(*mut c_void) -> NVENCSTATUS>,
    // [28] nvEncInvalidateRefFrames
    nv_enc_invalidate_ref_frames: Option<unsafe extern "C" fn(*mut c_void, u64) -> NVENCSTATUS>,
    // [29] nvEncOpenEncodeSessionEx
    nv_enc_open_encode_session_ex: Option<unsafe extern "C" fn(*mut types::NvEncOpenEncodeSessionExParams, *mut *mut c_void) -> NVENCSTATUS>,
    // [30] nvEncRegisterResource
    nv_enc_register_resource: Option<unsafe extern "C" fn(*mut c_void, *mut types::NvEncRegisterResourceParams) -> NVENCSTATUS>,
    // [31] nvEncUnregisterResource
    nv_enc_unregister_resource: Option<unsafe extern "C" fn(*mut c_void, NV_ENC_REGISTERED_PTR) -> NVENCSTATUS>,
    // [32] nvEncReconfigureEncoder
    nv_enc_reconfigure_encoder: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NVENCSTATUS>,
    // [33] reserved1
    reserved1: *mut c_void,
    // [34] nvEncCreateMVBuffer
    nv_enc_create_mv_buffer: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NVENCSTATUS>,
    // [35] nvEncDestroyMVBuffer
    nv_enc_destroy_mv_buffer: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NVENCSTATUS>,
    // [36] nvEncRunMotionEstimationOnly
    nv_enc_run_motion_estimation_only: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NVENCSTATUS>,
    // [37] nvEncGetLastErrorString
    nv_enc_get_last_error_string: Option<unsafe extern "C" fn(*mut c_void) -> *const u8>,
    // [38] nvEncSetIOCudaStreams
    nv_enc_set_io_cuda_streams: Option<unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> NVENCSTATUS>,
    // [39] nvEncGetEncodePresetConfigEx
    nv_enc_get_encode_preset_config_ex: Option<unsafe extern "C" fn(*mut c_void, NvEncGuid, NvEncGuid, u32, *mut c_void) -> NVENCSTATUS>,
    // [40] nvEncGetSequenceParamEx
    nv_enc_get_sequence_param_ex: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NVENCSTATUS>,
}

// ── IPC client singleton ───────────────────────────────────────────────

static IPC_CLIENT: OnceLock<NvencIpcClient> = OnceLock::new();

fn get_client() -> &'static NvencIpcClient {
    IPC_CLIENT.get_or_init(|| {
        let path = rgpu_common::platform::resolve_ipc_address();
        NvencIpcClient::new(&path)
    })
}

pub(crate) fn send_nvenc_command(cmd: NvencCommand) -> NvencResponse {
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
pub(crate) fn response_to_status(resp: &NvencResponse) -> NVENCSTATUS {
    match resp {
        NvencResponse::Success => NV_ENC_SUCCESS,
        NvencResponse::Error { code, .. } => *code,
        _ => NV_ENC_SUCCESS,
    }
}

/// Extract the encoder NetworkHandle from a local opaque encoder pointer.
pub(crate) fn encoder_handle_from_ptr(encoder: *mut c_void) -> Option<NetworkHandle> {
    let id = encoder as u64;
    handle_store::get_encoder(id)
}

// ── Input-to-output buffer pairing ─────────────────────────────────────
// ffmpeg creates input and bitstream buffers in pairs. We track the last
// created input buffer so we can pair it with the next bitstream buffer.
// This is needed because some ffmpeg versions don't set outputBitstream
// in NV_ENC_PIC_PARAMS.

pub(crate) static LAST_INPUT_BUFFER_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static INPUT_TO_OUTPUT_MAP: OnceLock<Mutex<HashMap<u64, u64>>> = OnceLock::new();

pub(crate) fn input_to_output_map() -> &'static Mutex<HashMap<u64, u64>> {
    INPUT_TO_OUTPUT_MAP.get_or_init(|| Mutex::new(HashMap::new()))
}

// ── Marker function for RGPU interpose DLL detection ───────────────────

/// Detection marker function for installer/daemon verification.
/// Returns 1 to indicate this is the RGPU interpose library.
#[no_mangle]
pub extern "C" fn rgpu_interpose_marker() -> c_int {
    1
}

// ── Static for GetLastErrorString stub ─────────────────────────────────

/// Stub for nvEncGetLastErrorString. Returns a static empty string.
/// The real driver's error strings are not accessible over the network.
pub(crate) static EMPTY_ERROR_STRING: [u8; 1] = [0u8; 1];

// ── Exported NVENC API Functions ───────────────────────────────────────

/// NvEncodeAPIGetMaxSupportedVersion - returns the maximum supported NVENC API version.
#[allow(non_snake_case)]
unsafe fn NvEncodeAPIGetMaxSupportedVersion_impl(version: *mut u32) -> NVENCSTATUS {
    // Initialize logging on first call
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("RGPU_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .try_init();

    debug!("NVENC GetMaxSupportedVersion: CALLED");

    if version.is_null() {
        return NV_ENC_ERR_INVALID_PTR;
    }

    let resp = send_nvenc_command(NvencCommand::GetMaxSupportedVersion);
    debug!("NVENC GetMaxSupportedVersion: got response: {:?}", resp);
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

#[no_mangle]
pub unsafe extern "C" fn NvEncodeAPIGetMaxSupportedVersion(version: *mut u32) -> NVENCSTATUS {
    rgpu_common::ffi::catch_panic(NV_ENC_ERR_GENERIC, || NvEncodeAPIGetMaxSupportedVersion_impl(version))
}

/// NvEncodeAPICreateInstance - fills the function table with our interceptor functions.
#[allow(non_snake_case)]
unsafe fn NvEncodeAPICreateInstance_impl(
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

    debug!("NVENC CreateInstance: fl version={:#x}, our struct size={}, ptr={:p}",
        fl.version, std::mem::size_of::<NvEncApiFunctionList>(), function_list);

    // Fill the function table with our interceptors
    fl.nv_enc_open_encode_session = Some(encoder::interpose_open_encode_session);
    fl.nv_enc_get_encode_guid_count = Some(encoder::interpose_get_encode_guid_count);
    fl.nv_enc_get_encode_guids = Some(encoder::interpose_get_encode_guids);
    fl.nv_enc_get_encode_profile_guid_count = Some(encoder::interpose_get_encode_profile_guid_count);
    fl.nv_enc_get_encode_profile_guids = Some(encoder::interpose_get_encode_profile_guids);
    fl.nv_enc_get_input_format_count = Some(encoder::interpose_get_input_format_count);
    fl.nv_enc_get_input_formats = Some(encoder::interpose_get_input_formats);
    fl.nv_enc_get_encode_caps = Some(encoder::interpose_get_encode_caps);
    fl.nv_enc_get_encode_preset_count = Some(encoder::interpose_get_encode_preset_count);
    fl.nv_enc_get_encode_preset_guids = Some(encoder::interpose_get_encode_preset_guids);
    fl.nv_enc_get_encode_preset_config = Some(encoder::interpose_get_encode_preset_config);
    fl.nv_enc_initialize_encoder = Some(encoder::interpose_initialize_encoder);
    fl.nv_enc_create_input_buffer = Some(buffer::interpose_create_input_buffer);
    fl.nv_enc_destroy_input_buffer = Some(buffer::interpose_destroy_input_buffer);
    fl.nv_enc_create_bitstream_buffer = Some(buffer::interpose_create_bitstream_buffer);
    fl.nv_enc_destroy_bitstream_buffer = Some(buffer::interpose_destroy_bitstream_buffer);
    fl.nv_enc_encode_picture = Some(encoder::interpose_encode_picture);
    fl.nv_enc_lock_bitstream = Some(buffer::interpose_lock_bitstream);
    fl.nv_enc_unlock_bitstream = Some(buffer::interpose_unlock_bitstream);
    fl.nv_enc_lock_input_buffer = Some(buffer::interpose_lock_input_buffer);
    fl.nv_enc_unlock_input_buffer = Some(buffer::interpose_unlock_input_buffer);
    fl.nv_enc_get_encode_stats = Some(stubs::interpose_get_encode_stats);
    fl.nv_enc_get_sequence_params = Some(stubs::interpose_get_sequence_params);
    fl.nv_enc_register_async_event = Some(stubs::interpose_register_async_event);
    fl.nv_enc_unregister_async_event = Some(stubs::interpose_unregister_async_event);
    fl.nv_enc_map_input_resource = Some(buffer::interpose_map_input_resource);
    fl.nv_enc_unmap_input_resource = Some(buffer::interpose_unmap_input_resource);
    fl.nv_enc_destroy_encoder = Some(stubs::interpose_destroy_encoder);
    fl.nv_enc_invalidate_ref_frames = Some(stubs::interpose_invalidate_ref_frames);
    fl.nv_enc_open_encode_session_ex = Some(encoder::interpose_open_encode_session_ex);
    fl.nv_enc_register_resource = Some(buffer::interpose_register_resource);
    fl.nv_enc_unregister_resource = Some(buffer::interpose_unregister_resource);
    fl.nv_enc_reconfigure_encoder = Some(stubs::interpose_reconfigure_encoder);
    fl.reserved1 = std::ptr::null_mut();
    // Slots 34-36: motion vector functions (stub — not needed for encoding)
    fl.nv_enc_create_mv_buffer = Some(stubs::interpose_stub_create_mv_buffer);
    fl.nv_enc_destroy_mv_buffer = Some(stubs::interpose_stub_destroy_mv_buffer);
    fl.nv_enc_run_motion_estimation_only = Some(stubs::interpose_stub_run_motion_estimation_only);
    // Slot 37: nvEncGetLastErrorString
    fl.nv_enc_get_last_error_string = Some(stubs::interpose_get_last_error_string);
    // Slot 38: nvEncSetIOCudaStreams
    fl.nv_enc_set_io_cuda_streams = Some(stubs::interpose_set_io_cuda_streams);
    // Slot 39: nvEncGetEncodePresetConfigEx (used by modern ffmpeg)
    fl.nv_enc_get_encode_preset_config_ex = Some(encoder::interpose_get_encode_preset_config_ex);
    // Slot 40: nvEncGetSequenceParamEx (stub)
    fl.nv_enc_get_sequence_param_ex = Some(stubs::interpose_stub_get_sequence_param_ex);

    // Diagnostic: dump function pointer values at critical slot offsets
    // This lets us verify the CORRECT DLL is loaded and slots are properly filled
    let raw_ptr = function_list as *const u8;
    // Slot 38 = SetIOCudaStreams: offset = 8 + 38*8 = 312
    let slot38_val = *(raw_ptr.add(312) as *const usize);
    let our_set_streams = stubs::interpose_set_io_cuda_streams as *const () as usize;
    // Slot 37 = GetLastErrorString: offset = 8 + 37*8 = 304
    let slot37_val = *(raw_ptr.add(304) as *const usize);
    // Slot 11 = InitializeEncoder: offset = 8 + 11*8 = 96
    let slot11_val = *(raw_ptr.add(96) as *const usize);
    debug!("NVENC CreateInstance BUILD=2026-02-27T2: slot11(InitEnc)={:#x} slot37(GetErr)={:#x} slot38(SetStreams)={:#x} expected={:#x} match={}",
        slot11_val, slot37_val, slot38_val, our_set_streams, slot38_val == our_set_streams);

    NV_ENC_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn NvEncodeAPICreateInstance(
    function_list: *mut NvEncApiFunctionList,
) -> NVENCSTATUS {
    rgpu_common::ffi::catch_panic(NV_ENC_ERR_GENERIC, || NvEncodeAPICreateInstance_impl(function_list))
}
