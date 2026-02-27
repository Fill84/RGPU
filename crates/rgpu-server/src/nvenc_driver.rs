//! Dynamic loading of the NVIDIA Video Encoder (NVENC) library.
//!
//! Uses `libloading` to load `nvEncodeAPI64.dll` (Windows) or `libnvidia-encode.so.1` (Linux)
//! and provides safe Rust wrappers around the NVENC API via its function table.
//!
//! The NVENC API uses a single entry point (`NvEncodeAPICreateInstance`) that populates
//! a function table struct with pointers to all encoder functions.

use std::ffi::{c_int, c_void};
use std::sync::Arc;

use libloading::{Library, Symbol};
use tracing::{debug, info, warn};

/// NVENC status code (0 = NV_ENC_SUCCESS).
pub type NvencStatus = i32;

pub const NV_ENC_SUCCESS: NvencStatus = 0;
pub const NV_ENC_ERR_NO_ENCODE_DEVICE: NvencStatus = 1;
pub const NV_ENC_ERR_UNSUPPORTED_DEVICE: NvencStatus = 2;
pub const NV_ENC_ERR_INVALID_ENCODERDEVICE: NvencStatus = 3;
pub const NV_ENC_ERR_INVALID_DEVICE: NvencStatus = 4;
pub const NV_ENC_ERR_DEVICE_NOT_EXIST: NvencStatus = 5;
pub const NV_ENC_ERR_INVALID_PTR: NvencStatus = 6;
pub const NV_ENC_ERR_INVALID_EVENT: NvencStatus = 7;
pub const NV_ENC_ERR_INVALID_PARAM: NvencStatus = 8;
pub const NV_ENC_ERR_INVALID_CALL: NvencStatus = 9;
pub const NV_ENC_ERR_OUT_OF_MEMORY: NvencStatus = 10;
pub const NV_ENC_ERR_ENCODER_NOT_INITIALIZED: NvencStatus = 11;
pub const NV_ENC_ERR_UNSUPPORTED_PARAM: NvencStatus = 12;
pub const NV_ENC_ERR_LOCK_BUSY: NvencStatus = 13;
pub const NV_ENC_ERR_NOT_ENOUGH_BUFFER: NvencStatus = 14;
pub const NV_ENC_ERR_INVALID_VERSION: NvencStatus = 15;
pub const NV_ENC_ERR_MAP_FAILED: NvencStatus = 16;
pub const NV_ENC_ERR_NEED_MORE_INPUT: NvencStatus = 17;
pub const NV_ENC_ERR_ENCODER_BUSY: NvencStatus = 18;
pub const NV_ENC_ERR_EVENT_NOT_REGISTERD: NvencStatus = 19;
pub const NV_ENC_ERR_GENERIC: NvencStatus = 20;
pub const NV_ENC_ERR_INCOMPATIBLE_CLIENT_KEY: NvencStatus = 21;
pub const NV_ENC_ERR_UNIMPLEMENTED: NvencStatus = 22;
pub const NV_ENC_ERR_RESOURCE_REGISTER_FAILED: NvencStatus = 23;
pub const NV_ENC_ERR_RESOURCE_NOT_REGISTERED: NvencStatus = 24;
pub const NV_ENC_ERR_RESOURCE_NOT_MAPPED: NvencStatus = 25;

/// NVENC API version for struct versioning and api_version fields: major | (minor << 24).
/// We target NVENC API version 12.2.
const NVENCAPI_VERSION: u32 = 12 | (2 << 24);

/// Struct version macro: NVENCAPI_VERSION | (struct_ver << 16) | (0x7 << 28).
/// The `ver` parameter is the struct version number (e.g. 1 for most structs, 2 for function list).
fn nvenc_struct_version(ver: u32) -> u32 {
    NVENCAPI_VERSION | (ver << 16) | (0x7 << 28)
}

// ── NVENC parameter structs (minimal FFI definitions) ─────────────────

/// NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS
#[repr(C)]
pub struct NvEncOpenEncodeSessionExParams {
    pub version: u32,
    pub device_type: u32,       // NV_ENC_DEVICE_TYPE (1 = CUDA)
    pub device: *mut c_void,    // CUcontext for CUDA device type
    pub reserved: *mut c_void,
    pub api_version: u32,       // NVENCAPI_VERSION
    pub reserved1: [u32; 253],
    pub reserved2: [*mut c_void; 64],
}

/// NV_ENC_CAPS_PARAM
#[repr(C)]
pub struct NvEncCapsParam {
    pub version: u32,
    pub caps_to_query: u32,     // NV_ENC_CAPS enum value
    pub reserved: [u32; 62],
}

/// NV_ENC_CREATE_INPUT_BUFFER
#[repr(C)]
pub struct NvEncCreateInputBuffer {
    pub version: u32,
    pub width: u32,
    pub height: u32,
    pub memory_heap: u32,       // Deprecated, set to 0
    pub buffer_fmt: u32,        // NV_ENC_BUFFER_FORMAT
    pub reserved: u32,
    pub input_buffer: *mut c_void, // OUT: input buffer pointer
    pub p_sys_mem: *mut c_void,
    pub reserved1: [u32; 57],
    pub reserved2: [*mut c_void; 63],
}

/// NV_ENC_CREATE_BITSTREAM_BUFFER
#[repr(C)]
pub struct NvEncCreateBitstreamBuffer {
    pub version: u32,
    pub reserved: u32,
    pub bitstream_buffer: *mut c_void, // OUT: bitstream buffer pointer
    pub reserved_mem_buf: *mut c_void,
    pub reserved_size: u32,
    pub reserved1: [u32; 57],
    pub reserved2: [*mut c_void; 63],
}

/// NV_ENC_LOCK_INPUT_BUFFER
#[repr(C)]
pub struct NvEncLockInputBuffer {
    pub version: u32,
    pub reserved_flags: u32,
    pub input_buffer: *mut c_void,
    pub buffer_data_ptr: *mut c_void,   // OUT: pointer to locked buffer
    pub pitch: u32,                     // OUT: pitch of locked buffer
    pub reserved1: [u32; 251],
    pub reserved2: [*mut c_void; 64],
}

/// NV_ENC_LOCK_BITSTREAM
#[repr(C)]
pub struct NvEncLockBitstream {
    pub version: u32,
    pub do_not_wait: u32,               // Bit 0: doNotWait
    pub output_bitstream: *mut c_void,
    pub slice_offsets: *mut u32,
    pub frame_idx: u32,
    pub hw_encode_status: u32,
    pub num_slices: u32,
    pub bitstream_size_in_bytes: u32,
    pub output_timestamp: u64,
    pub output_duration: u64,
    pub bitstream_buffer_ptr: *mut c_void, // OUT: pointer to bitstream data
    pub picture_type: u32,               // NV_ENC_PIC_TYPE
    pub picture_struct: u32,
    pub frame_avg_qp: u32,
    pub frame_satd: u32,
    pub ltr_frame_idx: u32,
    pub ltr_frame_bitmap: u32,
    pub reserved: [u32; 13],
    pub intra_refresh_cnt: u32,
    pub reserved1: [u32; 219],
    pub reserved2: [*mut c_void; 64],
}

/// NV_ENC_REGISTER_RESOURCE
#[repr(C)]
pub struct NvEncRegisterResource {
    pub version: u32,
    pub resource_type: u32,     // NV_ENC_INPUT_RESOURCE_TYPE
    pub width: u32,
    pub height: u32,
    pub pitch: u32,
    pub sub_resource_index: u32,
    pub resource_to_register: *mut c_void,
    pub registered_resource: *mut c_void, // OUT: registered resource handle
    pub buffer_format: u32,
    pub buffer_usage: u32,
    pub reserved1: [u32; 247],
    pub reserved2: [*mut c_void; 62],
}

/// NV_ENC_MAP_INPUT_RESOURCE
#[repr(C)]
pub struct NvEncMapInputResource {
    pub version: u32,
    pub sub_resource_index: u32,
    pub input_resource: *mut c_void,     // IN: from RegisterResource
    pub mapped_resource: *mut c_void,    // OUT: mapped resource handle
    pub mapped_buffer_fmt: u32,          // OUT
    pub reserved1: [u32; 255],
    pub reserved2: [*mut c_void; 63],
}

/// NV_ENC_SEQUENCE_PARAM_PAYLOAD
#[repr(C)]
pub struct NvEncSequenceParamPayload {
    pub version: u32,
    pub in_buffer_size: u32,
    pub sps_id: u32,
    pub pps_id: u32,
    pub sp_spps_buffer: *mut c_void,
    pub out_sps_pps_payload_size: *mut u32,
    pub reserved1: [u32; 250],
    pub reserved2: [*mut c_void; 64],
}

// ── Function table ────────────────────────────────────────────────────

/// The NVENC function table, populated by NvEncodeAPICreateInstance.
/// This matches the NV_ENCODE_API_FUNCTION_LIST layout from nvEncodeAPI.h.
#[repr(C)]
pub struct NvEncFunctionList {
    pub version: u32,
    pub reserved: u32,
    // Function pointers filled by NvEncodeAPICreateInstance
    pub nv_enc_open_encode_session: Option<unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> NvencStatus>,
    pub nv_enc_get_encode_guid_count: Option<unsafe extern "C" fn(*mut c_void, *mut u32) -> NvencStatus>,
    pub nv_enc_get_encode_guids: Option<unsafe extern "C" fn(*mut c_void, *mut [u8; 16], u32, *mut u32) -> NvencStatus>,
    pub nv_enc_get_encode_profile_guid_count: Option<unsafe extern "C" fn(*mut c_void, [u8; 16], *mut u32) -> NvencStatus>,
    pub nv_enc_get_encode_profile_guids: Option<unsafe extern "C" fn(*mut c_void, [u8; 16], *mut [u8; 16], u32, *mut u32) -> NvencStatus>,
    pub nv_enc_get_input_format_count: Option<unsafe extern "C" fn(*mut c_void, [u8; 16], *mut u32) -> NvencStatus>,
    pub nv_enc_get_input_formats: Option<unsafe extern "C" fn(*mut c_void, [u8; 16], *mut u32, u32, *mut u32) -> NvencStatus>,
    pub nv_enc_get_encode_caps: Option<unsafe extern "C" fn(*mut c_void, [u8; 16], *mut NvEncCapsParam, *mut c_int) -> NvencStatus>,
    pub nv_enc_get_encode_preset_count: Option<unsafe extern "C" fn(*mut c_void, [u8; 16], *mut u32) -> NvencStatus>,
    pub nv_enc_get_encode_preset_guids: Option<unsafe extern "C" fn(*mut c_void, [u8; 16], *mut [u8; 16], u32, *mut u32) -> NvencStatus>,
    pub nv_enc_get_encode_preset_config: Option<unsafe extern "C" fn(*mut c_void, [u8; 16], [u8; 16], *mut c_void) -> NvencStatus>,
    pub nv_enc_initialize_encoder: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NvencStatus>,
    pub nv_enc_create_input_buffer: Option<unsafe extern "C" fn(*mut c_void, *mut NvEncCreateInputBuffer) -> NvencStatus>,
    pub nv_enc_destroy_input_buffer: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NvencStatus>,
    pub nv_enc_create_bitstream_buffer: Option<unsafe extern "C" fn(*mut c_void, *mut NvEncCreateBitstreamBuffer) -> NvencStatus>,
    pub nv_enc_destroy_bitstream_buffer: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NvencStatus>,
    pub nv_enc_encode_picture: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NvencStatus>,
    pub nv_enc_lock_bitstream: Option<unsafe extern "C" fn(*mut c_void, *mut NvEncLockBitstream) -> NvencStatus>,
    pub nv_enc_unlock_bitstream: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NvencStatus>,
    pub nv_enc_lock_input_buffer: Option<unsafe extern "C" fn(*mut c_void, *mut NvEncLockInputBuffer) -> NvencStatus>,
    pub nv_enc_unlock_input_buffer: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NvencStatus>,
    pub nv_enc_get_encode_stats: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NvencStatus>,
    pub nv_enc_get_sequence_params: Option<unsafe extern "C" fn(*mut c_void, *mut NvEncSequenceParamPayload) -> NvencStatus>,
    pub nv_enc_register_async_event: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NvencStatus>,
    pub nv_enc_unregister_async_event: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NvencStatus>,
    pub nv_enc_map_input_resource: Option<unsafe extern "C" fn(*mut c_void, *mut NvEncMapInputResource) -> NvencStatus>,
    pub nv_enc_unmap_input_resource: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NvencStatus>,
    pub nv_enc_destroy_encoder: Option<unsafe extern "C" fn(*mut c_void) -> NvencStatus>,
    pub nv_enc_invalidate_ref_frames: Option<unsafe extern "C" fn(*mut c_void, u64) -> NvencStatus>,
    pub nv_enc_open_encode_session_ex: Option<unsafe extern "C" fn(*mut NvEncOpenEncodeSessionExParams, *mut *mut c_void) -> NvencStatus>,
    pub nv_enc_register_resource: Option<unsafe extern "C" fn(*mut c_void, *mut NvEncRegisterResource) -> NvencStatus>,
    pub nv_enc_unregister_resource: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NvencStatus>,
    pub nv_enc_reconfigure_encoder: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NvencStatus>,
    // Reserved function pointers (the struct may have more in newer API versions)
    pub reserved1: *mut c_void,
    pub reserved2: [*mut c_void; 277],
}

/// Type for the NvEncodeAPICreateInstance entry point.
type FnNvEncodeAPICreateInstance =
    unsafe extern "C" fn(function_list: *mut NvEncFunctionList) -> NvencStatus;

/// Type for the NvEncodeAPIGetMaxSupportedVersion entry point.
type FnNvEncodeAPIGetMaxSupportedVersion =
    unsafe extern "C" fn(version: *mut u32) -> NvencStatus;

/// Dynamically loaded NVENC library with the populated function table.
pub struct NvencDriver {
    _lib: Library,
    func_list: NvEncFunctionList,
    max_supported_version: u32,
}

// SAFETY: The NVENC library handles are valid across threads when used with
// proper encoder handle management. Each encoder session is bound to a CUDA context.
unsafe impl Send for NvencDriver {}
unsafe impl Sync for NvencDriver {}

impl NvencDriver {
    /// Load the NVENC library and populate the function table.
    pub fn load() -> Result<Arc<Self>, String> {
        let lib = Self::load_library()?;

        unsafe {
            // Get maximum supported version
            let get_max_version: Symbol<FnNvEncodeAPIGetMaxSupportedVersion> = lib
                .get(b"NvEncodeAPIGetMaxSupportedVersion")
                .map_err(|e| format!("NvEncodeAPIGetMaxSupportedVersion not found: {}", e))?;

            let mut max_version: u32 = 0;
            let res = get_max_version(&mut max_version);
            if res != NV_ENC_SUCCESS {
                return Err(format!(
                    "NvEncodeAPIGetMaxSupportedVersion failed: {} ({})",
                    nvenc_error_name(res),
                    res
                ));
            }
            info!("NVENC max supported version: {}.{}", max_version >> 4, max_version & 0xF);

            // Check that our target version is supported
            // GetMaxSupportedVersion returns (major << 4) | minor format
            let our_max_format: u32 = (12 << 4) | 2; // 12.2
            if our_max_format > max_version {
                warn!(
                    "NVENC API version 12.2 requested but max supported is {}.{}, proceeding anyway",
                    max_version >> 4, max_version & 0xF
                );
            }

            // Create and populate the function list
            let create_instance: Symbol<FnNvEncodeAPICreateInstance> = lib
                .get(b"NvEncodeAPICreateInstance")
                .map_err(|e| format!("NvEncodeAPICreateInstance not found: {}", e))?;

            let mut func_list: NvEncFunctionList = std::mem::zeroed();
            func_list.version = nvenc_struct_version(2); // NV_ENCODE_API_FUNCTION_LIST_VER

            let res = create_instance(&mut func_list);
            if res != NV_ENC_SUCCESS {
                return Err(format!(
                    "NvEncodeAPICreateInstance failed: {} ({})",
                    nvenc_error_name(res),
                    res
                ));
            }

            info!("NVENC function table populated successfully");

            Ok(Arc::new(Self {
                _lib: lib,
                func_list,
                max_supported_version: max_version,
            }))
        }
    }

    fn load_library() -> Result<Library, String> {
        // On Windows, try the _real.dll first to avoid loading our own interpose DLL
        // when System32\nvEncodeAPI64.dll has been replaced by the RGPU installer.
        #[cfg(target_os = "windows")]
        let lib_names = &["nvEncodeAPI64_real.dll", "nvEncodeAPI64.dll", "nvEncodeAPI.dll"];

        #[cfg(target_os = "linux")]
        let lib_names = &["libnvidia-encode.so.1", "libnvidia-encode.so"];

        #[cfg(target_os = "macos")]
        let lib_names: &[&str] = &[];

        let mut last_err = String::new();
        for name in lib_names {
            match unsafe { Library::new(name) } {
                Ok(lib) => {
                    info!("loaded NVENC library from: {}", name);
                    return Ok(lib);
                }
                Err(e) => {
                    last_err = format!("{}: {}", name, e);
                    debug!("failed to load {}: {}", name, e);
                }
            }
        }

        Err(format!("failed to load NVENC library: {}", last_err))
    }

    // ── Version query ────────────────────────────────────────────

    /// Get the maximum supported NVENC API version.
    pub fn get_max_supported_version(&self) -> u32 {
        self.max_supported_version
    }

    // ── Session management ───────────────────────────────────────

    /// Open an encode session using a CUDA context.
    pub fn open_encode_session_ex(
        &self,
        cuda_context: *mut c_void,
        device_type: u32,
    ) -> Result<*mut c_void, NvencStatus> {
        let func = self
            .func_list
            .nv_enc_open_encode_session_ex
            .ok_or(NV_ENC_ERR_UNIMPLEMENTED)?;

        let mut params: NvEncOpenEncodeSessionExParams = unsafe { std::mem::zeroed() };
        params.version = nvenc_struct_version(1);
        params.device_type = device_type;
        params.device = cuda_context;
        params.api_version = NVENCAPI_VERSION;

        let mut encoder: *mut c_void = std::ptr::null_mut();
        let res = unsafe { func(&mut params, &mut encoder) };
        if res == NV_ENC_SUCCESS {
            Ok(encoder)
        } else {
            Err(res)
        }
    }

    /// Destroy an encoder session.
    pub fn destroy_encoder(&self, encoder: *mut c_void) -> NvencStatus {
        if let Some(func) = self.func_list.nv_enc_destroy_encoder {
            unsafe { func(encoder) }
        } else {
            NV_ENC_ERR_UNIMPLEMENTED
        }
    }

    // ── Capability queries ───────────────────────────────────────

    /// Get count of supported encode GUIDs.
    pub fn get_encode_guid_count(&self, encoder: *mut c_void) -> Result<u32, NvencStatus> {
        let func = self
            .func_list
            .nv_enc_get_encode_guid_count
            .ok_or(NV_ENC_ERR_UNIMPLEMENTED)?;

        let mut count: u32 = 0;
        let res = unsafe { func(encoder, &mut count) };
        if res == NV_ENC_SUCCESS {
            Ok(count)
        } else {
            Err(res)
        }
    }

    /// Get list of supported encode GUIDs.
    pub fn get_encode_guids(
        &self,
        encoder: *mut c_void,
        max_count: u32,
    ) -> Result<Vec<[u8; 16]>, NvencStatus> {
        let func = self
            .func_list
            .nv_enc_get_encode_guids
            .ok_or(NV_ENC_ERR_UNIMPLEMENTED)?;

        let mut guids: Vec<[u8; 16]> = vec![[0u8; 16]; max_count as usize];
        let mut actual_count: u32 = 0;
        let res = unsafe { func(encoder, guids.as_mut_ptr(), max_count, &mut actual_count) };
        if res == NV_ENC_SUCCESS {
            guids.truncate(actual_count as usize);
            Ok(guids)
        } else {
            Err(res)
        }
    }

    /// Get count of supported presets for a codec.
    pub fn get_encode_preset_count(
        &self,
        encoder: *mut c_void,
        encode_guid: [u8; 16],
    ) -> Result<u32, NvencStatus> {
        let func = self
            .func_list
            .nv_enc_get_encode_preset_count
            .ok_or(NV_ENC_ERR_UNIMPLEMENTED)?;

        let mut count: u32 = 0;
        let res = unsafe { func(encoder, encode_guid, &mut count) };
        if res == NV_ENC_SUCCESS {
            Ok(count)
        } else {
            Err(res)
        }
    }

    /// Get list of supported preset GUIDs for a codec.
    pub fn get_encode_preset_guids(
        &self,
        encoder: *mut c_void,
        encode_guid: [u8; 16],
        max_count: u32,
    ) -> Result<Vec<[u8; 16]>, NvencStatus> {
        let func = self
            .func_list
            .nv_enc_get_encode_preset_guids
            .ok_or(NV_ENC_ERR_UNIMPLEMENTED)?;

        let mut presets: Vec<[u8; 16]> = vec![[0u8; 16]; max_count as usize];
        let mut actual_count: u32 = 0;
        let res =
            unsafe { func(encoder, encode_guid, presets.as_mut_ptr(), max_count, &mut actual_count) };
        if res == NV_ENC_SUCCESS {
            presets.truncate(actual_count as usize);
            Ok(presets)
        } else {
            Err(res)
        }
    }

    /// Get preset configuration for a codec/preset combination.
    /// Returns the raw preset config bytes.
    pub fn get_encode_preset_config(
        &self,
        encoder: *mut c_void,
        encode_guid: [u8; 16],
        preset_guid: [u8; 16],
    ) -> Result<Vec<u8>, NvencStatus> {
        let func = self
            .func_list
            .nv_enc_get_encode_preset_config
            .ok_or(NV_ENC_ERR_UNIMPLEMENTED)?;

        // Allocate a buffer for NV_ENC_PRESET_CONFIG (large enough for any version).
        // The struct is versioned and can vary in size, so we use a generous buffer.
        const PRESET_CONFIG_BUF_SIZE: usize = 8192;
        let mut config_buf = vec![0u8; PRESET_CONFIG_BUF_SIZE];
        // Set version at the start of the buffer
        // NV_ENC_PRESET_CONFIG_VER = NVENCAPI_STRUCT_VERSION(4) | (1<<31)
        let version = nvenc_struct_version(4) | (1 << 31);
        config_buf[0..4].copy_from_slice(&version.to_le_bytes());

        let res = unsafe {
            func(
                encoder,
                encode_guid,
                preset_guid,
                config_buf.as_mut_ptr() as *mut c_void,
            )
        };
        if res == NV_ENC_SUCCESS {
            Ok(config_buf)
        } else {
            Err(res)
        }
    }

    /// Query encoder capabilities.
    pub fn get_encode_caps(
        &self,
        encoder: *mut c_void,
        encode_guid: [u8; 16],
        caps_to_query: u32,
    ) -> Result<i32, NvencStatus> {
        let func = self
            .func_list
            .nv_enc_get_encode_caps
            .ok_or(NV_ENC_ERR_UNIMPLEMENTED)?;

        let mut caps_param: NvEncCapsParam = unsafe { std::mem::zeroed() };
        caps_param.version = nvenc_struct_version(1);
        caps_param.caps_to_query = caps_to_query;

        let mut caps_val: c_int = 0;
        let res = unsafe { func(encoder, encode_guid, &mut caps_param, &mut caps_val) };
        if res == NV_ENC_SUCCESS {
            Ok(caps_val)
        } else {
            Err(res)
        }
    }

    /// Get count of supported input formats for a codec.
    pub fn get_input_format_count(
        &self,
        encoder: *mut c_void,
        encode_guid: [u8; 16],
    ) -> Result<u32, NvencStatus> {
        let func = self
            .func_list
            .nv_enc_get_input_format_count
            .ok_or(NV_ENC_ERR_UNIMPLEMENTED)?;

        let mut count: u32 = 0;
        let res = unsafe { func(encoder, encode_guid, &mut count) };
        if res == NV_ENC_SUCCESS {
            Ok(count)
        } else {
            Err(res)
        }
    }

    /// Get list of supported input formats for a codec.
    pub fn get_input_formats(
        &self,
        encoder: *mut c_void,
        encode_guid: [u8; 16],
        max_count: u32,
    ) -> Result<Vec<u32>, NvencStatus> {
        let func = self
            .func_list
            .nv_enc_get_input_formats
            .ok_or(NV_ENC_ERR_UNIMPLEMENTED)?;

        let mut formats: Vec<u32> = vec![0u32; max_count as usize];
        let mut actual_count: u32 = 0;
        let res =
            unsafe { func(encoder, encode_guid, formats.as_mut_ptr(), max_count, &mut actual_count) };
        if res == NV_ENC_SUCCESS {
            formats.truncate(actual_count as usize);
            Ok(formats)
        } else {
            Err(res)
        }
    }

    // ── Encoder initialization ───────────────────────────────────

    /// Initialize the encoder with raw parameter bytes.
    /// The caller is responsible for providing a properly versioned NV_ENC_INITIALIZE_PARAMS struct.
    pub fn initialize_encoder(
        &self,
        encoder: *mut c_void,
        params: &mut [u8],
    ) -> NvencStatus {
        if let Some(func) = self.func_list.nv_enc_initialize_encoder {
            unsafe { func(encoder, params.as_mut_ptr() as *mut c_void) }
        } else {
            NV_ENC_ERR_UNIMPLEMENTED
        }
    }

    /// Reconfigure encoder dynamically with raw parameter bytes.
    pub fn reconfigure_encoder(
        &self,
        encoder: *mut c_void,
        params: &mut [u8],
    ) -> NvencStatus {
        if let Some(func) = self.func_list.nv_enc_reconfigure_encoder {
            unsafe { func(encoder, params.as_mut_ptr() as *mut c_void) }
        } else {
            NV_ENC_ERR_UNIMPLEMENTED
        }
    }

    // ── Input buffer management ──────────────────────────────────

    /// Create an input buffer.
    pub fn create_input_buffer(
        &self,
        encoder: *mut c_void,
        width: u32,
        height: u32,
        buffer_fmt: u32,
    ) -> Result<*mut c_void, NvencStatus> {
        let func = self
            .func_list
            .nv_enc_create_input_buffer
            .ok_or(NV_ENC_ERR_UNIMPLEMENTED)?;

        let mut params: NvEncCreateInputBuffer = unsafe { std::mem::zeroed() };
        params.version = nvenc_struct_version(1);
        params.width = width;
        params.height = height;
        params.buffer_fmt = buffer_fmt;

        let res = unsafe { func(encoder, &mut params) };
        if res == NV_ENC_SUCCESS {
            Ok(params.input_buffer)
        } else {
            Err(res)
        }
    }

    /// Destroy an input buffer.
    pub fn destroy_input_buffer(
        &self,
        encoder: *mut c_void,
        buffer: *mut c_void,
    ) -> NvencStatus {
        if let Some(func) = self.func_list.nv_enc_destroy_input_buffer {
            unsafe { func(encoder, buffer) }
        } else {
            NV_ENC_ERR_UNIMPLEMENTED
        }
    }

    /// Lock an input buffer for CPU access.
    pub fn lock_input_buffer(
        &self,
        encoder: *mut c_void,
        buffer: *mut c_void,
    ) -> Result<(*mut c_void, u32), NvencStatus> {
        let func = self
            .func_list
            .nv_enc_lock_input_buffer
            .ok_or(NV_ENC_ERR_UNIMPLEMENTED)?;

        let mut params: NvEncLockInputBuffer = unsafe { std::mem::zeroed() };
        params.version = nvenc_struct_version(1);
        params.input_buffer = buffer;

        let res = unsafe { func(encoder, &mut params) };
        if res == NV_ENC_SUCCESS {
            Ok((params.buffer_data_ptr, params.pitch))
        } else {
            Err(res)
        }
    }

    /// Unlock an input buffer.
    pub fn unlock_input_buffer(
        &self,
        encoder: *mut c_void,
        buffer: *mut c_void,
    ) -> NvencStatus {
        if let Some(func) = self.func_list.nv_enc_unlock_input_buffer {
            unsafe { func(encoder, buffer) }
        } else {
            NV_ENC_ERR_UNIMPLEMENTED
        }
    }

    // ── Bitstream buffer management ──────────────────────────────

    /// Create a bitstream output buffer.
    pub fn create_bitstream_buffer(
        &self,
        encoder: *mut c_void,
    ) -> Result<*mut c_void, NvencStatus> {
        let func = self
            .func_list
            .nv_enc_create_bitstream_buffer
            .ok_or(NV_ENC_ERR_UNIMPLEMENTED)?;

        let mut params: NvEncCreateBitstreamBuffer = unsafe { std::mem::zeroed() };
        params.version = nvenc_struct_version(1);

        let res = unsafe { func(encoder, &mut params) };
        if res == NV_ENC_SUCCESS {
            Ok(params.bitstream_buffer)
        } else {
            Err(res)
        }
    }

    /// Destroy a bitstream buffer.
    pub fn destroy_bitstream_buffer(
        &self,
        encoder: *mut c_void,
        buffer: *mut c_void,
    ) -> NvencStatus {
        if let Some(func) = self.func_list.nv_enc_destroy_bitstream_buffer {
            unsafe { func(encoder, buffer) }
        } else {
            NV_ENC_ERR_UNIMPLEMENTED
        }
    }

    /// Lock a bitstream buffer to read encoded data.
    pub fn lock_bitstream(
        &self,
        encoder: *mut c_void,
        bitstream_buffer: *mut c_void,
    ) -> Result<NvEncLockBitstreamResult, NvencStatus> {
        let func = self
            .func_list
            .nv_enc_lock_bitstream
            .ok_or(NV_ENC_ERR_UNIMPLEMENTED)?;

        let mut params: NvEncLockBitstream = unsafe { std::mem::zeroed() };
        params.version = nvenc_struct_version(1);
        params.output_bitstream = bitstream_buffer;

        let res = unsafe { func(encoder, &mut params) };
        if res == NV_ENC_SUCCESS {
            // Copy the bitstream data
            let data = if !params.bitstream_buffer_ptr.is_null()
                && params.bitstream_size_in_bytes > 0
            {
                unsafe {
                    std::slice::from_raw_parts(
                        params.bitstream_buffer_ptr as *const u8,
                        params.bitstream_size_in_bytes as usize,
                    )
                }
                .to_vec()
            } else {
                Vec::new()
            };

            Ok(NvEncLockBitstreamResult {
                data,
                picture_type: params.picture_type,
                frame_idx: params.frame_idx,
                output_timestamp: params.output_timestamp,
            })
        } else {
            Err(res)
        }
    }

    /// Unlock a bitstream buffer.
    pub fn unlock_bitstream(
        &self,
        encoder: *mut c_void,
        bitstream_buffer: *mut c_void,
    ) -> NvencStatus {
        if let Some(func) = self.func_list.nv_enc_unlock_bitstream {
            unsafe { func(encoder, bitstream_buffer) }
        } else {
            NV_ENC_ERR_UNIMPLEMENTED
        }
    }

    // ── Resource registration ────────────────────────────────────

    /// Register an external resource (CUDA device pointer, etc.) for use as encoder input.
    pub fn register_resource(
        &self,
        encoder: *mut c_void,
        resource_type: u32,
        resource: *mut c_void,
        width: u32,
        height: u32,
        pitch: u32,
        buffer_format: u32,
    ) -> Result<*mut c_void, NvencStatus> {
        let func = self
            .func_list
            .nv_enc_register_resource
            .ok_or(NV_ENC_ERR_UNIMPLEMENTED)?;

        let mut params: NvEncRegisterResource = unsafe { std::mem::zeroed() };
        params.version = nvenc_struct_version(4); // NV_ENC_REGISTER_RESOURCE_VER
        params.resource_type = resource_type;
        params.resource_to_register = resource;
        params.width = width;
        params.height = height;
        params.pitch = pitch;
        params.buffer_format = buffer_format;

        let res = unsafe { func(encoder, &mut params) };
        if res == NV_ENC_SUCCESS {
            Ok(params.registered_resource)
        } else {
            Err(res)
        }
    }

    /// Unregister a previously registered resource.
    pub fn unregister_resource(
        &self,
        encoder: *mut c_void,
        resource: *mut c_void,
    ) -> NvencStatus {
        if let Some(func) = self.func_list.nv_enc_unregister_resource {
            unsafe { func(encoder, resource) }
        } else {
            NV_ENC_ERR_UNIMPLEMENTED
        }
    }

    /// Map a registered resource for encoding.
    pub fn map_input_resource(
        &self,
        encoder: *mut c_void,
        registered_resource: *mut c_void,
    ) -> Result<(*mut c_void, u32), NvencStatus> {
        let func = self
            .func_list
            .nv_enc_map_input_resource
            .ok_or(NV_ENC_ERR_UNIMPLEMENTED)?;

        let mut params: NvEncMapInputResource = unsafe { std::mem::zeroed() };
        params.version = nvenc_struct_version(4); // NV_ENC_MAP_INPUT_RESOURCE_VER
        params.input_resource = registered_resource;

        let res = unsafe { func(encoder, &mut params) };
        if res == NV_ENC_SUCCESS {
            Ok((params.mapped_resource, params.mapped_buffer_fmt))
        } else {
            Err(res)
        }
    }

    /// Unmap a previously mapped resource.
    pub fn unmap_input_resource(
        &self,
        encoder: *mut c_void,
        mapped_resource: *mut c_void,
    ) -> NvencStatus {
        if let Some(func) = self.func_list.nv_enc_unmap_input_resource {
            unsafe { func(encoder, mapped_resource) }
        } else {
            NV_ENC_ERR_UNIMPLEMENTED
        }
    }

    // ── Encoding ─────────────────────────────────────────────────

    /// Encode a picture using raw parameter bytes.
    /// The caller must provide a properly versioned NV_ENC_PIC_PARAMS struct.
    pub fn encode_picture(
        &self,
        encoder: *mut c_void,
        params: &mut [u8],
    ) -> NvencStatus {
        if let Some(func) = self.func_list.nv_enc_encode_picture {
            unsafe { func(encoder, params.as_mut_ptr() as *mut c_void) }
        } else {
            NV_ENC_ERR_UNIMPLEMENTED
        }
    }

    // ── Parameter retrieval ──────────────────────────────────────

    /// Get SPS/PPS sequence parameter data.
    pub fn get_sequence_params(
        &self,
        encoder: *mut c_void,
    ) -> Result<Vec<u8>, NvencStatus> {
        let func = self
            .func_list
            .nv_enc_get_sequence_params
            .ok_or(NV_ENC_ERR_UNIMPLEMENTED)?;

        let mut payload: NvEncSequenceParamPayload = unsafe { std::mem::zeroed() };
        payload.version = nvenc_struct_version(1);

        // Allocate buffer for SPS/PPS data
        const MAX_SPS_PPS_SIZE: usize = 1024;
        let mut sps_pps_buf = vec![0u8; MAX_SPS_PPS_SIZE];
        let mut out_size: u32 = 0;

        payload.in_buffer_size = MAX_SPS_PPS_SIZE as u32;
        payload.sp_spps_buffer = sps_pps_buf.as_mut_ptr() as *mut c_void;
        payload.out_sps_pps_payload_size = &mut out_size;

        let res = unsafe { func(encoder, &mut payload) };
        if res == NV_ENC_SUCCESS {
            sps_pps_buf.truncate(out_size as usize);
            Ok(sps_pps_buf)
        } else {
            Err(res)
        }
    }

    /// Get encoding statistics (returned as raw bytes).
    pub fn get_encode_stats(
        &self,
        encoder: *mut c_void,
    ) -> Result<Vec<u8>, NvencStatus> {
        let func = self
            .func_list
            .nv_enc_get_encode_stats
            .ok_or(NV_ENC_ERR_UNIMPLEMENTED)?;

        // NV_ENC_STAT has a versioned header. Allocate generously.
        const STATS_BUF_SIZE: usize = 4096;
        let mut stats_buf = vec![0u8; STATS_BUF_SIZE];
        let version = nvenc_struct_version(1);
        stats_buf[0..4].copy_from_slice(&version.to_le_bytes());

        let res = unsafe { func(encoder, stats_buf.as_mut_ptr() as *mut c_void) };
        if res == NV_ENC_SUCCESS {
            Ok(stats_buf)
        } else {
            Err(res)
        }
    }

    /// Invalidate reference frames for error recovery.
    pub fn invalidate_ref_frames(
        &self,
        encoder: *mut c_void,
        timestamp: u64,
    ) -> NvencStatus {
        if let Some(func) = self.func_list.nv_enc_invalidate_ref_frames {
            unsafe { func(encoder, timestamp) }
        } else {
            NV_ENC_ERR_UNIMPLEMENTED
        }
    }
}

/// Result of locking a bitstream buffer (decoded data + metadata).
pub struct NvEncLockBitstreamResult {
    pub data: Vec<u8>,
    pub picture_type: u32,
    pub frame_idx: u32,
    pub output_timestamp: u64,
}

/// Convert an NVENCSTATUS error code to a human-readable string.
pub fn nvenc_error_name(status: NvencStatus) -> &'static str {
    match status {
        0 => "NV_ENC_SUCCESS",
        1 => "NV_ENC_ERR_NO_ENCODE_DEVICE",
        2 => "NV_ENC_ERR_UNSUPPORTED_DEVICE",
        3 => "NV_ENC_ERR_INVALID_ENCODERDEVICE",
        4 => "NV_ENC_ERR_INVALID_DEVICE",
        5 => "NV_ENC_ERR_DEVICE_NOT_EXIST",
        6 => "NV_ENC_ERR_INVALID_PTR",
        7 => "NV_ENC_ERR_INVALID_EVENT",
        8 => "NV_ENC_ERR_INVALID_PARAM",
        9 => "NV_ENC_ERR_INVALID_CALL",
        10 => "NV_ENC_ERR_OUT_OF_MEMORY",
        11 => "NV_ENC_ERR_ENCODER_NOT_INITIALIZED",
        12 => "NV_ENC_ERR_UNSUPPORTED_PARAM",
        13 => "NV_ENC_ERR_LOCK_BUSY",
        14 => "NV_ENC_ERR_NOT_ENOUGH_BUFFER",
        15 => "NV_ENC_ERR_INVALID_VERSION",
        16 => "NV_ENC_ERR_MAP_FAILED",
        17 => "NV_ENC_ERR_NEED_MORE_INPUT",
        18 => "NV_ENC_ERR_ENCODER_BUSY",
        19 => "NV_ENC_ERR_EVENT_NOT_REGISTERD",
        20 => "NV_ENC_ERR_GENERIC",
        21 => "NV_ENC_ERR_INCOMPATIBLE_CLIENT_KEY",
        22 => "NV_ENC_ERR_UNIMPLEMENTED",
        23 => "NV_ENC_ERR_RESOURCE_REGISTER_FAILED",
        24 => "NV_ENC_ERR_RESOURCE_NOT_REGISTERED",
        25 => "NV_ENC_ERR_RESOURCE_NOT_MAPPED",
        _ => "NV_ENC_ERR_UNKNOWN",
    }
}
