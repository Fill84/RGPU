//! NVDEC (CUVID) Video Decoder API interception library.
//!
//! This cdylib replaces the standard CUVID library (libnvcuvid.so / nvcuvid.dll).
//! It intercepts CUVID API calls and forwards them to the RGPU client daemon
//! via IPC, which in turn sends them to a remote RGPU server with a real GPU.
//!
//! Usage:
//! - Linux: LD_PRELOAD=librgpu_nvdec_interpose.so <application>
//! - Windows: Place as nvcuvid.dll in the application's directory or System32

mod ipc_client;
pub mod handle_store;

use std::ffi::{c_int, c_uint, c_void};
use std::sync::OnceLock;

use tracing::{debug, error};

use rgpu_protocol::handle::{NetworkHandle, ResourceType};
use rgpu_protocol::nvdec_commands::{NvdecCommand, NvdecResponse};

use ipc_client::NvdecIpcClient;

// ── Cross-DLL CUDA handle resolution ───────────────────────────────────

type CudaResolveFn = unsafe extern "C" fn(u64, *mut u16, *mut u32, *mut u64) -> c_int;

static CUDA_RESOLVE_CTX: OnceLock<Option<CudaResolveFn>> = OnceLock::new();

fn load_cuda_resolve_fn(name: &[u8]) -> Option<CudaResolveFn> {
    unsafe {
        #[cfg(target_os = "windows")]
        let lib_name = "nvcuda.dll";
        #[cfg(not(target_os = "windows"))]
        let lib_name = "libcuda.so.1";

        let lib = libloading::Library::new(lib_name).ok()?;
        let sym: libloading::Symbol<CudaResolveFn> = lib.get(name).ok()?;
        let func = *sym;
        std::mem::forget(lib);
        Some(func)
    }
}

fn resolve_cuda_ctx_handle(local_id: u64) -> Option<NetworkHandle> {
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

// ── FFI types ───────────────────────────────────────────────────────────────

type CUresult = c_int;
type CUcontext = *mut c_void;
type CUvideodecoder = *mut c_void;
type CUvideoparser = *mut c_void;
type CUvideoctxlock = *mut c_void;

const CUDA_SUCCESS: CUresult = 0;
const CUDA_ERROR_INVALID_VALUE: CUresult = 1;
const CUDA_ERROR_NOT_SUPPORTED: CUresult = 801;
const CUDA_ERROR_UNKNOWN: CUresult = 999;

// ── IPC client singleton ────────────────────────────────────────────────────

static IPC_CLIENT: OnceLock<NvdecIpcClient> = OnceLock::new();

fn get_client() -> &'static NvdecIpcClient {
    IPC_CLIENT.get_or_init(|| {
        let path = rgpu_common::platform::resolve_ipc_address();
        NvdecIpcClient::new(&path)
    })
}

fn send_nvdec_command(cmd: NvdecCommand) -> NvdecResponse {
    let client = get_client();
    match client.send_command(cmd) {
        Ok(resp) => resp,
        Err(e) => {
            error!("NVDEC IPC error: {}", e);
            NvdecResponse::Error {
                code: CUDA_ERROR_UNKNOWN,
                message: e.to_string(),
            }
        }
    }
}

/// Convert a NvdecResponse::Error to a CUresult code.
fn response_error_code(resp: &NvdecResponse) -> CUresult {
    match resp {
        NvdecResponse::Error { code, .. } => *code,
        _ => CUDA_SUCCESS,
    }
}

// ── Detection marker ────────────────────────────────────────────────────────

/// Marker function for RGPU interpose DLL detection.
/// Used by the installer/daemon to verify that nvcuvid.dll is the RGPU interpose
/// library and not the real NVIDIA CUVID driver.
#[no_mangle]
pub extern "C" fn rgpu_interpose_marker() -> c_int {
    1
}

// ═══════════════════════════════════════════════════════════════════════════
// CAPABILITY QUERY
// ═══════════════════════════════════════════════════════════════════════════

/// Query decoder capabilities for a given codec/chroma/bitdepth combination.
///
/// CUVIDDECODECAPS struct layout (64-bit):
/// - Offset 0: eCodecType (u32) - INPUT
/// - Offset 4: eChromaFormat (u32) - INPUT
/// - Offset 8: nBitDepthMinus8 (u32) - INPUT
/// - Offset 24: bIsSupported (u32) - OUTPUT
/// - Offset 28: nNumNVDECs (u32) - OUTPUT
/// - Offset 36: nMaxWidth (u32) - OUTPUT
/// - Offset 40: nMaxHeight (u32) - OUTPUT
/// - Offset 44: nMaxMBCount (u32) - OUTPUT
/// - Offset 48: nMinWidth (u16) - OUTPUT
/// - Offset 50: nMinHeight (u16) - OUTPUT
#[no_mangle]
pub unsafe extern "C" fn cuvidGetDecoderCaps(caps: *mut c_void) -> CUresult {
    // Initialize logging on first call
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("RGPU_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .try_init();

    if caps.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }

    let caps_bytes = caps as *const u8;

    // Read input fields from the CUVIDDECODECAPS struct
    let codec_type = (caps_bytes.add(0) as *const u32).read_unaligned();
    let chroma_format = (caps_bytes.add(4) as *const u32).read_unaligned();
    let bit_depth_minus8 = (caps_bytes.add(8) as *const u32).read_unaligned();

    let cmd = NvdecCommand::GetDecoderCaps {
        codec_type,
        chroma_format,
        bit_depth_minus8,
    };

    let resp = send_nvdec_command(cmd);

    match resp {
        NvdecResponse::DecoderCaps {
            is_supported,
            num_nvdecs,
            min_width,
            min_height,
            max_width,
            max_height,
            max_mb_count,
        } => {
            let caps_write = caps as *mut u8;

            // Write output fields
            let b_is_supported: u32 = if is_supported { 1 } else { 0 };
            (caps_write.add(24) as *mut u32).write_unaligned(b_is_supported);
            (caps_write.add(28) as *mut u32).write_unaligned(num_nvdecs);
            (caps_write.add(36) as *mut u32).write_unaligned(max_width);
            (caps_write.add(40) as *mut u32).write_unaligned(max_height);
            (caps_write.add(44) as *mut u32).write_unaligned(max_mb_count);
            (caps_write.add(48) as *mut u16).write_unaligned(min_width as u16);
            (caps_write.add(50) as *mut u16).write_unaligned(min_height as u16);

            CUDA_SUCCESS
        }
        NvdecResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// DECODER LIFECYCLE
// ═══════════════════════════════════════════════════════════════════════════

/// Create a decoder. Serializes the entire CUVIDDECODECREATEINFO struct as raw bytes.
#[no_mangle]
pub unsafe extern "C" fn cuvidCreateDecoder(
    decoder_out: *mut CUvideodecoder,
    params: *mut c_void,
) -> CUresult {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("RGPU_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .try_init();

    if decoder_out.is_null() || params.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }

    // CUVIDDECODECREATEINFO is a large struct. We copy the raw bytes.
    // The size varies by CUDA version but is typically around 220-256 bytes.
    // We send a generous amount to cover all fields.
    let struct_size = 256usize;
    let create_info = std::slice::from_raw_parts(params as *const u8, struct_size).to_vec();

    let cmd = NvdecCommand::CreateDecoder { create_info };
    let resp = send_nvdec_command(cmd);

    match resp {
        NvdecResponse::DecoderCreated { handle } => {
            let local_id = handle_store::store_decoder(handle);
            *decoder_out = local_id as CUvideodecoder;
            CUDA_SUCCESS
        }
        NvdecResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

/// Destroy a decoder.
#[no_mangle]
pub unsafe extern "C" fn cuvidDestroyDecoder(decoder: CUvideodecoder) -> CUresult {
    let local_id = decoder as u64;
    let handle = match handle_store::get_decoder(local_id) {
        Some(h) => h,
        None => {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    let cmd = NvdecCommand::DestroyDecoder { decoder: handle };
    let resp = send_nvdec_command(cmd);

    match resp {
        NvdecResponse::Success => {
            handle_store::remove_decoder(local_id);
            CUDA_SUCCESS
        }
        ref e => response_error_code(e),
    }
}

/// Decode a picture. Extracts bitstream data from CUVIDPICPARAMS and sends both
/// the struct and the bitstream data to the server.
///
/// CUVIDPICPARAMS bitstream fields (64-bit):
/// - Offset 24: nBitstreamDataLen (u32) - length of bitstream
/// - Offset 32: pBitstreamData (pointer, u64) - pointer to compressed data
#[no_mangle]
pub unsafe extern "C" fn cuvidDecodePicture(
    decoder: CUvideodecoder,
    picparams: *mut c_void,
) -> CUresult {
    if picparams.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }

    let local_id = decoder as u64;
    let handle = match handle_store::get_decoder(local_id) {
        Some(h) => h,
        None => {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    let params_ptr = picparams as *const u8;

    // Read bitstream length and pointer from CUVIDPICPARAMS
    let bs_len = (params_ptr.add(24) as *const u32).read_unaligned() as usize;
    let bs_ptr = (params_ptr.add(32) as *const u64).read_unaligned() as *const u8;

    // Copy bitstream data
    let bitstream_data = if bs_len > 0 && !bs_ptr.is_null() {
        std::slice::from_raw_parts(bs_ptr, bs_len).to_vec()
    } else {
        Vec::new()
    };

    // Copy the raw CUVIDPICPARAMS struct.
    // The size varies but is typically around 1400-2000 bytes depending on codec.
    // We use a generous size to cover all codecs (H.264, HEVC, VP9, AV1, etc.).
    let pic_params_size = 2048usize;
    let pic_params = std::slice::from_raw_parts(params_ptr, pic_params_size).to_vec();

    let cmd = NvdecCommand::DecodePicture {
        decoder: handle,
        pic_params,
        bitstream_data,
    };

    let resp = send_nvdec_command(cmd);

    match resp {
        NvdecResponse::Success => CUDA_SUCCESS,
        ref e => response_error_code(e),
    }
}

/// Get decode status for a picture index.
///
/// CUVIDGETDECODESTATUS struct: we write decodeStatus at offset 4 (after reserved field).
#[no_mangle]
pub unsafe extern "C" fn cuvidGetDecodeStatus(
    decoder: CUvideodecoder,
    pic_idx: c_int,
    status: *mut c_void,
) -> CUresult {
    if status.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }

    let local_id = decoder as u64;
    let handle = match handle_store::get_decoder(local_id) {
        Some(h) => h,
        None => {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    let cmd = NvdecCommand::GetDecodeStatus {
        decoder: handle,
        picture_index: pic_idx,
    };

    let resp = send_nvdec_command(cmd);

    match resp {
        NvdecResponse::DecodeStatus { decode_status } => {
            // Write decode_status at offset 4 in the CUVIDGETDECODESTATUS struct
            // (offset 0-3 is a reserved field)
            let status_write = status as *mut u8;
            (status_write.add(4) as *mut i32).write_unaligned(decode_status);
            CUDA_SUCCESS
        }
        NvdecResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

/// Reconfigure decoder for resolution or parameter changes.
#[no_mangle]
pub unsafe extern "C" fn cuvidReconfigureDecoder(
    decoder: CUvideodecoder,
    params: *mut c_void,
) -> CUresult {
    if params.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }

    let local_id = decoder as u64;
    let handle = match handle_store::get_decoder(local_id) {
        Some(h) => h,
        None => {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    // CUVIDRECONFIGUREDECODERINFO struct is around 64 bytes
    let struct_size = 128usize;
    let reconfig_params =
        std::slice::from_raw_parts(params as *const u8, struct_size).to_vec();

    let cmd = NvdecCommand::ReconfigureDecoder {
        decoder: handle,
        reconfig_params,
    };

    let resp = send_nvdec_command(cmd);

    match resp {
        NvdecResponse::Success => CUDA_SUCCESS,
        ref e => response_error_code(e),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// FRAME MAPPING
// ═══════════════════════════════════════════════════════════════════════════

/// Map a decoded video frame, returning a CUDA device pointer and pitch.
///
/// This is the 64-bit version. cuvidMapVideoFrame forwards here too.
#[no_mangle]
pub unsafe extern "C" fn cuvidMapVideoFrame64(
    decoder: CUvideodecoder,
    pic_idx: c_int,
    dev_ptr: *mut u64,
    pitch: *mut c_uint,
    params: *mut c_void,
) -> CUresult {
    if dev_ptr.is_null() || pitch.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }

    let local_id = decoder as u64;
    let handle = match handle_store::get_decoder(local_id) {
        Some(h) => h,
        None => {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    // CUVIDPROCPARAMS struct is around 128 bytes
    let proc_params = if !params.is_null() {
        let struct_size = 128usize;
        std::slice::from_raw_parts(params as *const u8, struct_size).to_vec()
    } else {
        Vec::new()
    };

    let cmd = NvdecCommand::MapVideoFrame {
        decoder: handle,
        picture_index: pic_idx,
        proc_params,
    };

    let resp = send_nvdec_command(cmd);

    match resp {
        NvdecResponse::VideoFrameMapped {
            device_ptr: frame_handle,
            pitch: frame_pitch,
        } => {
            // Store the mapped frame handle so we can unmap it later.
            // The local_id serves as the device pointer from the application's perspective.
            let frame_local_id = handle_store::store_mapped_frame(frame_handle);
            *dev_ptr = frame_local_id;
            *pitch = frame_pitch;
            CUDA_SUCCESS
        }
        NvdecResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

/// Unmap a previously mapped video frame.
///
/// This is the 64-bit version. cuvidUnmapVideoFrame forwards here too.
#[no_mangle]
pub unsafe extern "C" fn cuvidUnmapVideoFrame64(
    decoder: CUvideodecoder,
    mapped_frame: u64,
) -> CUresult {
    let decoder_local_id = decoder as u64;
    let decoder_handle = match handle_store::get_decoder(decoder_local_id) {
        Some(h) => h,
        None => {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    let frame_handle = match handle_store::get_mapped_frame(mapped_frame) {
        Some(h) => h,
        None => {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    let cmd = NvdecCommand::UnmapVideoFrame {
        decoder: decoder_handle,
        mapped_frame: frame_handle,
    };

    let resp = send_nvdec_command(cmd);

    match resp {
        NvdecResponse::Success => {
            handle_store::remove_mapped_frame(mapped_frame);
            CUDA_SUCCESS
        }
        ref e => response_error_code(e),
    }
}

// ── 32-bit aliases (forward to 64-bit versions) ─────────────────────────

/// 32-bit version of cuvidMapVideoFrame64. Forwards to the 64-bit version.
#[no_mangle]
pub unsafe extern "C" fn cuvidMapVideoFrame(
    decoder: CUvideodecoder,
    pic_idx: c_int,
    dev_ptr: *mut u64,
    pitch: *mut c_uint,
    params: *mut c_void,
) -> CUresult {
    cuvidMapVideoFrame64(decoder, pic_idx, dev_ptr, pitch, params)
}

/// 32-bit version of cuvidUnmapVideoFrame64. Forwards to the 64-bit version.
#[no_mangle]
pub unsafe extern "C" fn cuvidUnmapVideoFrame(
    decoder: CUvideodecoder,
    mapped_frame: u64,
) -> CUresult {
    cuvidUnmapVideoFrame64(decoder, mapped_frame)
}

// ═══════════════════════════════════════════════════════════════════════════
// VIDEO PARSER (STUB - NOT SUPPORTED)
// ═══════════════════════════════════════════════════════════════════════════

// FFmpeg does its own parsing and calls the decoder-level functions directly.
// The CUVID video parser uses C callbacks which cannot work over network IPC.
// These stubs return CUDA_ERROR_NOT_SUPPORTED.

/// Create a video parser (stub - returns NOT_SUPPORTED).
#[no_mangle]
pub unsafe extern "C" fn cuvidCreateVideoParser(
    _parser: *mut CUvideoparser,
    _params: *mut c_void,
) -> CUresult {
    CUDA_ERROR_NOT_SUPPORTED
}

/// Parse video data (stub - returns NOT_SUPPORTED).
#[no_mangle]
pub unsafe extern "C" fn cuvidParseVideoData(
    _parser: CUvideoparser,
    _packet: *mut c_void,
) -> CUresult {
    CUDA_ERROR_NOT_SUPPORTED
}

/// Destroy a video parser (stub - returns NOT_SUPPORTED).
#[no_mangle]
pub unsafe extern "C" fn cuvidDestroyVideoParser(
    _parser: CUvideoparser,
) -> CUresult {
    CUDA_ERROR_NOT_SUPPORTED
}

// ═══════════════════════════════════════════════════════════════════════════
// CONTEXT LOCK
// ═══════════════════════════════════════════════════════════════════════════

/// Create a video context lock for multi-threaded decode.
#[no_mangle]
pub unsafe extern "C" fn cuvidCtxLockCreate(
    lock_out: *mut CUvideoctxlock,
    ctx: CUcontext,
) -> CUresult {
    if lock_out.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }

    // Resolve the CUDA context local ID to a proper NetworkHandle via cross-DLL lookup.
    let ctx_handle = match resolve_cuda_ctx_handle(ctx as u64) {
        Some(h) => {
            debug!("NVDEC CtxLockCreate: resolved CUDA ctx -> {:?}", h);
            h
        }
        None => {
            error!("NVDEC CtxLockCreate: could not resolve CUDA context local_id={:#x}", ctx as u64);
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    let cmd = NvdecCommand::CtxLockCreate {
        cuda_context: ctx_handle,
    };

    let resp = send_nvdec_command(cmd);

    match resp {
        NvdecResponse::CtxLockCreated { handle } => {
            let local_id = handle_store::store_ctx_lock(handle);
            *lock_out = local_id as CUvideoctxlock;
            CUDA_SUCCESS
        }
        NvdecResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

/// Destroy a video context lock.
#[no_mangle]
pub unsafe extern "C" fn cuvidCtxLockDestroy(lock: CUvideoctxlock) -> CUresult {
    let local_id = lock as u64;
    let handle = match handle_store::get_ctx_lock(local_id) {
        Some(h) => h,
        None => {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    let cmd = NvdecCommand::CtxLockDestroy { lock: handle };
    let resp = send_nvdec_command(cmd);

    match resp {
        NvdecResponse::Success => {
            handle_store::remove_ctx_lock(local_id);
            CUDA_SUCCESS
        }
        ref e => response_error_code(e),
    }
}

/// Lock a video context lock (no-op locally, actual locking happens server-side).
#[no_mangle]
pub unsafe extern "C" fn cuvidCtxLock(
    _lock: CUvideoctxlock,
    _reserved: c_uint,
) -> CUresult {
    CUDA_SUCCESS
}

/// Unlock a video context lock (no-op locally, actual unlocking happens server-side).
#[no_mangle]
pub unsafe extern "C" fn cuvidCtxUnlock(
    _lock: CUvideoctxlock,
    _reserved: c_uint,
) -> CUresult {
    CUDA_SUCCESS
}
