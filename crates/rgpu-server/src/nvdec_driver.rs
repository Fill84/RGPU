//! Dynamic loading of the NVIDIA CUVID (NVDEC) video decoder library.
//!
//! Uses `libloading` to load `nvcuvid.dll` (Windows) or `libnvcuvid.so.1` (Linux)
//! and provides safe Rust wrappers around the raw CUVID API functions.
//!
//! Unlike NVENC which uses a function-table pattern, CUVID exports all functions
//! directly from the shared library.

use std::ffi::{c_int, c_uint, c_void};
use std::sync::Arc;

use libloading::{Library, Symbol};
use tracing::{debug, info};

/// CUVID result type (CUresult — same as CUDA driver).
pub type CUresult = c_int;

/// Opaque decoder handle.
pub type CUvideodecoder = *mut c_void;

/// Opaque video parser handle.
pub type CUvideoparser = *mut c_void;

/// Opaque context lock handle.
pub type CUvideoctxlock = *mut c_void;

/// CUDA context (opaque pointer, same as cuda_driver::CUcontext).
pub type CUcontext = *mut c_void;

pub const CUDA_SUCCESS: CUresult = 0;
pub const CUDA_ERROR_NOT_SUPPORTED: CUresult = 801;

// ── Function pointer type definitions ────────────────────────────────────

// Capability query
type FnCuvidGetDecoderCaps =
    unsafe extern "C" fn(caps: *mut c_void) -> CUresult;

// Decoder lifecycle
type FnCuvidCreateDecoder =
    unsafe extern "C" fn(decoder: *mut CUvideodecoder, params: *mut c_void) -> CUresult;
type FnCuvidDestroyDecoder =
    unsafe extern "C" fn(decoder: CUvideodecoder) -> CUresult;
type FnCuvidDecodePicture =
    unsafe extern "C" fn(decoder: CUvideodecoder, params: *mut c_void) -> CUresult;
type FnCuvidGetDecodeStatus =
    unsafe extern "C" fn(decoder: CUvideodecoder, pic_idx: c_int, status: *mut c_void) -> CUresult;
type FnCuvidReconfigureDecoder =
    unsafe extern "C" fn(decoder: CUvideodecoder, params: *mut c_void) -> CUresult;

// Frame mapping (64-bit variants)
type FnCuvidMapVideoFrame64 =
    unsafe extern "C" fn(decoder: CUvideodecoder, pic_idx: c_int, devptr: *mut u64, pitch: *mut c_uint, params: *mut c_void) -> CUresult;
type FnCuvidUnmapVideoFrame64 =
    unsafe extern "C" fn(decoder: CUvideodecoder, devptr: u64) -> CUresult;

// Video parser
type FnCuvidCreateVideoParser =
    unsafe extern "C" fn(parser: *mut CUvideoparser, params: *mut c_void) -> CUresult;
type FnCuvidParseVideoData =
    unsafe extern "C" fn(parser: CUvideoparser, packet: *mut c_void) -> CUresult;
type FnCuvidDestroyVideoParser =
    unsafe extern "C" fn(parser: CUvideoparser) -> CUresult;

// Context locking
type FnCuvidCtxLockCreate =
    unsafe extern "C" fn(lock: *mut CUvideoctxlock, ctx: CUcontext) -> CUresult;
type FnCuvidCtxLockDestroy =
    unsafe extern "C" fn(lock: CUvideoctxlock) -> CUresult;

/// Dynamically loaded CUVID library with function pointers.
pub struct NvdecDriver {
    _lib: Library,
    // Capability query
    cuvid_get_decoder_caps: Option<FnCuvidGetDecoderCaps>,
    // Decoder lifecycle
    cuvid_create_decoder: FnCuvidCreateDecoder,
    cuvid_destroy_decoder: FnCuvidDestroyDecoder,
    cuvid_decode_picture: FnCuvidDecodePicture,
    cuvid_get_decode_status: Option<FnCuvidGetDecodeStatus>,
    cuvid_reconfigure_decoder: Option<FnCuvidReconfigureDecoder>,
    // Frame mapping
    cuvid_map_video_frame: FnCuvidMapVideoFrame64,
    cuvid_unmap_video_frame: FnCuvidUnmapVideoFrame64,
    // Video parser
    cuvid_create_video_parser: Option<FnCuvidCreateVideoParser>,
    cuvid_parse_video_data: Option<FnCuvidParseVideoData>,
    cuvid_destroy_video_parser: Option<FnCuvidDestroyVideoParser>,
    // Context locking
    cuvid_ctx_lock_create: Option<FnCuvidCtxLockCreate>,
    cuvid_ctx_lock_destroy: Option<FnCuvidCtxLockDestroy>,
}

// SAFETY: The CUVID library handles are valid from any thread.
// The CUVID API handles thread safety via the context lock mechanism.
unsafe impl Send for NvdecDriver {}
unsafe impl Sync for NvdecDriver {}

impl NvdecDriver {
    /// Load the CUVID library and resolve all function pointers.
    pub fn load() -> Result<Arc<Self>, String> {
        let lib = Self::load_library()?;

        unsafe {
            let driver = Self {
                // Capability query (added in later CUVID versions)
                cuvid_get_decoder_caps: Self::load_fn_opt(&lib, "cuvidGetDecoderCaps"),
                // Decoder lifecycle
                cuvid_create_decoder: Self::load_fn(&lib, "cuvidCreateDecoder")?,
                cuvid_destroy_decoder: Self::load_fn(&lib, "cuvidDestroyDecoder")?,
                cuvid_decode_picture: Self::load_fn(&lib, "cuvidDecodePicture")?,
                cuvid_get_decode_status: Self::load_fn_opt(&lib, "cuvidGetDecodeStatus"),
                cuvid_reconfigure_decoder: Self::load_fn_opt(&lib, "cuvidReconfigureDecoder"),
                // Frame mapping — prefer 64-bit variants, fall back to 32-bit
                cuvid_map_video_frame: Self::load_fn::<FnCuvidMapVideoFrame64>(&lib, "cuvidMapVideoFrame64")
                    .or_else(|_| Self::load_fn(&lib, "cuvidMapVideoFrame"))?,
                cuvid_unmap_video_frame: Self::load_fn::<FnCuvidUnmapVideoFrame64>(&lib, "cuvidUnmapVideoFrame64")
                    .or_else(|_| Self::load_fn(&lib, "cuvidUnmapVideoFrame"))?,
                // Video parser
                cuvid_create_video_parser: Self::load_fn_opt(&lib, "cuvidCreateVideoParser"),
                cuvid_parse_video_data: Self::load_fn_opt(&lib, "cuvidParseVideoData"),
                cuvid_destroy_video_parser: Self::load_fn_opt(&lib, "cuvidDestroyVideoParser"),
                // Context locking
                cuvid_ctx_lock_create: Self::load_fn_opt(&lib, "cuvidCtxLockCreate"),
                cuvid_ctx_lock_destroy: Self::load_fn_opt(&lib, "cuvidCtxLockDestroy"),
                _lib: lib,
            };

            info!("CUVID (NVDEC) driver loaded successfully");
            Ok(Arc::new(driver))
        }
    }

    fn load_library() -> Result<Library, String> {
        // On Windows, try the _real.dll first to avoid loading our own interpose DLL
        // when System32\nvcuvid.dll has been replaced by the RGPU installer.
        #[cfg(target_os = "windows")]
        let lib_names = &["nvcuvid_real.dll", "nvcuvid.dll"];

        #[cfg(target_os = "linux")]
        let lib_names = &["libnvcuvid.so.1", "libnvcuvid.so"];

        #[cfg(target_os = "macos")]
        let lib_names: &[&str] = &[];

        let mut last_err = String::new();
        for name in lib_names {
            match unsafe { Library::new(name) } {
                Ok(lib) => {
                    info!("loaded CUVID library from: {}", name);
                    return Ok(lib);
                }
                Err(e) => {
                    last_err = format!("{}: {}", name, e);
                    debug!("failed to load {}: {}", name, e);
                }
            }
        }

        Err(format!("failed to load CUVID library: {}", last_err))
    }

    unsafe fn load_fn<F: Copy>(lib: &Library, name: &str) -> Result<F, String> {
        let sym: Symbol<F> = lib
            .get(name.as_bytes())
            .map_err(|e| format!("failed to load {}: {}", name, e))?;
        Ok(*sym)
    }

    unsafe fn load_fn_opt<F: Copy>(lib: &Library, name: &str) -> Option<F> {
        lib.get(name.as_bytes()).ok().map(|s: Symbol<F>| *s)
    }

    // ── Capability Query ─────────────────────────────────────────────

    /// Query decoder capabilities.
    ///
    /// `caps_struct` must be a properly laid-out CUVIDDECODECAPS struct as raw bytes.
    /// The codec_type, chroma_format, and bit_depth_minus8 fields must be set by the caller.
    /// On success the struct is filled with capability information.
    pub fn get_decoder_caps(&self, caps_struct: &mut [u8]) -> CUresult {
        if let Some(func) = self.cuvid_get_decoder_caps {
            unsafe { func(caps_struct.as_mut_ptr() as *mut c_void) }
        } else {
            CUDA_ERROR_NOT_SUPPORTED
        }
    }

    // ── Decoder Lifecycle ────────────────────────────────────────────

    /// Create a video decoder.
    ///
    /// `create_info` is a raw CUVIDDECODECREATEINFO struct as bytes.
    /// Returns the opaque decoder handle on success.
    pub fn create_decoder(&self, create_info: &mut [u8]) -> Result<CUvideodecoder, CUresult> {
        let mut decoder: CUvideodecoder = std::ptr::null_mut();
        let res = unsafe {
            (self.cuvid_create_decoder)(&mut decoder, create_info.as_mut_ptr() as *mut c_void)
        };
        if res == CUDA_SUCCESS {
            Ok(decoder)
        } else {
            Err(res)
        }
    }

    /// Destroy a video decoder.
    pub fn destroy_decoder(&self, decoder: CUvideodecoder) -> CUresult {
        unsafe { (self.cuvid_destroy_decoder)(decoder) }
    }

    /// Decode a single picture.
    ///
    /// `pic_params` is a raw CUVIDPICPARAMS struct as bytes. The bitstream data pointer
    /// inside the struct must already be set to point at valid memory.
    pub fn decode_picture(&self, decoder: CUvideodecoder, pic_params: &mut [u8]) -> CUresult {
        unsafe {
            (self.cuvid_decode_picture)(decoder, pic_params.as_mut_ptr() as *mut c_void)
        }
    }

    /// Get decode status for a picture index.
    ///
    /// `status_struct` is a raw CUVIDGETDECODESTATUS struct as bytes (output).
    pub fn get_decode_status(
        &self,
        decoder: CUvideodecoder,
        pic_idx: i32,
        status_struct: &mut [u8],
    ) -> CUresult {
        if let Some(func) = self.cuvid_get_decode_status {
            unsafe {
                func(decoder, pic_idx as c_int, status_struct.as_mut_ptr() as *mut c_void)
            }
        } else {
            CUDA_ERROR_NOT_SUPPORTED
        }
    }

    /// Reconfigure a decoder (e.g., for resolution changes).
    ///
    /// `reconfig_params` is a raw CUVIDRECONFIGUREDECODERINFO struct as bytes.
    pub fn reconfigure_decoder(
        &self,
        decoder: CUvideodecoder,
        reconfig_params: &mut [u8],
    ) -> CUresult {
        if let Some(func) = self.cuvid_reconfigure_decoder {
            unsafe { func(decoder, reconfig_params.as_mut_ptr() as *mut c_void) }
        } else {
            CUDA_ERROR_NOT_SUPPORTED
        }
    }

    // ── Frame Mapping ────────────────────────────────────────────────

    /// Map a decoded video frame to get a CUDA device pointer and pitch.
    ///
    /// `proc_params` is a raw CUVIDPROCPARAMS struct as bytes.
    /// Returns (device_pointer, pitch) on success.
    pub fn map_video_frame(
        &self,
        decoder: CUvideodecoder,
        pic_idx: i32,
        proc_params: &mut [u8],
    ) -> Result<(u64, u32), CUresult> {
        let mut devptr: u64 = 0;
        let mut pitch: c_uint = 0;
        let res = unsafe {
            (self.cuvid_map_video_frame)(
                decoder,
                pic_idx as c_int,
                &mut devptr,
                &mut pitch,
                proc_params.as_mut_ptr() as *mut c_void,
            )
        };
        if res == CUDA_SUCCESS {
            Ok((devptr, pitch as u32))
        } else {
            Err(res)
        }
    }

    /// Unmap a previously mapped video frame.
    pub fn unmap_video_frame(&self, decoder: CUvideodecoder, devptr: u64) -> CUresult {
        unsafe { (self.cuvid_unmap_video_frame)(decoder, devptr) }
    }

    // ── Video Parser ─────────────────────────────────────────────────

    /// Create a video parser.
    ///
    /// `parser_params` is a raw CUVIDPARSERPARAMS struct as bytes.
    /// Note: parser callbacks are function pointers that cannot be forwarded
    /// over the network, so this function is primarily for local use.
    pub fn create_video_parser(
        &self,
        parser_params: &mut [u8],
    ) -> Result<CUvideoparser, CUresult> {
        if let Some(func) = self.cuvid_create_video_parser {
            let mut parser: CUvideoparser = std::ptr::null_mut();
            let res = unsafe {
                func(&mut parser, parser_params.as_mut_ptr() as *mut c_void)
            };
            if res == CUDA_SUCCESS {
                Ok(parser)
            } else {
                Err(res)
            }
        } else {
            Err(CUDA_ERROR_NOT_SUPPORTED)
        }
    }

    /// Parse video data (feed bitstream to parser).
    ///
    /// `packet` is a raw CUVIDSOURCEDATAPACKET struct as bytes.
    pub fn parse_video_data(
        &self,
        parser: CUvideoparser,
        packet: &mut [u8],
    ) -> CUresult {
        if let Some(func) = self.cuvid_parse_video_data {
            unsafe { func(parser, packet.as_mut_ptr() as *mut c_void) }
        } else {
            CUDA_ERROR_NOT_SUPPORTED
        }
    }

    /// Destroy a video parser.
    pub fn destroy_video_parser(&self, parser: CUvideoparser) -> CUresult {
        if let Some(func) = self.cuvid_destroy_video_parser {
            unsafe { func(parser) }
        } else {
            CUDA_ERROR_NOT_SUPPORTED
        }
    }

    // ── Context Locking ──────────────────────────────────────────────

    /// Create a context lock for multi-threaded decoding.
    pub fn ctx_lock_create(&self, cuda_ctx: CUcontext) -> Result<CUvideoctxlock, CUresult> {
        if let Some(func) = self.cuvid_ctx_lock_create {
            let mut lock: CUvideoctxlock = std::ptr::null_mut();
            let res = unsafe { func(&mut lock, cuda_ctx) };
            if res == CUDA_SUCCESS {
                Ok(lock)
            } else {
                Err(res)
            }
        } else {
            Err(CUDA_ERROR_NOT_SUPPORTED)
        }
    }

    /// Destroy a context lock.
    pub fn ctx_lock_destroy(&self, lock: CUvideoctxlock) -> CUresult {
        if let Some(func) = self.cuvid_ctx_lock_destroy {
            unsafe { func(lock) }
        } else {
            CUDA_ERROR_NOT_SUPPORTED
        }
    }
}

/// Convert a CUresult error code to a human-readable name.
pub fn cuvid_error_name(code: CUresult) -> &'static str {
    match code {
        0 => "CUDA_SUCCESS",
        1 => "CUDA_ERROR_INVALID_VALUE",
        2 => "CUDA_ERROR_OUT_OF_MEMORY",
        3 => "CUDA_ERROR_NOT_INITIALIZED",
        4 => "CUDA_ERROR_DEINITIALIZED",
        100 => "CUDA_ERROR_NO_DEVICE",
        101 => "CUDA_ERROR_INVALID_DEVICE",
        200 => "CUDA_ERROR_INVALID_IMAGE",
        201 => "CUDA_ERROR_INVALID_CONTEXT",
        400 => "CUDA_ERROR_INVALID_HANDLE",
        700 => "CUDA_ERROR_ILLEGAL_ADDRESS",
        719 => "CUDA_ERROR_LAUNCH_FAILED",
        801 => "CUDA_ERROR_NOT_SUPPORTED",
        999 => "CUDA_ERROR_UNKNOWN",
        _ => "CUDA_ERROR_UNKNOWN",
    }
}
