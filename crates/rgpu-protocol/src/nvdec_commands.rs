use serde::{Deserialize, Serialize};

use crate::handle::NetworkHandle;

/// NVDEC/CUVID Video Decoder API commands sent from client to server.
#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub enum NvdecCommand {
    // ── Capability query ─────────────────────────────────────
    /// Query decoder capabilities.
    GetDecoderCaps {
        /// Codec type (cudaVideoCodec enum value).
        codec_type: u32,
        /// Chroma format (cudaVideoChromaFormat enum value).
        chroma_format: u32,
        /// Bit depth minus 8 (0=8-bit, 2=10-bit, 4=12-bit).
        bit_depth_minus8: u32,
    },

    // ── Decoder lifecycle ────────────────────────────────────
    /// Create a decoder.
    CreateDecoder {
        /// Serialized CUVIDDECODECREATEINFO as raw bytes.
        create_info: Vec<u8>,
    },

    /// Destroy a decoder.
    DestroyDecoder {
        decoder: NetworkHandle,
    },

    /// Decode a picture.
    DecodePicture {
        decoder: NetworkHandle,
        /// Serialized CUVIDPICPARAMS as raw bytes (includes codec-specific data).
        pic_params: Vec<u8>,
        /// Compressed bitstream data (the actual NAL units / video data).
        bitstream_data: Vec<u8>,
    },

    /// Get decode status for a picture index.
    GetDecodeStatus {
        decoder: NetworkHandle,
        picture_index: i32,
    },

    /// Reconfigure decoder for resolution/parameter changes.
    ReconfigureDecoder {
        decoder: NetworkHandle,
        /// Serialized CUVIDRECONFIGUREDECODERINFO as raw bytes.
        reconfig_params: Vec<u8>,
    },

    // ── Frame mapping ────────────────────────────────────────
    /// Map a decoded video frame to get a CUDA device pointer.
    MapVideoFrame {
        decoder: NetworkHandle,
        picture_index: i32,
        /// Serialized CUVIDPROCPARAMS as raw bytes.
        proc_params: Vec<u8>,
    },

    /// Unmap a previously mapped video frame.
    UnmapVideoFrame {
        decoder: NetworkHandle,
        /// The device pointer returned by MapVideoFrame.
        mapped_frame: NetworkHandle,
    },

    // ── Video parser (optional, for apps that use it) ────────
    /// Create a video parser.
    CreateVideoParser {
        /// Codec type for the parser.
        codec_type: u32,
        /// Maximum number of decode surfaces.
        max_num_decode_surfaces: u32,
        /// Clock rate (0 = default).
        clock_rate: u32,
        /// Error threshold (0 = default 100).
        error_threshold: u32,
        /// Maximum display delay (0 = default 4).
        max_display_delay: u32,
    },

    /// Parse video data (feed bitstream to parser).
    ParseVideoData {
        parser: NetworkHandle,
        /// The bitstream packet data.
        payload: Vec<u8>,
        /// Packet flags (CUVID_PKT_*).
        flags: u32,
        /// Presentation timestamp.
        timestamp: i64,
    },

    /// Destroy a video parser.
    DestroyVideoParser {
        parser: NetworkHandle,
    },

    // ── Context locking (for multi-threaded decode) ──────────
    /// Create a context lock.
    CtxLockCreate {
        /// CUDA context handle.
        cuda_context: NetworkHandle,
    },

    /// Destroy a context lock.
    CtxLockDestroy {
        lock: NetworkHandle,
    },
}

/// NVDEC/CUVID API responses from server to client.
#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub enum NvdecResponse {
    /// Operation succeeded with no additional data.
    Success,

    /// Operation failed.
    Error {
        /// CUresult error code.
        code: i32,
        message: String,
    },

    /// Decoder capabilities.
    DecoderCaps {
        /// Whether this codec/chroma/bitdepth combo is supported.
        is_supported: bool,
        /// Number of NVDEC engines on the GPU.
        num_nvdecs: u32,
        /// Minimum supported width.
        min_width: u32,
        /// Minimum supported height.
        min_height: u32,
        /// Maximum supported width.
        max_width: u32,
        /// Maximum supported height.
        max_height: u32,
        /// Maximum MB count.
        max_mb_count: u32,
    },

    /// Decoder created.
    DecoderCreated {
        handle: NetworkHandle,
    },

    /// Decode status for a picture.
    DecodeStatus {
        /// 0 = invalid/unavailable, 1 = in progress, 2 = success, 8 = error/concealed.
        decode_status: i32,
    },

    /// Mapped video frame info.
    VideoFrameMapped {
        /// CUDA device pointer to the decoded frame (as NetworkHandle for routing).
        device_ptr: NetworkHandle,
        /// Pitch (stride in bytes) of the mapped frame.
        pitch: u32,
    },

    /// Video parser created.
    VideoParserCreated {
        handle: NetworkHandle,
    },

    /// Parser produced decode/display callbacks — returns actions to take.
    ParseResult {
        /// Decoder calls requested by the parser.
        decode_calls: Vec<NvdecParserDecodeCall>,
        /// Display calls requested by the parser.
        display_calls: Vec<NvdecParserDisplayCall>,
    },

    /// Context lock created.
    CtxLockCreated {
        handle: NetworkHandle,
    },
}

/// A decode call requested by the parser callback.
#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct NvdecParserDecodeCall {
    /// Serialized CUVIDPICPARAMS.
    pub pic_params: Vec<u8>,
}

/// A display call requested by the parser callback.
#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct NvdecParserDisplayCall {
    /// Picture index to display.
    pub picture_index: i32,
    /// Presentation timestamp.
    pub timestamp: i64,
}
