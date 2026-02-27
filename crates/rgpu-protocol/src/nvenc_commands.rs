use serde::{Deserialize, Serialize};

use crate::handle::NetworkHandle;

/// GUID represented as 16 bytes for network serialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct NvGuid(pub [u8; 16]);

/// NVENC Video Encoder API commands sent from client to server.
#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub enum NvencCommand {
    // ── Version query ────────────────────────────────────────
    /// Get maximum supported NVENC API version.
    GetMaxSupportedVersion,

    // ── Session management ───────────────────────────────────
    /// Open an encoding session.
    OpenEncodeSession {
        /// CUDA context handle (from our CUDA interpose).
        cuda_context: NetworkHandle,
        /// Device type (1 = CUDA).
        device_type: u32,
    },

    /// Destroy an encoding session.
    DestroyEncoder {
        encoder: NetworkHandle,
    },

    // ── Capability queries ───────────────────────────────────
    /// Get count of supported encode GUIDs.
    GetEncodeGUIDCount {
        encoder: NetworkHandle,
    },

    /// Get list of supported encode GUIDs.
    GetEncodeGUIDs {
        encoder: NetworkHandle,
    },

    /// Get count of supported presets for a codec.
    GetEncodePresetCount {
        encoder: NetworkHandle,
        encode_guid: NvGuid,
    },

    /// Get list of supported preset GUIDs for a codec.
    GetEncodePresetGUIDs {
        encoder: NetworkHandle,
        encode_guid: NvGuid,
    },

    /// Get preset configuration for a codec/preset combination.
    GetEncodePresetConfig {
        encoder: NetworkHandle,
        encode_guid: NvGuid,
        preset_guid: NvGuid,
    },

    /// Query encoder capabilities for a specific attribute.
    GetEncodeCaps {
        encoder: NetworkHandle,
        encode_guid: NvGuid,
        /// NV_ENC_CAPS_* attribute to query.
        caps_param: i32,
    },

    /// Get count of supported input formats for a codec.
    GetInputFormatCount {
        encoder: NetworkHandle,
        encode_guid: NvGuid,
    },

    /// Get list of supported input formats for a codec.
    GetInputFormats {
        encoder: NetworkHandle,
        encode_guid: NvGuid,
    },

    // ── Encoder initialization ───────────────────────────────
    /// Initialize the encoder with parameters.
    InitializeEncoder {
        encoder: NetworkHandle,
        /// Serialized NV_ENC_INITIALIZE_PARAMS as raw bytes.
        params: Vec<u8>,
    },

    /// Reconfigure encoder dynamically.
    ReconfigureEncoder {
        encoder: NetworkHandle,
        /// Serialized NV_ENC_RECONFIGURE_PARAMS as raw bytes.
        params: Vec<u8>,
    },

    // ── Input buffer management ──────────────────────────────
    /// Create an input buffer.
    CreateInputBuffer {
        encoder: NetworkHandle,
        width: u32,
        height: u32,
        /// NV_ENC_BUFFER_FORMAT value.
        buffer_fmt: u32,
    },

    /// Destroy an input buffer.
    DestroyInputBuffer {
        encoder: NetworkHandle,
        input_buffer: NetworkHandle,
    },

    /// Lock input buffer for CPU writing.
    LockInputBuffer {
        encoder: NetworkHandle,
        input_buffer: NetworkHandle,
    },

    /// Unlock input buffer after CPU writing.
    UnlockInputBuffer {
        encoder: NetworkHandle,
        input_buffer: NetworkHandle,
        /// The pixel data written by the client (only for CPU-side input).
        data: Vec<u8>,
        pitch: u32,
    },

    // ── Bitstream buffer management ──────────────────────────
    /// Create a bitstream output buffer.
    CreateBitstreamBuffer {
        encoder: NetworkHandle,
    },

    /// Destroy a bitstream buffer.
    DestroyBitstreamBuffer {
        encoder: NetworkHandle,
        bitstream_buffer: NetworkHandle,
    },

    /// Lock bitstream to read encoded data.
    LockBitstream {
        encoder: NetworkHandle,
        bitstream_buffer: NetworkHandle,
    },

    /// Unlock bitstream after reading.
    UnlockBitstream {
        encoder: NetworkHandle,
        bitstream_buffer: NetworkHandle,
    },

    // ── Resource registration (for CUDA/Vulkan inputs) ──────
    /// Register a CUDA/Vulkan resource for use as encoder input.
    RegisterResource {
        encoder: NetworkHandle,
        /// Resource type (0=DirectX, 1=CUDA, etc.).
        resource_type: u32,
        /// The CUDA device pointer or other resource handle.
        resource: NetworkHandle,
        width: u32,
        height: u32,
        pitch: u32,
        buffer_fmt: u32,
    },

    /// Unregister a previously registered resource.
    UnregisterResource {
        encoder: NetworkHandle,
        registered_resource: NetworkHandle,
    },

    /// Map a registered resource for encoding.
    MapInputResource {
        encoder: NetworkHandle,
        registered_resource: NetworkHandle,
    },

    /// Unmap a previously mapped resource.
    UnmapInputResource {
        encoder: NetworkHandle,
        mapped_resource: NetworkHandle,
    },

    // ── Encoding ─────────────────────────────────────────────
    /// Encode a picture.
    EncodePicture {
        encoder: NetworkHandle,
        /// Serialized NV_ENC_PIC_PARAMS as raw bytes (minus the large buffers).
        params: Vec<u8>,
        /// Input buffer or mapped resource handle.
        input: NetworkHandle,
        /// Output bitstream buffer handle.
        output: NetworkHandle,
        /// Picture type (0=P, 1=B, 2=I, 3=IDR, 4=BI, 5=SKIPPED).
        picture_type: u32,
    },

    // ── Parameter retrieval ──────────────────────────────────
    /// Get SPS/PPS sequence parameter data.
    GetSequenceParams {
        encoder: NetworkHandle,
    },

    /// Get encoding statistics.
    GetEncodeStats {
        encoder: NetworkHandle,
    },

    /// Invalidate reference frames for error recovery.
    InvalidateRefFrames {
        encoder: NetworkHandle,
        /// Timestamp of the invalid reference frame.
        invalid_ref_frame_timestamp: u64,
    },

    // ── Async events ─────────────────────────────────────────
    /// Register an async completion event.
    RegisterAsyncEvent {
        encoder: NetworkHandle,
    },

    /// Unregister an async completion event.
    UnregisterAsyncEvent {
        encoder: NetworkHandle,
        event: NetworkHandle,
    },
}

/// NVENC API responses from server to client.
#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub enum NvencResponse {
    /// Operation succeeded with no additional data.
    Success,

    /// Operation failed.
    Error {
        /// NVENCSTATUS error code.
        code: i32,
        message: String,
    },

    /// Maximum supported API version.
    MaxSupportedVersion {
        version: u32,
    },

    /// Encoder session created.
    EncoderOpened {
        handle: NetworkHandle,
    },

    /// Encode GUID count.
    GUIDCount(u32),

    /// List of encode GUIDs.
    GUIDs(Vec<NvGuid>),

    /// Preset configuration (serialized as raw bytes).
    PresetConfig(Vec<u8>),

    /// Capability value for a specific attribute.
    CapsValue(i32),

    /// Input format count.
    InputFormatCount(u32),

    /// List of supported input formats (NV_ENC_BUFFER_FORMAT values).
    InputFormats(Vec<u32>),

    /// Input buffer created.
    InputBufferCreated {
        handle: NetworkHandle,
    },

    /// Input buffer locked — returns pitch and buffer size.
    InputBufferLocked {
        pitch: u32,
        buffer_size: u32,
    },

    /// Bitstream buffer created.
    BitstreamBufferCreated {
        handle: NetworkHandle,
    },

    /// Bitstream locked — returns encoded data.
    BitstreamData {
        data: Vec<u8>,
        /// NV_ENC_PIC_TYPE of the encoded frame.
        picture_type: u32,
        /// Frame index.
        frame_idx: u32,
        /// Output timestamp.
        output_timestamp: u64,
    },

    /// Registered resource handle.
    ResourceRegistered {
        handle: NetworkHandle,
    },

    /// Mapped resource handle.
    ResourceMapped {
        handle: NetworkHandle,
    },

    /// Sequence parameter data (SPS/PPS).
    SequenceParams {
        data: Vec<u8>,
    },

    /// Encoding statistics.
    EncodeStats {
        /// Serialized stats as raw bytes.
        data: Vec<u8>,
    },

    /// Async event registered.
    AsyncEventRegistered {
        handle: NetworkHandle,
    },
}
