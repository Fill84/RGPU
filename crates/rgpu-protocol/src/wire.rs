use std::borrow::Cow;

use crate::messages::Message;

/// Wire protocol magic bytes: "RG"
pub const MAGIC: [u8; 2] = [0x52, 0x47];

/// Maximum frame payload size: 256 MB
pub const MAX_FRAME_SIZE: u32 = 256 * 1024 * 1024;

/// Frame header size in bytes: magic(2) + flags(1) + stream_id(4) + length(4) = 11
pub const HEADER_SIZE: usize = 11;

/// Minimum payload size to attempt LZ4 compression (bytes).
/// Payloads smaller than this are sent uncompressed to avoid overhead.
const COMPRESSION_THRESHOLD: usize = 512;

bitflags::bitflags! {
    /// Frame flags byte.
    #[derive(Debug, Clone, Copy)]
    pub struct FrameFlags: u8 {
        const COMPRESSED  = 0b0000_0001;
        const HAS_BULK    = 0b0000_0010;
        const RESPONSE    = 0b0000_0100;
        const ERROR       = 0b0000_1000;
        const BATCH       = 0b0001_0000;
    }
}

/// Encode a Message into bytes (header + payload), with optional LZ4 compression.
pub fn encode_message(msg: &Message, stream_id: u32) -> Result<Vec<u8>, WireError> {
    let payload = rkyv::to_bytes::<rkyv::rancor::Error>(msg)
        .map_err(|e| WireError::Serialization(e.to_string()))?;

    // Attempt LZ4 compression for payloads above threshold
    let (final_payload, compression_flag) = if payload.len() > COMPRESSION_THRESHOLD {
        let compressed = lz4_flex::compress_prepend_size(&payload);
        if compressed.len() < payload.len() {
            (Cow::Owned(compressed), FrameFlags::COMPRESSED)
        } else {
            // Compression didn't help, send uncompressed
            (Cow::Borrowed(payload.as_slice()), FrameFlags::empty())
        }
    } else {
        (Cow::Borrowed(payload.as_slice()), FrameFlags::empty())
    };

    let msg_flags = match msg {
        Message::Error(_) => FrameFlags::ERROR,
        Message::CudaBatch(_) => FrameFlags::BATCH,
        Message::CudaResponse { .. }
        | Message::VulkanResponse { .. }
        | Message::AuthResult { .. }
        | Message::MetricsData { .. }
        | Message::Pong => FrameFlags::RESPONSE,
        _ => FrameFlags::empty(),
    };

    let flags = compression_flag | msg_flags;
    let payload_len = final_payload.len() as u32;

    let mut frame = Vec::with_capacity(HEADER_SIZE + final_payload.len());
    frame.extend_from_slice(&MAGIC);
    frame.push(flags.bits());
    frame.extend_from_slice(&stream_id.to_le_bytes());
    frame.extend_from_slice(&payload_len.to_le_bytes());
    frame.extend_from_slice(&final_payload);

    Ok(frame)
}

/// Decode a frame header. Returns (flags, stream_id, payload_length).
pub fn decode_header(header: &[u8; HEADER_SIZE]) -> Result<(FrameFlags, u32, u32), WireError> {
    if header[0] != MAGIC[0] || header[1] != MAGIC[1] {
        return Err(WireError::InvalidMagic);
    }

    let flags = FrameFlags::from_bits_truncate(header[2]);
    let stream_id = u32::from_le_bytes([header[3], header[4], header[5], header[6]]);
    let length = u32::from_le_bytes([header[7], header[8], header[9], header[10]]);

    if length > MAX_FRAME_SIZE {
        return Err(WireError::FrameTooLarge(length));
    }

    Ok((flags, stream_id, length))
}

/// Decode a message from payload bytes, decompressing if the COMPRESSED flag is set.
pub fn decode_message(payload: &[u8], flags: FrameFlags) -> Result<Message, WireError> {
    let data: Cow<'_, [u8]> = if flags.contains(FrameFlags::COMPRESSED) {
        Cow::Owned(
            lz4_flex::decompress_size_prepended(payload)
                .map_err(|e| WireError::DecompressionError(e.to_string()))?,
        )
    } else {
        Cow::Borrowed(payload)
    };
    rkyv::from_bytes::<Message, rkyv::rancor::Error>(&data)
        .map_err(|e| WireError::Serialization(e.to_string()))
}

#[derive(Debug, thiserror::Error)]
pub enum WireError {
    #[error("invalid magic bytes")]
    InvalidMagic,
    #[error("frame too large: {0} bytes")]
    FrameTooLarge(u32),
    #[error("serialization error: {0}")]
    Serialization(String),
    #[error("decompression error: {0}")]
    DecompressionError(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
