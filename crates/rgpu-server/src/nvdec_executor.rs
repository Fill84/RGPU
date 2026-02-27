use std::sync::Arc;

use dashmap::DashMap;
use tracing::{debug, error, info, warn};

use rgpu_protocol::nvdec_commands::{NvdecCommand, NvdecResponse};
use rgpu_protocol::handle::{NetworkHandle, ResourceType};

use crate::nvdec_driver::{self, NvdecDriver, CUDA_ERROR_NOT_SUPPORTED, CUDA_SUCCESS};
use crate::session::Session;

/// Server-side NVDEC (CUVID) command executor.
/// Executes CUVID video decoder commands on real GPU hardware via dynamically loaded library.
pub struct NvdecExecutor {
    /// The real CUVID driver (loaded via libloading)
    driver: Option<Arc<NvdecDriver>>,
    /// Maps NetworkHandle -> real CUvideodecoder pointer
    decoder_handles: DashMap<NetworkHandle, nvdec_driver::CUvideodecoder>,
    /// Maps NetworkHandle -> real CUvideoparser pointer
    parser_handles: DashMap<NetworkHandle, nvdec_driver::CUvideoparser>,
    /// Maps NetworkHandle -> real CUvideoctxlock pointer
    ctx_lock_handles: DashMap<NetworkHandle, nvdec_driver::CUvideoctxlock>,
    /// Maps NetworkHandle -> device pointer for mapped video frames
    mapped_frame_handles: DashMap<NetworkHandle, u64>,
}

// SAFETY: CUVID driver pointers are valid across threads when used with proper context management
unsafe impl Send for NvdecExecutor {}
unsafe impl Sync for NvdecExecutor {}

impl NvdecExecutor {
    pub fn new() -> Self {
        let driver = match NvdecDriver::load() {
            Ok(d) => {
                info!("NVDEC (CUVID) driver loaded successfully");
                Some(d)
            }
            Err(e) => {
                warn!("NVDEC (CUVID) driver not available: {} - NVDEC commands will fail", e);
                None
            }
        };

        Self {
            driver,
            decoder_handles: DashMap::new(),
            parser_handles: DashMap::new(),
            ctx_lock_handles: DashMap::new(),
            mapped_frame_handles: DashMap::new(),
        }
    }

    /// Check if the CUVID driver is available.
    fn driver(&self) -> Result<&NvdecDriver, NvdecResponse> {
        self.driver.as_deref().ok_or(NvdecResponse::Error {
            code: 3, // CUDA_ERROR_NOT_INITIALIZED
            message: "CUVID (NVDEC) driver not loaded on server".to_string(),
        })
    }

    /// Convert a CUresult to an NvdecResponse::Error.
    fn cuvid_err(code: nvdec_driver::CUresult) -> NvdecResponse {
        NvdecResponse::Error {
            code,
            message: nvdec_driver::cuvid_error_name(code).to_string(),
        }
    }

    /// Execute an NVDEC command and return the response.
    pub fn execute(&self, session: &Session, cmd: NvdecCommand) -> NvdecResponse {
        match cmd {
            // ── Capability query ─────────────────────────────────────
            NvdecCommand::GetDecoderCaps {
                codec_type,
                chroma_format,
                bit_depth_minus8,
            } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                debug!(
                    session_id = session.session_id,
                    "GetDecoderCaps(codec={}, chroma={}, bitdepth_minus8={})",
                    codec_type, chroma_format, bit_depth_minus8
                );

                // CUVIDDECODECAPS is a large struct. We construct a minimal version:
                // The first 3 u32 fields are eCodecType, eChromaFormat, nBitDepthMinus8
                // followed by reserved/padding, then output fields.
                // Total struct size is typically 132 bytes (NVIDIA Video Codec SDK 12.x).
                const CAPS_STRUCT_SIZE: usize = 132;
                let mut caps = vec![0u8; CAPS_STRUCT_SIZE];

                // Set the input fields (little-endian u32)
                caps[0..4].copy_from_slice(&codec_type.to_le_bytes());
                caps[4..8].copy_from_slice(&chroma_format.to_le_bytes());
                caps[8..12].copy_from_slice(&bit_depth_minus8.to_le_bytes());

                let res = d.get_decoder_caps(&mut caps);
                if res != CUDA_SUCCESS {
                    return Self::cuvid_err(res);
                }

                // Parse output fields from the struct.
                // Offsets (CUVIDDECODECAPS layout, SDK 12.x):
                //   0: eCodecType (u32)
                //   4: eChromaFormat (u32)
                //   8: nBitDepthMinus8 (u32)
                //  12: reserved1[3] (3 x u32 = 12 bytes)
                //  24: bIsSupported (u32, actually unsigned char but padded)
                //  28: nNumNVDECs (u32)
                //  32: nOutputFormatMask (u32)
                //  36: nMaxWidth (u32)
                //  40: nMaxHeight (u32)
                //  44: nMaxMBCount (u32)
                //  48: nMinWidth (u16)
                //  50: nMinHeight (u16)
                // These offsets may vary across SDK versions. The raw bytes
                // approach allows the client to also interpret the struct.
                let is_supported = u32::from_le_bytes([caps[24], caps[25], caps[26], caps[27]]) != 0;
                let num_nvdecs = u32::from_le_bytes([caps[28], caps[29], caps[30], caps[31]]);
                let max_width = u32::from_le_bytes([caps[36], caps[37], caps[38], caps[39]]);
                let max_height = u32::from_le_bytes([caps[40], caps[41], caps[42], caps[43]]);
                let max_mb_count = u32::from_le_bytes([caps[44], caps[45], caps[46], caps[47]]);
                let min_width = u16::from_le_bytes([caps[48], caps[49]]) as u32;
                let min_height = u16::from_le_bytes([caps[50], caps[51]]) as u32;

                NvdecResponse::DecoderCaps {
                    is_supported,
                    num_nvdecs,
                    min_width,
                    min_height,
                    max_width,
                    max_height,
                    max_mb_count,
                }
            }

            // ── Decoder lifecycle ────────────────────────────────────
            NvdecCommand::CreateDecoder { create_info } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                debug!(
                    session_id = session.session_id,
                    "CreateDecoder (create_info={} bytes)", create_info.len()
                );

                let mut info_buf = create_info;
                match d.create_decoder(&mut info_buf) {
                    Ok(decoder) => {
                        let handle = session.alloc_handle(ResourceType::CuVideoDecoder);
                        self.decoder_handles.insert(handle, decoder);
                        info!(
                            session_id = session.session_id,
                            "NVDEC decoder created: {:?}", handle
                        );
                        NvdecResponse::DecoderCreated { handle }
                    }
                    Err(code) => {
                        error!(
                            session_id = session.session_id,
                            "cuvidCreateDecoder failed: {} ({})",
                            nvdec_driver::cuvid_error_name(code), code
                        );
                        Self::cuvid_err(code)
                    }
                }
            }

            NvdecCommand::DestroyDecoder { decoder } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                if let Some((_, real_decoder)) = self.decoder_handles.remove(&decoder) {
                    debug!(
                        session_id = session.session_id,
                        "DestroyDecoder {:?}", decoder
                    );
                    let res = d.destroy_decoder(real_decoder);
                    session.remove_handle(&decoder);
                    if res == CUDA_SUCCESS {
                        NvdecResponse::Success
                    } else {
                        Self::cuvid_err(res)
                    }
                } else {
                    NvdecResponse::Error {
                        code: 400, // CUDA_ERROR_INVALID_HANDLE
                        message: "Unknown decoder handle".to_string(),
                    }
                }
            }

            NvdecCommand::DecodePicture {
                decoder,
                pic_params,
                bitstream_data,
            } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                let real_decoder = match self.decoder_handles.get(&decoder) {
                    Some(entry) => *entry,
                    None => {
                        return NvdecResponse::Error {
                            code: 400,
                            message: "Unknown decoder handle".to_string(),
                        };
                    }
                };

                debug!(
                    session_id = session.session_id,
                    "DecodePicture (pic_params={} bytes, bitstream={} bytes)",
                    pic_params.len(), bitstream_data.len()
                );

                // The CUVIDPICPARAMS struct contains a pointer field `pBitstreamData`
                // and a length field `nBitstreamDataLen`. We need to patch these
                // to point at our local bitstream_data buffer.
                //
                // CUVIDPICPARAMS layout (partial, SDK 12.x):
                //   0: PicWidthInMbs (i32)
                //   4: FrameHeightInMbs (i32)
                //   8: CurrPicIdx (i32)
                //  12: field_pic_flag (i32)
                //  16: bottom_field_flag (i32)
                //  20: second_field (i32)
                //  24: nBitstreamDataLen (u32)
                //  28: padding/reserved
                //  32: pBitstreamData (pointer, 8 bytes on 64-bit)
                //  40: nNumSlices (u32)
                //  44: padding
                //  48: pSliceDataOffsets (pointer, 8 bytes on 64-bit)
                //
                // We patch the pointer at offset 32 and length at offset 24.
                let mut params = pic_params;
                if params.len() >= 40 {
                    // Patch nBitstreamDataLen at offset 24
                    let bs_len = bitstream_data.len() as u32;
                    params[24..28].copy_from_slice(&bs_len.to_le_bytes());

                    // Patch pBitstreamData pointer at offset 32
                    let bs_ptr = bitstream_data.as_ptr() as u64;
                    params[32..40].copy_from_slice(&bs_ptr.to_le_bytes());
                }

                let res = d.decode_picture(real_decoder, &mut params);
                if res == CUDA_SUCCESS {
                    NvdecResponse::Success
                } else {
                    Self::cuvid_err(res)
                }
            }

            NvdecCommand::GetDecodeStatus {
                decoder,
                picture_index,
            } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                let real_decoder = match self.decoder_handles.get(&decoder) {
                    Some(entry) => *entry,
                    None => {
                        return NvdecResponse::Error {
                            code: 400,
                            message: "Unknown decoder handle".to_string(),
                        };
                    }
                };

                debug!(
                    session_id = session.session_id,
                    "GetDecodeStatus(pic_idx={})", picture_index
                );

                // CUVIDGETDECODESTATUS is a small struct:
                //   0: decodeStatus (u32 — CuvidDecodeStatus enum)
                //   4: reserved[31] (31 x u32)
                // Total: 128 bytes
                const STATUS_STRUCT_SIZE: usize = 128;
                let mut status_buf = vec![0u8; STATUS_STRUCT_SIZE];
                let res = d.get_decode_status(real_decoder, picture_index, &mut status_buf);
                if res == CUDA_SUCCESS {
                    let decode_status = i32::from_le_bytes([
                        status_buf[0], status_buf[1], status_buf[2], status_buf[3],
                    ]);
                    NvdecResponse::DecodeStatus { decode_status }
                } else {
                    Self::cuvid_err(res)
                }
            }

            NvdecCommand::ReconfigureDecoder {
                decoder,
                reconfig_params,
            } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                let real_decoder = match self.decoder_handles.get(&decoder) {
                    Some(entry) => *entry,
                    None => {
                        return NvdecResponse::Error {
                            code: 400,
                            message: "Unknown decoder handle".to_string(),
                        };
                    }
                };

                debug!(
                    session_id = session.session_id,
                    "ReconfigureDecoder (params={} bytes)", reconfig_params.len()
                );

                let mut params = reconfig_params;
                let res = d.reconfigure_decoder(real_decoder, &mut params);
                if res == CUDA_SUCCESS {
                    NvdecResponse::Success
                } else {
                    Self::cuvid_err(res)
                }
            }

            // ── Frame mapping ────────────────────────────────────────
            NvdecCommand::MapVideoFrame {
                decoder,
                picture_index,
                proc_params,
            } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                let real_decoder = match self.decoder_handles.get(&decoder) {
                    Some(entry) => *entry,
                    None => {
                        return NvdecResponse::Error {
                            code: 400,
                            message: "Unknown decoder handle".to_string(),
                        };
                    }
                };

                debug!(
                    session_id = session.session_id,
                    "MapVideoFrame(pic_idx={}, proc_params={} bytes)",
                    picture_index, proc_params.len()
                );

                let mut params = proc_params;
                match d.map_video_frame(real_decoder, picture_index, &mut params) {
                    Ok((devptr, pitch)) => {
                        // Store the device pointer as a NetworkHandle so the client
                        // (and CUDA interpose) can reference it for UnmapVideoFrame
                        // and for CUDA memory operations on the decoded frame.
                        let handle = session.alloc_handle(ResourceType::CuDevicePtr);
                        self.mapped_frame_handles.insert(handle, devptr);
                        debug!(
                            session_id = session.session_id,
                            "MapVideoFrame -> devptr=0x{:x}, pitch={}, handle={:?}",
                            devptr, pitch, handle
                        );
                        NvdecResponse::VideoFrameMapped {
                            device_ptr: handle,
                            pitch,
                        }
                    }
                    Err(code) => Self::cuvid_err(code),
                }
            }

            NvdecCommand::UnmapVideoFrame {
                decoder,
                mapped_frame,
            } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                let real_decoder = match self.decoder_handles.get(&decoder) {
                    Some(entry) => *entry,
                    None => {
                        return NvdecResponse::Error {
                            code: 400,
                            message: "Unknown decoder handle".to_string(),
                        };
                    }
                };

                let devptr = match self.mapped_frame_handles.remove(&mapped_frame) {
                    Some((_, ptr)) => ptr,
                    None => {
                        return NvdecResponse::Error {
                            code: 400,
                            message: "Unknown mapped frame handle".to_string(),
                        };
                    }
                };

                debug!(
                    session_id = session.session_id,
                    "UnmapVideoFrame(devptr=0x{:x})", devptr
                );

                session.remove_handle(&mapped_frame);
                let res = d.unmap_video_frame(real_decoder, devptr);
                if res == CUDA_SUCCESS {
                    NvdecResponse::Success
                } else {
                    Self::cuvid_err(res)
                }
            }

            // ── Video parser (not supported over network) ────────────
            //
            // The CUVID parser uses C callback function pointers which cannot
            // be forwarded over the network. FFmpeg and most applications do
            // their own parsing and call the decoder-level functions directly.
            // We return NOT_SUPPORTED for parser functions.
            NvdecCommand::CreateVideoParser { .. } => {
                warn!(
                    session_id = session.session_id,
                    "CreateVideoParser: not supported over network (parser uses callbacks)"
                );
                NvdecResponse::Error {
                    code: CUDA_ERROR_NOT_SUPPORTED,
                    message: "CUVID video parser is not supported over network (uses C callbacks). \
                              Use decoder-level functions (CreateDecoder/DecodePicture/MapVideoFrame) instead."
                        .to_string(),
                }
            }

            NvdecCommand::ParseVideoData { .. } => {
                NvdecResponse::Error {
                    code: CUDA_ERROR_NOT_SUPPORTED,
                    message: "CUVID ParseVideoData is not supported over network".to_string(),
                }
            }

            NvdecCommand::DestroyVideoParser { parser } => {
                // If somehow a parser handle was tracked, clean it up
                if self.parser_handles.remove(&parser).is_some() {
                    session.remove_handle(&parser);
                }
                NvdecResponse::Error {
                    code: CUDA_ERROR_NOT_SUPPORTED,
                    message: "CUVID video parser is not supported over network".to_string(),
                }
            }

            // ── Context locking ──────────────────────────────────────
            NvdecCommand::CtxLockCreate { cuda_context } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                debug!(
                    session_id = session.session_id,
                    "CtxLockCreate(cuda_context={:?})", cuda_context
                );

                // The cuda_context NetworkHandle needs to be resolved to a real
                // CUcontext pointer. For now, we use a null context which makes
                // CUVID use the current CUDA context on the thread.
                // In a full implementation, this would look up the real context
                // from the CudaExecutor's context_handles map.
                let real_ctx: nvdec_driver::CUcontext = std::ptr::null_mut();

                match d.ctx_lock_create(real_ctx) {
                    Ok(lock) => {
                        let handle = session.alloc_handle(ResourceType::CuVideoCtxLock);
                        self.ctx_lock_handles.insert(handle, lock);
                        info!(
                            session_id = session.session_id,
                            "NVDEC ctx lock created: {:?}", handle
                        );
                        NvdecResponse::CtxLockCreated { handle }
                    }
                    Err(code) => Self::cuvid_err(code),
                }
            }

            NvdecCommand::CtxLockDestroy { lock } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                if let Some((_, real_lock)) = self.ctx_lock_handles.remove(&lock) {
                    debug!(
                        session_id = session.session_id,
                        "CtxLockDestroy {:?}", lock
                    );
                    let res = d.ctx_lock_destroy(real_lock);
                    session.remove_handle(&lock);
                    if res == CUDA_SUCCESS {
                        NvdecResponse::Success
                    } else {
                        Self::cuvid_err(res)
                    }
                } else {
                    NvdecResponse::Error {
                        code: 400,
                        message: "Unknown ctx lock handle".to_string(),
                    }
                }
            }
        }
    }

    /// Clean up all NVDEC resources belonging to a disconnected session.
    pub fn cleanup_session(&self, session: &Session) {
        let handles = session.all_handles();
        if handles.is_empty() {
            return;
        }

        let driver = match &self.driver {
            Some(d) => d,
            None => return,
        };

        let mut cleaned = 0u32;

        // Pass 1: Unmap video frames
        for h in handles.iter().filter(|h| h.resource_type == ResourceType::CuDevicePtr) {
            if let Some((_, devptr)) = self.mapped_frame_handles.remove(h) {
                // We cannot unmap without the decoder handle, but at cleanup time
                // the decoder is about to be destroyed which implicitly unmaps.
                // Just remove from tracking.
                let _ = devptr;
                cleaned += 1;
            }
        }

        // Pass 2: Context locks
        for h in handles.iter().filter(|h| h.resource_type == ResourceType::CuVideoCtxLock) {
            if let Some((_, lock)) = self.ctx_lock_handles.remove(h) {
                driver.ctx_lock_destroy(lock);
                cleaned += 1;
            }
        }

        // Pass 3: Video parsers (should not have any, but clean up defensively)
        for h in handles.iter().filter(|h| h.resource_type == ResourceType::CuVideoParser) {
            if let Some((_, parser)) = self.parser_handles.remove(h) {
                driver.destroy_video_parser(parser);
                cleaned += 1;
            }
        }

        // Pass 4: Decoders (destroy last, as they own mapped frames)
        for h in handles.iter().filter(|h| h.resource_type == ResourceType::CuVideoDecoder) {
            if let Some((_, decoder)) = self.decoder_handles.remove(h) {
                driver.destroy_decoder(decoder);
                cleaned += 1;
            }
        }

        if cleaned > 0 {
            info!(
                session_id = session.session_id,
                "NVDEC cleanup: destroyed {} resources", cleaned
            );
        }

        // Report any leaked handles that we could not clean up
        let leaked: Vec<_> = handles
            .iter()
            .filter(|h| matches!(
                h.resource_type,
                ResourceType::CuVideoDecoder
                | ResourceType::CuVideoParser
                | ResourceType::CuVideoCtxLock
            ))
            .filter(|h| {
                self.decoder_handles.contains_key(h)
                    || self.parser_handles.contains_key(h)
                    || self.ctx_lock_handles.contains_key(h)
            })
            .collect();

        if !leaked.is_empty() {
            warn!(
                session_id = session.session_id,
                "NVDEC cleanup: {} handle(s) could not be cleaned up", leaked.len()
            );
        }
    }
}
