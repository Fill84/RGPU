//! Server-side NVENC command executor.
//!
//! Executes NVENC (NVIDIA Video Encoder) API commands on real GPU hardware
//! via the dynamically loaded NVENC library. Follows the same pattern as
//! `cuda_executor.rs` — matches on `NvencCommand` variants and dispatches
//! to the `NvencDriver`.

use std::ffi::c_void;
use std::sync::Arc;

use dashmap::DashMap;
use tracing::{debug, error, info, warn};

use rgpu_protocol::handle::{NetworkHandle, ResourceType};
use rgpu_protocol::nvenc_commands::{NvGuid, NvencCommand, NvencResponse};

use crate::cuda_executor::CudaExecutor;
use crate::nvenc_driver::{self, NvencDriver, NV_ENC_SUCCESS};
use crate::session::Session;

/// Server-side NVENC command executor.
/// Executes NVENC API commands on real GPU hardware via the dynamically loaded NVENC library.
pub struct NvencExecutor {
    /// The real NVENC driver (loaded via libloading)
    driver: Option<Arc<NvencDriver>>,
    /// Reference to CudaExecutor for resolving CUDA context handles
    cuda_executor: Arc<CudaExecutor>,
    /// Maps NetworkHandle -> real encoder session pointer
    encoder_handles: DashMap<NetworkHandle, *mut c_void>,
    /// Maps encoder NetworkHandle -> the CUDA context used to create the session.
    /// NVENC requires the CUDA context to be current for ALL API calls on an encoder.
    encoder_contexts: DashMap<NetworkHandle, *mut c_void>,
    /// Maps NetworkHandle -> real input buffer pointer
    input_buffer_handles: DashMap<NetworkHandle, *mut c_void>,
    /// Maps NetworkHandle -> real bitstream buffer pointer
    bitstream_buffer_handles: DashMap<NetworkHandle, *mut c_void>,
    /// Maps NetworkHandle -> real registered resource pointer
    registered_resource_handles: DashMap<NetworkHandle, *mut c_void>,
    /// Maps NetworkHandle -> real mapped resource pointer
    mapped_resource_handles: DashMap<NetworkHandle, *mut c_void>,
    /// Maps input buffer NetworkHandle -> locked data pointer (from LockInputBuffer).
    /// Stored so UnlockInputBuffer can copy data without re-locking.
    locked_input_ptrs: DashMap<NetworkHandle, *mut c_void>,
}

// SAFETY: NVENC driver pointers are valid across threads when used with proper
// encoder session management. Each session is bound to a CUDA context.
unsafe impl Send for NvencExecutor {}
unsafe impl Sync for NvencExecutor {}

impl NvencExecutor {
    pub fn new(cuda_executor: Arc<CudaExecutor>) -> Self {
        // Try to load the NVENC driver
        let driver = match NvencDriver::load() {
            Ok(d) => {
                info!(
                    "NVENC driver loaded (max API version {}.{})",
                    d.get_max_supported_version() >> 4,
                    d.get_max_supported_version() & 0xF
                );
                Some(d)
            }
            Err(e) => {
                warn!("NVENC driver not available: {} - encoding disabled", e);
                None
            }
        };

        Self {
            driver,
            cuda_executor,
            encoder_handles: DashMap::new(),
            encoder_contexts: DashMap::new(),
            input_buffer_handles: DashMap::new(),
            bitstream_buffer_handles: DashMap::new(),
            registered_resource_handles: DashMap::new(),
            mapped_resource_handles: DashMap::new(),
            locked_input_ptrs: DashMap::new(),
        }
    }

    /// Check if the NVENC driver is available.
    fn driver(&self) -> Result<&NvencDriver, NvencResponse> {
        self.driver.as_deref().ok_or(NvencResponse::Error {
            code: nvenc_driver::NV_ENC_ERR_NO_ENCODE_DEVICE,
            message: "NVENC driver not loaded on server".to_string(),
        })
    }

    /// Convert an NvencStatus to an NvencResponse::Error.
    fn nvenc_err(code: nvenc_driver::NvencStatus) -> NvencResponse {
        NvencResponse::Error {
            code,
            message: nvenc_driver::nvenc_error_name(code).to_string(),
        }
    }

    /// Resolve a CUDA context NetworkHandle to a real CUcontext pointer.
    fn resolve_cuda_context(&self, handle: &NetworkHandle) -> Option<*mut c_void> {
        self.cuda_executor.get_context_ptr(handle)
    }

    /// Resolve an encoder NetworkHandle to a real encoder pointer.
    fn resolve_encoder(&self, handle: &NetworkHandle) -> Result<*mut c_void, NvencResponse> {
        self.encoder_handles.get(handle).map(|e| *e).ok_or(NvencResponse::Error {
            code: nvenc_driver::NV_ENC_ERR_INVALID_ENCODERDEVICE,
            message: "invalid encoder handle".to_string(),
        })
    }

    /// Resolve an input buffer NetworkHandle to a real buffer pointer.
    fn resolve_input_buffer(&self, handle: &NetworkHandle) -> Result<*mut c_void, NvencResponse> {
        self.input_buffer_handles
            .get(handle)
            .map(|b| *b)
            .ok_or(NvencResponse::Error {
                code: nvenc_driver::NV_ENC_ERR_INVALID_PARAM,
                message: "invalid input buffer handle".to_string(),
            })
    }

    /// Resolve a bitstream buffer NetworkHandle to a real buffer pointer.
    fn resolve_bitstream_buffer(
        &self,
        handle: &NetworkHandle,
    ) -> Result<*mut c_void, NvencResponse> {
        let result = self.bitstream_buffer_handles
            .get(handle)
            .map(|b| *b);
        if result.is_none() {
            warn!("resolve_bitstream_buffer FAILED for {:?}, map has {} entries",
                handle,
                self.bitstream_buffer_handles.len(),
            );
        }
        result.ok_or(NvencResponse::Error {
                code: nvenc_driver::NV_ENC_ERR_INVALID_PARAM,
                message: "invalid bitstream buffer handle".to_string(),
            })
    }

    /// Resolve a registered resource NetworkHandle to a real resource pointer.
    fn resolve_registered_resource(
        &self,
        handle: &NetworkHandle,
    ) -> Result<*mut c_void, NvencResponse> {
        self.registered_resource_handles
            .get(handle)
            .map(|r| *r)
            .ok_or(NvencResponse::Error {
                code: nvenc_driver::NV_ENC_ERR_RESOURCE_NOT_REGISTERED,
                message: "invalid registered resource handle".to_string(),
            })
    }

    /// Resolve a mapped resource NetworkHandle to a real resource pointer.
    fn resolve_mapped_resource(
        &self,
        handle: &NetworkHandle,
    ) -> Result<*mut c_void, NvencResponse> {
        self.mapped_resource_handles
            .get(handle)
            .map(|r| *r)
            .ok_or(NvencResponse::Error {
                code: nvenc_driver::NV_ENC_ERR_RESOURCE_NOT_MAPPED,
                message: "invalid mapped resource handle".to_string(),
            })
    }

    /// Ensure the CUDA context associated with an encoder is current on this thread.
    /// NVENC requires the creating CUDA context to be current for ALL API calls.
    /// Due to tokio's thread migration, we must re-establish the context before each call.
    fn ensure_encoder_context(&self, encoder_handle: &NetworkHandle) {
        if let Some(ctx) = self.encoder_contexts.get(encoder_handle) {
            if let Some(cuda_driver) = self.cuda_executor.driver_ref() {
                let _ = cuda_driver.ctx_set_current(*ctx as crate::cuda_driver::CUcontext);
            }
        }
    }

    /// Execute an NVENC command and return the response.
    pub fn execute(&self, session: &Session, cmd: NvencCommand) -> NvencResponse {
        match cmd {
            // ── Version query ────────────────────────────────────────
            NvencCommand::GetMaxSupportedVersion => {
                match self.driver() {
                    Ok(d) => NvencResponse::MaxSupportedVersion {
                        version: d.get_max_supported_version(),
                    },
                    Err(e) => e,
                }
            }

            // ── Session management ───────────────────────────────────
            NvencCommand::OpenEncodeSession {
                cuda_context,
                device_type,
            } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };

                // Resolve the CUDA context handle to a real CUcontext pointer
                let real_ctx = match self.resolve_cuda_context(&cuda_context) {
                    Some(ctx) => ctx,
                    None => {
                        return NvencResponse::Error {
                            code: nvenc_driver::NV_ENC_ERR_INVALID_DEVICE,
                            message: format!(
                                "could not resolve CUDA context handle {:?}",
                                cuda_context
                            ),
                        };
                    }
                };

                // Ensure the CUDA context is current on this thread before NVENC call
                if let Some(cuda_driver) = self.cuda_executor.driver_ref() {
                    let _ = cuda_driver.ctx_set_current(real_ctx as crate::cuda_driver::CUcontext);
                }

                match d.open_encode_session_ex(real_ctx, device_type) {
                    Ok(encoder) => {
                        let handle = session.alloc_handle(ResourceType::NvEncSession);
                        self.encoder_handles.insert(handle, encoder);
                        // Track the CUDA context for this encoder session — we need
                        // to re-establish it before every subsequent NVENC call.
                        self.encoder_contexts.insert(handle, real_ctx);
                        debug!(
                            session_id = session.session_id,
                            "NvEnc OpenEncodeSession -> {:?}", handle
                        );
                        NvencResponse::EncoderOpened { handle }
                    }
                    Err(e) => {
                        error!(
                            session_id = session.session_id,
                            "NvEnc OpenEncodeSession failed: {} ({})",
                            nvenc_driver::nvenc_error_name(e),
                            e
                        );
                        Self::nvenc_err(e)
                    }
                }
            }

            NvencCommand::DestroyEncoder { encoder } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                self.ensure_encoder_context(&encoder);
                let real_encoder = match self.encoder_handles.remove(&encoder) {
                    Some((_, e)) => e,
                    None => {
                        return NvencResponse::Error {
                            code: nvenc_driver::NV_ENC_ERR_INVALID_ENCODERDEVICE,
                            message: "invalid encoder handle".to_string(),
                        };
                    }
                };

                let res = d.destroy_encoder(real_encoder);
                self.encoder_contexts.remove(&encoder);
                session.remove_handle(&encoder);
                debug!(
                    session_id = session.session_id,
                    "NvEnc DestroyEncoder {:?} -> {}", encoder, res
                );
                if res == NV_ENC_SUCCESS {
                    NvencResponse::Success
                } else {
                    Self::nvenc_err(res)
                }
            }

            // ── Capability queries ───────────────────────────────────
            NvencCommand::GetEncodeGUIDCount { encoder } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_encoder = match self.resolve_encoder(&encoder) {
                    Ok(e) => e,
                    Err(e) => return e,
                };
                self.ensure_encoder_context(&encoder);

                match d.get_encode_guid_count(real_encoder) {
                    Ok(count) => {
                        debug!(
                            session_id = session.session_id,
                            "NvEnc GetEncodeGUIDCount -> {}", count
                        );
                        NvencResponse::GUIDCount(count)
                    }
                    Err(e) => Self::nvenc_err(e),
                }
            }

            NvencCommand::GetEncodeGUIDs { encoder } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_encoder = match self.resolve_encoder(&encoder) {
                    Ok(e) => e,
                    Err(e) => return e,
                };
                self.ensure_encoder_context(&encoder);

                // First get the count, then get the GUIDs
                let count = match d.get_encode_guid_count(real_encoder) {
                    Ok(c) => c,
                    Err(e) => return Self::nvenc_err(e),
                };

                match d.get_encode_guids(real_encoder, count) {
                    Ok(guids) => {
                        let nv_guids: Vec<NvGuid> =
                            guids.into_iter().map(NvGuid).collect();
                        debug!(
                            session_id = session.session_id,
                            "NvEnc GetEncodeGUIDs -> {} guid(s)", nv_guids.len()
                        );
                        NvencResponse::GUIDs(nv_guids)
                    }
                    Err(e) => Self::nvenc_err(e),
                }
            }

            NvencCommand::GetEncodePresetCount {
                encoder,
                encode_guid,
            } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_encoder = match self.resolve_encoder(&encoder) {
                    Ok(e) => e,
                    Err(e) => return e,
                };
                self.ensure_encoder_context(&encoder);

                match d.get_encode_preset_count(real_encoder, encode_guid.0) {
                    Ok(count) => {
                        debug!(
                            session_id = session.session_id,
                            "NvEnc GetEncodePresetCount -> {}", count
                        );
                        NvencResponse::GUIDCount(count)
                    }
                    Err(e) => Self::nvenc_err(e),
                }
            }

            NvencCommand::GetEncodePresetGUIDs {
                encoder,
                encode_guid,
            } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_encoder = match self.resolve_encoder(&encoder) {
                    Ok(e) => e,
                    Err(e) => return e,
                };
                self.ensure_encoder_context(&encoder);

                // First get count
                let count = match d.get_encode_preset_count(real_encoder, encode_guid.0) {
                    Ok(c) => c,
                    Err(e) => return Self::nvenc_err(e),
                };

                match d.get_encode_preset_guids(real_encoder, encode_guid.0, count) {
                    Ok(presets) => {
                        let nv_guids: Vec<NvGuid> =
                            presets.into_iter().map(NvGuid).collect();
                        debug!(
                            session_id = session.session_id,
                            "NvEnc GetEncodePresetGUIDs -> {} preset(s)", nv_guids.len()
                        );
                        NvencResponse::GUIDs(nv_guids)
                    }
                    Err(e) => Self::nvenc_err(e),
                }
            }

            NvencCommand::GetEncodePresetConfig {
                encoder,
                encode_guid,
                preset_guid,
            } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_encoder = match self.resolve_encoder(&encoder) {
                    Ok(e) => e,
                    Err(e) => return e,
                };
                self.ensure_encoder_context(&encoder);

                match d.get_encode_preset_config(real_encoder, encode_guid.0, preset_guid.0)
                {
                    Ok(config) => {
                        debug!(
                            session_id = session.session_id,
                            "NvEnc GetEncodePresetConfig -> {} bytes", config.len()
                        );
                        NvencResponse::PresetConfig(config)
                    }
                    Err(e) => Self::nvenc_err(e),
                }
            }

            NvencCommand::GetEncodePresetConfigEx {
                encoder,
                encode_guid,
                preset_guid,
                tuning_info,
            } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_encoder = match self.resolve_encoder(&encoder) {
                    Ok(e) => e,
                    Err(e) => return e,
                };
                self.ensure_encoder_context(&encoder);

                match d.get_encode_preset_config_ex(real_encoder, encode_guid.0, preset_guid.0, tuning_info)
                {
                    Ok(config) => {
                        debug!(
                            session_id = session.session_id,
                            "NvEnc GetEncodePresetConfigEx tuning={} -> {} bytes", tuning_info, config.len()
                        );
                        NvencResponse::PresetConfig(config)
                    }
                    Err(e) => Self::nvenc_err(e),
                }
            }

            NvencCommand::GetEncodeCaps {
                encoder,
                encode_guid,
                caps_param,
            } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_encoder = match self.resolve_encoder(&encoder) {
                    Ok(e) => e,
                    Err(e) => return e,
                };
                self.ensure_encoder_context(&encoder);

                match d.get_encode_caps(real_encoder, encode_guid.0, caps_param as u32) {
                    Ok(val) => {
                        debug!(
                            session_id = session.session_id,
                            "NvEnc GetEncodeCaps({}) -> {}", caps_param, val
                        );
                        NvencResponse::CapsValue(val)
                    }
                    Err(e) => {
                        warn!(
                            session_id = session.session_id,
                            "NvEnc GetEncodeCaps({}) -> ERROR {}", caps_param, e
                        );
                        Self::nvenc_err(e)
                    }
                }
            }

            NvencCommand::GetInputFormatCount {
                encoder,
                encode_guid,
            } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_encoder = match self.resolve_encoder(&encoder) {
                    Ok(e) => e,
                    Err(e) => return e,
                };
                self.ensure_encoder_context(&encoder);

                match d.get_input_format_count(real_encoder, encode_guid.0) {
                    Ok(count) => {
                        debug!(
                            session_id = session.session_id,
                            "NvEnc GetInputFormatCount -> {}", count
                        );
                        NvencResponse::InputFormatCount(count)
                    }
                    Err(e) => Self::nvenc_err(e),
                }
            }

            NvencCommand::GetInputFormats {
                encoder,
                encode_guid,
            } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_encoder = match self.resolve_encoder(&encoder) {
                    Ok(e) => e,
                    Err(e) => return e,
                };
                self.ensure_encoder_context(&encoder);

                // First get count
                let count = match d.get_input_format_count(real_encoder, encode_guid.0) {
                    Ok(c) => c,
                    Err(e) => return Self::nvenc_err(e),
                };

                match d.get_input_formats(real_encoder, encode_guid.0, count) {
                    Ok(formats) => {
                        debug!(
                            session_id = session.session_id,
                            "NvEnc GetInputFormats -> {} format(s)", formats.len()
                        );
                        NvencResponse::InputFormats(formats)
                    }
                    Err(e) => Self::nvenc_err(e),
                }
            }

            // ── Encoder initialization ───────────────────────────────
            NvencCommand::InitializeEncoder {
                encoder,
                mut params,
                encode_config,
            } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_encoder = match self.resolve_encoder(&encoder) {
                    Ok(e) => e,
                    Err(e) => return e,
                };
                self.ensure_encoder_context(&encoder);

                // ── Patch API version fields ──────────────────────────
                // The client may be compiled against a different SDK version (e.g. 12.2)
                // than the server's driver (e.g. 13.0). NVENC rejects structs whose API
                // version component doesn't match the session's negotiated version.
                // Patch all version fields to use the server driver's API version.

                // NV_ENC_INITIALIZE_PARAMS.version at offset 0
                d.patch_struct_version(&mut params, 0);

                // If the client sent NV_ENC_CONFIG data, patch the encodeConfig pointer
                // in NV_ENC_INITIALIZE_PARAMS to point to a local buffer.
                // On 64-bit, NV_ENC_INITIALIZE_PARAMS.encodeConfig is a pointer at offset 88.
                const ENCODE_CONFIG_PTR_OFFSET: usize = 88;

                // Pad config to a safe size so the real driver doesn't read past our buffer.
                // NV_ENC_CONFIG is ~4600 bytes on 64-bit; use 8192 for safety.
                const SAFE_CONFIG_SIZE: usize = 8192;

                let mut config_buf: Option<Vec<u8>> = encode_config.map(|mut data| {
                    // Patch NV_ENC_CONFIG.version at offset 0
                    d.patch_struct_version(&mut data, 0);
                    // Patch NV_ENC_RC_PARAMS.version at offset 40 (within NV_ENC_CONFIG)
                    d.patch_struct_version(&mut data, 40);

                    // Pad to safe size (zeros for reserved fields)
                    if data.len() < SAFE_CONFIG_SIZE {
                        data.resize(SAFE_CONFIG_SIZE, 0);
                    }
                    data
                });

                if let Some(ref mut config_data) = config_buf {
                    let config_ptr = config_data.as_mut_ptr() as usize;
                    if params.len() > ENCODE_CONFIG_PTR_OFFSET + 8 {
                        params[ENCODE_CONFIG_PTR_OFFSET..ENCODE_CONFIG_PTR_OFFSET + 8]
                            .copy_from_slice(&config_ptr.to_le_bytes());
                        debug!(
                            session_id = session.session_id,
                            "NvEnc InitializeEncoder: patched encodeConfig ptr to {:p}, config_len={}",
                            config_data.as_ptr(), config_data.len()
                        );
                    }
                }

                let res = d.initialize_encoder(real_encoder, &mut params);
                if res == NV_ENC_SUCCESS {
                    info!(
                        session_id = session.session_id,
                        "NvEnc InitializeEncoder -> SUCCESS"
                    );
                    NvencResponse::Success
                } else {
                    let err_str = d.get_last_error_string(real_encoder)
                        .unwrap_or_default();
                    error!(
                        session_id = session.session_id,
                        "NvEnc InitializeEncoder -> {} ({}): {}",
                        nvenc_driver::nvenc_error_name(res), res, err_str
                    );
                    Self::nvenc_err(res)
                }
            }

            NvencCommand::ReconfigureEncoder {
                encoder,
                mut params,
            } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_encoder = match self.resolve_encoder(&encoder) {
                    Ok(e) => e,
                    Err(e) => return e,
                };
                self.ensure_encoder_context(&encoder);

                // Patch NV_ENC_RECONFIGURE_PARAMS.version at offset 0
                d.patch_struct_version(&mut params, 0);

                let res = d.reconfigure_encoder(real_encoder, &mut params);
                debug!(
                    session_id = session.session_id,
                    "NvEnc ReconfigureEncoder -> {}", res
                );
                if res == NV_ENC_SUCCESS {
                    NvencResponse::Success
                } else {
                    Self::nvenc_err(res)
                }
            }

            // ── Input buffer management ──────────────────────────────
            NvencCommand::CreateInputBuffer {
                encoder,
                width,
                height,
                buffer_fmt,
            } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_encoder = match self.resolve_encoder(&encoder) {
                    Ok(e) => e,
                    Err(e) => return e,
                };
                self.ensure_encoder_context(&encoder);

                match d.create_input_buffer(real_encoder, width, height, buffer_fmt) {
                    Ok(buffer) => {
                        let handle = session.alloc_handle(ResourceType::NvEncInputBuffer);
                        self.input_buffer_handles.insert(handle, buffer);
                        debug!(
                            session_id = session.session_id,
                            "NvEnc CreateInputBuffer {}x{} -> {:?}", width, height, handle
                        );
                        NvencResponse::InputBufferCreated { handle }
                    }
                    Err(e) => Self::nvenc_err(e),
                }
            }

            NvencCommand::DestroyInputBuffer {
                encoder,
                input_buffer,
            } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_encoder = match self.resolve_encoder(&encoder) {
                    Ok(e) => e,
                    Err(e) => return e,
                };
                self.ensure_encoder_context(&encoder);
                let real_buffer = match self.input_buffer_handles.remove(&input_buffer) {
                    Some((_, b)) => b,
                    None => {
                        return NvencResponse::Error {
                            code: nvenc_driver::NV_ENC_ERR_INVALID_PARAM,
                            message: "invalid input buffer handle".to_string(),
                        };
                    }
                };

                let res = d.destroy_input_buffer(real_encoder, real_buffer);
                session.remove_handle(&input_buffer);
                debug!(
                    session_id = session.session_id,
                    "NvEnc DestroyInputBuffer {:?} -> {}", input_buffer, res
                );
                if res == NV_ENC_SUCCESS {
                    NvencResponse::Success
                } else {
                    Self::nvenc_err(res)
                }
            }

            NvencCommand::LockInputBuffer {
                encoder,
                input_buffer,
            } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_encoder = match self.resolve_encoder(&encoder) {
                    Ok(e) => e,
                    Err(e) => return e,
                };
                self.ensure_encoder_context(&encoder);
                let real_buffer = match self.resolve_input_buffer(&input_buffer) {
                    Ok(b) => b,
                    Err(e) => return e,
                };

                match d.lock_input_buffer(real_encoder, real_buffer) {
                    Ok((data_ptr, pitch)) => {
                        // Store the data pointer so UnlockInputBuffer can copy to it
                        // without having to re-lock (which would fail since it's already locked).
                        self.locked_input_ptrs.insert(input_buffer, data_ptr);
                        debug!(
                            session_id = session.session_id,
                            "NvEnc LockInputBuffer {:?} -> pitch={} ptr={:?}", input_buffer, pitch, data_ptr
                        );
                        NvencResponse::InputBufferLocked {
                            pitch,
                            buffer_size: 0, // Client calculates from pitch + tracked height
                        }
                    }
                    Err(e) => Self::nvenc_err(e),
                }
            }

            NvencCommand::UnlockInputBuffer {
                encoder,
                input_buffer,
                data,
                pitch: _,
            } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_encoder = match self.resolve_encoder(&encoder) {
                    Ok(e) => e,
                    Err(e) => return e,
                };
                self.ensure_encoder_context(&encoder);
                let real_buffer = match self.resolve_input_buffer(&input_buffer) {
                    Ok(b) => b,
                    Err(e) => return e,
                };

                // The buffer was locked during LockInputBuffer — use the stored data pointer.
                if !data.is_empty() {
                    if let Some((_, data_ptr)) = self.locked_input_ptrs.remove(&input_buffer) {
                        if !data_ptr.is_null() {
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    data.as_ptr(),
                                    data_ptr as *mut u8,
                                    data.len(),
                                );
                            }
                        }
                    } else {
                        warn!(
                            session_id = session.session_id,
                            "NvEnc UnlockInputBuffer: no stored ptr for {:?}, data will be lost", input_buffer
                        );
                    }
                } else {
                    // No data sent, just remove stored ptr if any
                    self.locked_input_ptrs.remove(&input_buffer);
                }

                let res = d.unlock_input_buffer(real_encoder, real_buffer);
                debug!(
                    session_id = session.session_id,
                    "NvEnc UnlockInputBuffer {:?} ({} bytes data) -> {}",
                    input_buffer, data.len(), res
                );
                if res == NV_ENC_SUCCESS {
                    NvencResponse::Success
                } else {
                    Self::nvenc_err(res)
                }
            }

            // ── Bitstream buffer management ──────────────────────────
            NvencCommand::CreateBitstreamBuffer { encoder } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_encoder = match self.resolve_encoder(&encoder) {
                    Ok(e) => e,
                    Err(e) => return e,
                };
                self.ensure_encoder_context(&encoder);

                match d.create_bitstream_buffer(real_encoder) {
                    Ok(buffer) => {
                        let handle =
                            session.alloc_handle(ResourceType::NvEncBitstreamBuffer);
                        self.bitstream_buffer_handles.insert(handle, buffer);
                        debug!(
                            session_id = session.session_id,
                            "NvEnc CreateBitstreamBuffer -> {:?} (real={:?})",
                            handle, buffer
                        );
                        NvencResponse::BitstreamBufferCreated { handle }
                    }
                    Err(e) => Self::nvenc_err(e),
                }
            }

            NvencCommand::DestroyBitstreamBuffer {
                encoder,
                bitstream_buffer,
            } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_encoder = match self.resolve_encoder(&encoder) {
                    Ok(e) => e,
                    Err(e) => return e,
                };
                self.ensure_encoder_context(&encoder);
                let real_buffer =
                    match self.bitstream_buffer_handles.remove(&bitstream_buffer) {
                        Some((_, b)) => b,
                        None => {
                            return NvencResponse::Error {
                                code: nvenc_driver::NV_ENC_ERR_INVALID_PARAM,
                                message: "invalid bitstream buffer handle".to_string(),
                            };
                        }
                    };

                let res = d.destroy_bitstream_buffer(real_encoder, real_buffer);
                session.remove_handle(&bitstream_buffer);
                debug!(
                    session_id = session.session_id,
                    "NvEnc DestroyBitstreamBuffer {:?} -> {}", bitstream_buffer, res
                );
                if res == NV_ENC_SUCCESS {
                    NvencResponse::Success
                } else {
                    Self::nvenc_err(res)
                }
            }

            NvencCommand::LockBitstream {
                encoder,
                bitstream_buffer,
            } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_encoder = match self.resolve_encoder(&encoder) {
                    Ok(e) => e,
                    Err(e) => return e,
                };
                self.ensure_encoder_context(&encoder);
                let real_buffer = match self.resolve_bitstream_buffer(&bitstream_buffer) {
                    Ok(b) => b,
                    Err(e) => return e,
                };

                match d.lock_bitstream(real_encoder, real_buffer) {
                    Ok(result) => {
                        debug!(
                            session_id = session.session_id,
                            "NvEnc LockBitstream {:?} -> {} bytes, type={}, frame={}",
                            bitstream_buffer,
                            result.data.len(),
                            result.picture_type,
                            result.frame_idx
                        );
                        // Unlock the bitstream now that we've copied the data
                        let unlock_res = d.unlock_bitstream(real_encoder, real_buffer);
                        if unlock_res != NV_ENC_SUCCESS {
                            warn!(
                                session_id = session.session_id,
                                "NvEnc UnlockBitstream after read failed: {}",
                                nvenc_driver::nvenc_error_name(unlock_res)
                            );
                        }
                        NvencResponse::BitstreamData {
                            data: result.data,
                            picture_type: result.picture_type,
                            frame_idx: result.frame_idx,
                            output_timestamp: result.output_timestamp,
                        }
                    }
                    Err(e) => Self::nvenc_err(e),
                }
            }

            NvencCommand::UnlockBitstream {
                encoder,
                bitstream_buffer,
            } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_encoder = match self.resolve_encoder(&encoder) {
                    Ok(e) => e,
                    Err(e) => return e,
                };
                self.ensure_encoder_context(&encoder);
                let real_buffer = match self.resolve_bitstream_buffer(&bitstream_buffer) {
                    Ok(b) => b,
                    Err(e) => return e,
                };

                let res = d.unlock_bitstream(real_encoder, real_buffer);
                debug!(
                    session_id = session.session_id,
                    "NvEnc UnlockBitstream {:?} -> {}", bitstream_buffer, res
                );
                if res == NV_ENC_SUCCESS {
                    NvencResponse::Success
                } else {
                    Self::nvenc_err(res)
                }
            }

            // ── Resource registration ────────────────────────────────
            NvencCommand::RegisterResource {
                encoder,
                resource_type,
                resource,
                width,
                height,
                pitch,
                buffer_fmt,
            } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_encoder = match self.resolve_encoder(&encoder) {
                    Ok(e) => e,
                    Err(e) => return e,
                };
                self.ensure_encoder_context(&encoder);

                // Resolve the resource handle. For CUDA resources (type 1),
                // this would be a device pointer from our CUDA executor.
                let real_resource = if resource_type == 1 {
                    // CUDA device pointer
                    match self.cuda_executor.get_device_ptr(&resource) {
                        Some(dptr) => dptr as *mut c_void,
                        None => {
                            return NvencResponse::Error {
                                code: nvenc_driver::NV_ENC_ERR_INVALID_PARAM,
                                message: "could not resolve CUDA device pointer for resource"
                                    .to_string(),
                            };
                        }
                    }
                } else {
                    // For other resource types, we'd need additional resolution.
                    // For now, treat the resource_id as a raw pointer (not ideal, but
                    // handles the case where the client passes an already-resolved ptr).
                    resource.resource_id as *mut c_void
                };

                match d.register_resource(
                    real_encoder,
                    resource_type,
                    real_resource,
                    width,
                    height,
                    pitch,
                    buffer_fmt,
                ) {
                    Ok(registered) => {
                        let handle =
                            session.alloc_handle(ResourceType::NvEncRegisteredResource);
                        self.registered_resource_handles.insert(handle, registered);
                        debug!(
                            session_id = session.session_id,
                            "NvEnc RegisterResource -> {:?}", handle
                        );
                        NvencResponse::ResourceRegistered { handle }
                    }
                    Err(e) => Self::nvenc_err(e),
                }
            }

            NvencCommand::UnregisterResource {
                encoder,
                registered_resource,
            } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_encoder = match self.resolve_encoder(&encoder) {
                    Ok(e) => e,
                    Err(e) => return e,
                };
                self.ensure_encoder_context(&encoder);
                let real_resource =
                    match self.registered_resource_handles.remove(&registered_resource) {
                        Some((_, r)) => r,
                        None => {
                            return NvencResponse::Error {
                                code: nvenc_driver::NV_ENC_ERR_RESOURCE_NOT_REGISTERED,
                                message: "invalid registered resource handle".to_string(),
                            };
                        }
                    };

                let res = d.unregister_resource(real_encoder, real_resource);
                session.remove_handle(&registered_resource);
                debug!(
                    session_id = session.session_id,
                    "NvEnc UnregisterResource {:?} -> {}", registered_resource, res
                );
                if res == NV_ENC_SUCCESS {
                    NvencResponse::Success
                } else {
                    Self::nvenc_err(res)
                }
            }

            NvencCommand::MapInputResource {
                encoder,
                registered_resource,
            } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_encoder = match self.resolve_encoder(&encoder) {
                    Ok(e) => e,
                    Err(e) => return e,
                };
                self.ensure_encoder_context(&encoder);
                let real_resource =
                    match self.resolve_registered_resource(&registered_resource) {
                        Ok(r) => r,
                        Err(e) => return e,
                    };

                match d.map_input_resource(real_encoder, real_resource) {
                    Ok((mapped, _fmt)) => {
                        let handle =
                            session.alloc_handle(ResourceType::NvEncMappedResource);
                        self.mapped_resource_handles.insert(handle, mapped);
                        debug!(
                            session_id = session.session_id,
                            "NvEnc MapInputResource -> {:?}", handle
                        );
                        NvencResponse::ResourceMapped { handle }
                    }
                    Err(e) => Self::nvenc_err(e),
                }
            }

            NvencCommand::UnmapInputResource {
                encoder,
                mapped_resource,
            } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_encoder = match self.resolve_encoder(&encoder) {
                    Ok(e) => e,
                    Err(e) => return e,
                };
                self.ensure_encoder_context(&encoder);
                let real_resource =
                    match self.mapped_resource_handles.remove(&mapped_resource) {
                        Some((_, r)) => r,
                        None => {
                            return NvencResponse::Error {
                                code: nvenc_driver::NV_ENC_ERR_RESOURCE_NOT_MAPPED,
                                message: "invalid mapped resource handle".to_string(),
                            };
                        }
                    };

                let res = d.unmap_input_resource(real_encoder, real_resource);
                session.remove_handle(&mapped_resource);
                debug!(
                    session_id = session.session_id,
                    "NvEnc UnmapInputResource {:?} -> {}", mapped_resource, res
                );
                if res == NV_ENC_SUCCESS {
                    NvencResponse::Success
                } else {
                    Self::nvenc_err(res)
                }
            }

            // ── Encoding ─────────────────────────────────────────────
            NvencCommand::EncodePicture {
                encoder,
                mut params,
                input,
                output,
                picture_type: _,
            } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_encoder = match self.resolve_encoder(&encoder) {
                    Ok(e) => e,
                    Err(e) => return e,
                };
                self.ensure_encoder_context(&encoder);

                // Resolve input handle - could be an input buffer or mapped resource
                let real_input = self
                    .input_buffer_handles
                    .get(&input)
                    .map(|b| *b)
                    .or_else(|| self.mapped_resource_handles.get(&input).map(|r| *r))
                    .unwrap_or(std::ptr::null_mut());

                // Resolve output handle - should be a bitstream buffer
                let real_output = self.resolve_bitstream_buffer(&output)
                    .unwrap_or(std::ptr::null_mut());

                // Patch NV_ENC_PIC_PARAMS.version at offset 0
                d.patch_struct_version(&mut params, 0);

                // Patch the real input/output buffer pointers into the params.
                // The client sent fake IDs which we now replace with real GPU pointers.
                // NV_ENC_PIC_PARAMS layout (64-bit):
                //   offset 40: inputBuffer (NV_ENC_INPUT_PTR = void*)
                //   offset 48: outputBitstream (NV_ENC_OUTPUT_PTR = void*)
                if params.len() >= 64 {
                    let input_bytes = (real_input as u64).to_le_bytes();
                    let output_bytes = (real_output as u64).to_le_bytes();
                    params[40..48].copy_from_slice(&input_bytes);
                    params[48..56].copy_from_slice(&output_bytes);
                    // Zero out completionEvent at offset 56 (client-side pointer)
                    params[56..64].copy_from_slice(&[0u8; 8]);
                    debug!(
                        session_id = session.session_id,
                        "NvEnc EncodePicture: patched input={:?} output={:?} (from {:?}/{:?})",
                        real_input, real_output, input, output
                    );
                }

                let res = d.encode_picture(real_encoder, &mut params);
                debug!(
                    session_id = session.session_id,
                    "NvEnc EncodePicture -> {}", res
                );
                if res == NV_ENC_SUCCESS {
                    NvencResponse::Success
                } else if res == nvenc_driver::NV_ENC_ERR_NEED_MORE_INPUT {
                    // Not an error - encoder needs more frames before producing output
                    NvencResponse::Success
                } else {
                    Self::nvenc_err(res)
                }
            }

            // ── Parameter retrieval ──────────────────────────────────
            NvencCommand::GetSequenceParams { encoder } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_encoder = match self.resolve_encoder(&encoder) {
                    Ok(e) => e,
                    Err(e) => return e,
                };
                self.ensure_encoder_context(&encoder);

                match d.get_sequence_params(real_encoder) {
                    Ok(data) => {
                        debug!(
                            session_id = session.session_id,
                            "NvEnc GetSequenceParams -> {} bytes", data.len()
                        );
                        NvencResponse::SequenceParams { data }
                    }
                    Err(e) => Self::nvenc_err(e),
                }
            }

            NvencCommand::GetEncodeStats { encoder } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_encoder = match self.resolve_encoder(&encoder) {
                    Ok(e) => e,
                    Err(e) => return e,
                };
                self.ensure_encoder_context(&encoder);

                match d.get_encode_stats(real_encoder) {
                    Ok(data) => {
                        debug!(
                            session_id = session.session_id,
                            "NvEnc GetEncodeStats -> {} bytes", data.len()
                        );
                        NvencResponse::EncodeStats { data }
                    }
                    Err(e) => Self::nvenc_err(e),
                }
            }

            NvencCommand::InvalidateRefFrames {
                encoder,
                invalid_ref_frame_timestamp,
            } => {
                let d = match self.driver() {
                    Ok(d) => d,
                    Err(e) => return e,
                };
                let real_encoder = match self.resolve_encoder(&encoder) {
                    Ok(e) => e,
                    Err(e) => return e,
                };
                self.ensure_encoder_context(&encoder);

                let res =
                    d.invalidate_ref_frames(real_encoder, invalid_ref_frame_timestamp);
                debug!(
                    session_id = session.session_id,
                    "NvEnc InvalidateRefFrames(ts={}) -> {}",
                    invalid_ref_frame_timestamp, res
                );
                if res == NV_ENC_SUCCESS {
                    NvencResponse::Success
                } else {
                    Self::nvenc_err(res)
                }
            }

            // ── Async events ─────────────────────────────────────────
            NvencCommand::RegisterAsyncEvent { encoder: _ } => {
                // Async events are platform-specific (Windows event handles).
                // Over the network, we handle synchronization differently, so
                // we return a placeholder handle.
                warn!("NvEnc RegisterAsyncEvent not supported over network");
                NvencResponse::Error {
                    code: nvenc_driver::NV_ENC_ERR_UNIMPLEMENTED,
                    message: "async events not supported over network transport"
                        .to_string(),
                }
            }

            NvencCommand::UnregisterAsyncEvent {
                encoder: _,
                event: _,
            } => {
                warn!("NvEnc UnregisterAsyncEvent not supported over network");
                NvencResponse::Error {
                    code: nvenc_driver::NV_ENC_ERR_UNIMPLEMENTED,
                    message: "async events not supported over network transport"
                        .to_string(),
                }
            }
        }
    }

    /// Clean up all NVENC resources owned by a disconnecting session.
    /// Destroys resources in reverse-dependency order.
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

        // Ensure CUDA context is current for cleanup — find any encoder for this session
        for entry in self.encoder_contexts.iter() {
            if entry.key().session_id == session.session_id {
                if let Some(cuda_driver) = self.cuda_executor.driver_ref() {
                    let _ = cuda_driver.ctx_set_current(*entry.value() as crate::cuda_driver::CUcontext);
                }
                break;
            }
        }

        // Pass 1: Unmap mapped resources
        for h in handles
            .iter()
            .filter(|h| h.resource_type == ResourceType::NvEncMappedResource)
        {
            if let Some((_, mapped)) = self.mapped_resource_handles.remove(h) {
                // Need the encoder handle to unmap. Find the encoder for this session.
                for entry in self.encoder_handles.iter() {
                    if entry.key().session_id == session.session_id {
                        driver.unmap_input_resource(*entry.value(), mapped);
                        break;
                    }
                }
                cleaned += 1;
            }
        }

        // Pass 2: Unregister registered resources
        for h in handles
            .iter()
            .filter(|h| h.resource_type == ResourceType::NvEncRegisteredResource)
        {
            if let Some((_, resource)) = self.registered_resource_handles.remove(h) {
                for entry in self.encoder_handles.iter() {
                    if entry.key().session_id == session.session_id {
                        driver.unregister_resource(*entry.value(), resource);
                        break;
                    }
                }
                cleaned += 1;
            }
        }

        // Pass 3: Destroy bitstream buffers
        for h in handles
            .iter()
            .filter(|h| h.resource_type == ResourceType::NvEncBitstreamBuffer)
        {
            if let Some((_, buffer)) = self.bitstream_buffer_handles.remove(h) {
                for entry in self.encoder_handles.iter() {
                    if entry.key().session_id == session.session_id {
                        driver.destroy_bitstream_buffer(*entry.value(), buffer);
                        break;
                    }
                }
                cleaned += 1;
            }
        }

        // Pass 4: Destroy input buffers
        for h in handles
            .iter()
            .filter(|h| h.resource_type == ResourceType::NvEncInputBuffer)
        {
            if let Some((_, buffer)) = self.input_buffer_handles.remove(h) {
                for entry in self.encoder_handles.iter() {
                    if entry.key().session_id == session.session_id {
                        driver.destroy_input_buffer(*entry.value(), buffer);
                        break;
                    }
                }
                cleaned += 1;
            }
        }

        // Pass 5: Destroy encoder sessions
        for h in handles
            .iter()
            .filter(|h| h.resource_type == ResourceType::NvEncSession)
        {
            // Ensure CUDA context is current for cleanup
            self.ensure_encoder_context(h);
            if let Some((_, encoder)) = self.encoder_handles.remove(h) {
                driver.destroy_encoder(encoder);
                cleaned += 1;
            }
            self.encoder_contexts.remove(h);
        }

        if cleaned > 0 {
            info!(
                session_id = session.session_id,
                "cleaned up {} NVENC resource(s)", cleaned
            );
        }
    }
}
