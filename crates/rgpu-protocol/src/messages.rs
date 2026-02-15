use serde::{Deserialize, Serialize};

use crate::cuda_commands::{CudaCommand, CudaResponse};
use crate::error::ProtocolError;
use crate::gpu_info::GpuInfo;
use crate::vulkan_commands::{VulkanCommand, VulkanResponse};

/// A unique identifier for a request, used for matching responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct RequestId(pub u64);

/// Top-level message envelope for the RGPU protocol.
#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub enum Message {
    // ── Connection establishment ────────────────────────────
    /// Initial handshake from client or server.
    Hello {
        protocol_version: u32,
        name: String,
        /// Server sends a random challenge for auth.
        challenge: Option<Vec<u8>>,
    },

    /// Authentication message from client.
    Authenticate {
        token: String,
        challenge_response: Vec<u8>,
    },

    /// Authentication result from server.
    AuthResult {
        success: bool,
        session_id: Option<u32>,
        server_id: Option<u16>,
        available_gpus: Vec<GpuInfo>,
        error_message: Option<String>,
    },

    // ── GPU discovery ───────────────────────────────────────
    /// Request GPU information from server.
    QueryGpus,

    /// Response with available GPU info.
    GpuList(Vec<GpuInfo>),

    // ── CUDA commands ───────────────────────────────────────
    CudaCommand {
        request_id: RequestId,
        command: CudaCommand,
    },
    CudaResponse {
        request_id: RequestId,
        response: CudaResponse,
    },

    // ── Vulkan commands ─────────────────────────────────────
    VulkanCommand {
        request_id: RequestId,
        command: VulkanCommand,
    },
    VulkanResponse {
        request_id: RequestId,
        response: VulkanResponse,
    },

    // ── Batching ────────────────────────────────────────────
    /// Batched CUDA commands for pipelining (void commands sent fire-and-forget).
    CudaBatch(Vec<CudaCommand>),

    // ── Monitoring ──────────────────────────────────────────
    /// Request server metrics snapshot.
    QueryMetrics,

    /// Server metrics response.
    MetricsData {
        connections_total: u64,
        connections_active: u32,
        requests_total: u64,
        errors_total: u64,
        cuda_commands: u64,
        vulkan_commands: u64,
        uptime_secs: u64,
        server_id: u16,
        server_address: String,
    },

    // ── Keepalive ───────────────────────────────────────────
    Ping,
    Pong,

    // ── Error ───────────────────────────────────────────────
    Error(ProtocolError),
}

/// Current protocol version.
pub const PROTOCOL_VERSION: u32 = 3;
