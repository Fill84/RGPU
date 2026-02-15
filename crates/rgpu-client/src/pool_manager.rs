use std::collections::HashMap;

use tokio::sync::RwLock;
use tracing::info;

use rgpu_protocol::gpu_info::GpuInfo;
use rgpu_protocol::handle::NetworkHandle;

use rgpu_core::config::{GpuOrdering, ServerEndpoint};

/// Represents metadata about a connection to one RGPU server.
pub struct ServerConnection {
    pub endpoint: ServerEndpoint,
    pub server_id: u16,
    pub gpus: Vec<GpuInfo>,
    pub status: ConnectionStatus,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConnectionStatus {
    Connected,
    Connecting,
    Disconnected(String),
}

/// An entry in the unified GPU pool.
#[derive(Debug, Clone)]
pub struct GpuPoolEntry {
    /// Index in the unified pool
    pub pool_index: u32,
    /// Which server owns this GPU (index into server_conns vec)
    pub server_index: usize,
    /// Server-side device index
    pub server_device_index: u32,
    /// Full GPU info
    pub info: GpuInfo,
    /// Is this a local GPU?
    pub is_local: bool,
}

/// Manages connections to multiple RGPU servers and aggregates their GPUs.
/// Acts as the routing brain: resolves handles and ordinals to server indices.
pub struct GpuPoolManager {
    servers: RwLock<Vec<ServerConnection>>,
    gpu_pool: RwLock<Vec<GpuPoolEntry>>,
    /// Maps server_id → index in server_conns vec
    server_id_to_index: RwLock<HashMap<u16, usize>>,
    ordering: GpuOrdering,
}

impl GpuPoolManager {
    pub fn new(ordering: GpuOrdering) -> Self {
        Self {
            servers: RwLock::new(Vec::new()),
            gpu_pool: RwLock::new(Vec::new()),
            server_id_to_index: RwLock::new(HashMap::new()),
            ordering,
        }
    }

    /// Add a server and its discovered GPUs to the pool.
    pub async fn add_server(
        &self,
        endpoint: ServerEndpoint,
        server_id: u16,
        gpus: Vec<GpuInfo>,
    ) {
        let server_index = {
            let mut servers = self.servers.write().await;
            let idx = servers.len();
            servers.push(ServerConnection {
                endpoint,
                server_id,
                gpus: gpus.clone(),
                status: ConnectionStatus::Connected,
            });
            idx
        };

        // Register the server_id → index mapping
        self.server_id_to_index
            .write()
            .await
            .insert(server_id, server_index);

        // Add GPUs to the pool
        let mut pool = self.gpu_pool.write().await;
        for gpu in gpus {
            let pool_index = pool.len() as u32;
            pool.push(GpuPoolEntry {
                pool_index,
                server_index,
                server_device_index: gpu.server_device_index,
                info: gpu,
                is_local: false,
            });
        }

        info!(
            "GPU pool now has {} GPU(s) from {} server(s)",
            pool.len(),
            server_index + 1
        );
    }

    /// Add a mapping from server_id to server_conns index.
    pub async fn add_server_mapping(&self, server_id: u16, server_index: usize) {
        self.server_id_to_index
            .write()
            .await
            .insert(server_id, server_index);
    }

    /// Resolve a NetworkHandle to the server_conns index that owns it.
    pub async fn server_index_for_handle(&self, handle: &NetworkHandle) -> Option<usize> {
        self.server_id_to_index
            .read()
            .await
            .get(&handle.server_id)
            .copied()
    }

    /// Map a pool GPU ordinal to (server_index, server_local_device_index).
    pub async fn server_for_pool_ordinal(&self, ordinal: u32) -> Option<(usize, u32)> {
        let pool = self.gpu_pool.read().await;
        pool.iter()
            .find(|g| g.pool_index == ordinal)
            .map(|g| (g.server_index, g.server_device_index))
    }

    /// Get the first connected server index (fallback for creation commands).
    pub async fn default_server_index(&self) -> Option<usize> {
        let servers = self.servers.read().await;
        servers
            .iter()
            .enumerate()
            .find(|(_, s)| s.status == ConnectionStatus::Connected)
            .map(|(i, _)| i)
    }

    /// Get all connected server indices (for broadcast commands like CreateInstance).
    pub async fn all_connected_server_indices(&self) -> Vec<usize> {
        let servers = self.servers.read().await;
        servers
            .iter()
            .enumerate()
            .filter(|(_, s)| s.status == ConnectionStatus::Connected)
            .map(|(i, _)| i)
            .collect()
    }

    /// Update the connection status for a server.
    pub async fn set_server_status(&self, server_index: usize, status: ConnectionStatus) {
        let mut servers = self.servers.write().await;
        if let Some(server) = servers.get_mut(server_index) {
            server.status = status;
        }
    }

    /// Get all GPUs in the pool.
    pub async fn get_all_gpus(&self) -> Vec<GpuPoolEntry> {
        self.gpu_pool.read().await.clone()
    }

    /// Get the number of CUDA-capable GPUs in the pool.
    pub async fn cuda_device_count(&self) -> u32 {
        let pool = self.gpu_pool.read().await;
        pool.iter().filter(|g| g.info.supports_cuda).count() as u32
    }

    /// Get the number of Vulkan-capable GPUs in the pool.
    pub async fn vulkan_device_count(&self) -> u32 {
        let pool = self.gpu_pool.read().await;
        pool.iter().filter(|g| g.info.supports_vulkan).count() as u32
    }

    /// Get a specific GPU by pool index.
    pub async fn get_gpu(&self, pool_index: u32) -> Option<GpuPoolEntry> {
        let pool = self.gpu_pool.read().await;
        pool.iter().find(|g| g.pool_index == pool_index).cloned()
    }

    /// Number of connected servers.
    pub async fn server_count(&self) -> usize {
        self.servers.read().await.len()
    }

    /// Apply GPU ordering based on the configured preference.
    pub async fn apply_ordering(&self) {
        let mut pool = self.gpu_pool.write().await;

        match self.ordering {
            GpuOrdering::LocalFirst => {
                pool.sort_by(|a, b| b.is_local.cmp(&a.is_local));
            }
            GpuOrdering::RemoteFirst => {
                pool.sort_by(|a, b| a.is_local.cmp(&b.is_local));
            }
            GpuOrdering::ByCapability => {
                pool.sort_by(|a, b| b.info.total_memory.cmp(&a.info.total_memory));
            }
        }

        // Renumber pool indices after sorting
        for (i, entry) in pool.iter_mut().enumerate() {
            entry.pool_index = i as u32;
        }

        info!(
            "GPU pool ordered ({:?}): {}",
            self.ordering,
            pool.iter()
                .map(|g| format!("{}({})", g.info.device_name, g.server_index))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
}
