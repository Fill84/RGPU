use std::sync::atomic::{AtomicU64, Ordering};

use dashmap::DashMap;
use rgpu_protocol::handle::{NetworkHandle, ResourceType};

/// Bidirectional handle mapping between local opaque handles and network handles.
/// Used by the CUDA interposition library and Vulkan ICD to translate between
/// local handles (seen by the application) and network handles (used over the wire).
pub struct HandleMap {
    /// Local handle -> NetworkHandle
    local_to_network: DashMap<u64, NetworkHandle>,
    /// NetworkHandle -> Local handle
    network_to_local: DashMap<NetworkHandle, u64>,
    /// Counter for generating unique local handles
    next_local: AtomicU64,
}

impl HandleMap {
    pub fn new() -> Self {
        Self {
            local_to_network: DashMap::new(),
            network_to_local: DashMap::new(),
            // Start from 1 to avoid confusion with NULL/0 handles
            next_local: AtomicU64::new(1),
        }
    }

    /// Register a network handle and return a new local handle.
    pub fn insert(&self, network_handle: NetworkHandle) -> u64 {
        let local = self.next_local.fetch_add(1, Ordering::Relaxed);
        self.local_to_network.insert(local, network_handle);
        self.network_to_local.insert(network_handle, local);
        local
    }

    /// Look up a network handle by local handle.
    pub fn to_network(&self, local: u64) -> Option<NetworkHandle> {
        self.local_to_network.get(&local).map(|v| *v)
    }

    /// Look up a local handle by network handle.
    pub fn to_local(&self, network: &NetworkHandle) -> Option<u64> {
        self.network_to_local.get(network).map(|v| *v)
    }

    /// Remove a handle pair by local handle.
    pub fn remove_by_local(&self, local: u64) -> Option<NetworkHandle> {
        if let Some((_, network)) = self.local_to_network.remove(&local) {
            self.network_to_local.remove(&network);
            Some(network)
        } else {
            None
        }
    }

    /// Remove a handle pair by network handle.
    pub fn remove_by_network(&self, network: &NetworkHandle) -> Option<u64> {
        if let Some((_, local)) = self.network_to_local.remove(network) {
            self.local_to_network.remove(&local);
            Some(local)
        } else {
            None
        }
    }

    /// Return number of active handles.
    pub fn len(&self) -> usize {
        self.local_to_network.len()
    }

    pub fn is_empty(&self) -> bool {
        self.local_to_network.is_empty()
    }
}

impl Default for HandleMap {
    fn default() -> Self {
        Self::new()
    }
}

/// Server-side handle allocator. Generates unique NetworkHandles for a session.
pub struct HandleAllocator {
    server_id: u16,
    session_id: u32,
    next_id: AtomicU64,
}

impl HandleAllocator {
    pub fn new(server_id: u16, session_id: u32) -> Self {
        Self {
            server_id,
            session_id,
            next_id: AtomicU64::new(1),
        }
    }

    /// Allocate a new network handle with the given resource type.
    pub fn alloc(&self, resource_type: ResourceType) -> NetworkHandle {
        NetworkHandle {
            server_id: self.server_id,
            session_id: self.session_id,
            resource_id: self.next_id.fetch_add(1, Ordering::Relaxed),
            resource_type,
        }
    }
}
