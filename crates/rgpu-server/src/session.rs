use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

use rgpu_protocol::handle::{NetworkHandle, ResourceType};

/// Per-client session state on the server side.
/// Tracks all resources allocated by a client for cleanup on disconnect.
pub struct Session {
    pub session_id: u32,
    pub client_name: String,
    /// All handles allocated by this session
    allocated_handles: parking_lot::RwLock<HashSet<NetworkHandle>>,
    /// Handle counter
    next_resource_id: AtomicU64,
    /// Server ID (for handle generation)
    server_id: u16,
}

impl Session {
    pub fn new(session_id: u32, server_id: u16, client_name: String) -> Self {
        Self {
            session_id,
            client_name,
            allocated_handles: parking_lot::RwLock::new(HashSet::new()),
            next_resource_id: AtomicU64::new(1),
            server_id,
        }
    }

    /// Allocate a new network handle with the given resource type.
    pub fn alloc_handle(&self, resource_type: ResourceType) -> NetworkHandle {
        let handle = NetworkHandle {
            server_id: self.server_id,
            session_id: self.session_id,
            resource_id: self.next_resource_id.fetch_add(1, Ordering::Relaxed),
            resource_type,
        };
        self.allocated_handles.write().insert(handle);
        handle
    }

    /// Validate that a handle belongs to this session.
    pub fn validate_handle(&self, handle: &NetworkHandle) -> bool {
        handle.session_id == self.session_id
            && self.allocated_handles.read().contains(handle)
    }

    /// Get all allocated handles (for cleanup).
    pub fn all_handles(&self) -> Vec<NetworkHandle> {
        self.allocated_handles.read().iter().cloned().collect()
    }

    /// Remove a handle from tracking.
    pub fn remove_handle(&self, handle: &NetworkHandle) {
        self.allocated_handles.write().remove(handle);
    }

    /// Get this session's server ID.
    pub fn server_id(&self) -> u16 {
        self.server_id
    }
}
