//! Thread-safe handle-to-NetworkHandle mapping for the NVENC interpose library.
//!
//! Maps local opaque IDs (returned to the application as void* encoder, NV_ENC_INPUT_PTR, etc.)
//! to NetworkHandles used for IPC communication with the RGPU daemon.

use dashmap::DashMap;
use rgpu_protocol::handle::NetworkHandle;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

/// Start ID at 0x2000 to avoid collision with CUDA interpose handles (starting at 0x1000).
static NEXT_ID: AtomicU64 = AtomicU64::new(0x2000);

static ENCODER_MAP: OnceLock<DashMap<u64, NetworkHandle>> = OnceLock::new();
static INPUT_BUFFER_MAP: OnceLock<DashMap<u64, NetworkHandle>> = OnceLock::new();
static BITSTREAM_BUFFER_MAP: OnceLock<DashMap<u64, NetworkHandle>> = OnceLock::new();
static REGISTERED_RESOURCE_MAP: OnceLock<DashMap<u64, NetworkHandle>> = OnceLock::new();
static MAPPED_RESOURCE_MAP: OnceLock<DashMap<u64, NetworkHandle>> = OnceLock::new();
static ASYNC_EVENT_MAP: OnceLock<DashMap<u64, NetworkHandle>> = OnceLock::new();

fn encoder_map() -> &'static DashMap<u64, NetworkHandle> {
    ENCODER_MAP.get_or_init(DashMap::new)
}
fn input_buffer_map() -> &'static DashMap<u64, NetworkHandle> {
    INPUT_BUFFER_MAP.get_or_init(DashMap::new)
}
fn bitstream_buffer_map() -> &'static DashMap<u64, NetworkHandle> {
    BITSTREAM_BUFFER_MAP.get_or_init(DashMap::new)
}
fn registered_resource_map() -> &'static DashMap<u64, NetworkHandle> {
    REGISTERED_RESOURCE_MAP.get_or_init(DashMap::new)
}
fn mapped_resource_map() -> &'static DashMap<u64, NetworkHandle> {
    MAPPED_RESOURCE_MAP.get_or_init(DashMap::new)
}
fn async_event_map() -> &'static DashMap<u64, NetworkHandle> {
    ASYNC_EVENT_MAP.get_or_init(DashMap::new)
}

fn alloc_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

// ── Encoder sessions ─────────────────────────────────────────────
pub fn store_encoder(handle: NetworkHandle) -> u64 {
    let id = alloc_id();
    encoder_map().insert(id, handle);
    id
}
pub fn get_encoder(id: u64) -> Option<NetworkHandle> {
    encoder_map().get(&id).map(|v| *v)
}
pub fn remove_encoder(id: u64) {
    encoder_map().remove(&id);
}

// ── Input buffers ────────────────────────────────────────────────
pub fn store_input_buffer(handle: NetworkHandle) -> u64 {
    let id = alloc_id();
    input_buffer_map().insert(id, handle);
    id
}
pub fn get_input_buffer(id: u64) -> Option<NetworkHandle> {
    input_buffer_map().get(&id).map(|v| *v)
}
pub fn remove_input_buffer(id: u64) {
    input_buffer_map().remove(&id);
}

// ── Bitstream buffers ────────────────────────────────────────────
pub fn store_bitstream_buffer(handle: NetworkHandle) -> u64 {
    let id = alloc_id();
    bitstream_buffer_map().insert(id, handle);
    id
}
pub fn get_bitstream_buffer(id: u64) -> Option<NetworkHandle> {
    bitstream_buffer_map().get(&id).map(|v| *v)
}
pub fn remove_bitstream_buffer(id: u64) {
    bitstream_buffer_map().remove(&id);
}

// ── Registered resources ─────────────────────────────────────────
pub fn store_registered_resource(handle: NetworkHandle) -> u64 {
    let id = alloc_id();
    registered_resource_map().insert(id, handle);
    id
}
pub fn get_registered_resource(id: u64) -> Option<NetworkHandle> {
    registered_resource_map().get(&id).map(|v| *v)
}
pub fn remove_registered_resource(id: u64) {
    registered_resource_map().remove(&id);
}

// ── Mapped resources ─────────────────────────────────────────────
pub fn store_mapped_resource(handle: NetworkHandle) -> u64 {
    let id = alloc_id();
    mapped_resource_map().insert(id, handle);
    id
}
pub fn get_mapped_resource(id: u64) -> Option<NetworkHandle> {
    mapped_resource_map().get(&id).map(|v| *v)
}
pub fn remove_mapped_resource(id: u64) {
    mapped_resource_map().remove(&id);
}

// ── Async events ─────────────────────────────────────────────────
pub fn store_async_event(handle: NetworkHandle) -> u64 {
    let id = alloc_id();
    async_event_map().insert(id, handle);
    id
}
pub fn get_async_event(id: u64) -> Option<NetworkHandle> {
    async_event_map().get(&id).map(|v| *v)
}
pub fn remove_async_event(id: u64) {
    async_event_map().remove(&id);
}
