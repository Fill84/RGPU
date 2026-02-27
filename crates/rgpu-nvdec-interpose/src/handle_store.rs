//! Thread-safe handle-to-NetworkHandle mapping for the NVDEC interpose library.
//!
//! Maps local opaque IDs (returned to the application as CUvideodecoder, CUvideoparser, etc.)
//! to NetworkHandles used for IPC communication with the RGPU daemon.

use dashmap::DashMap;
use rgpu_protocol::handle::NetworkHandle;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

/// Allocator starting at 0x3000 to avoid collision with CUDA interpose (0x1000)
/// and VK ICD (0x2000) handle ranges.
static NEXT_ID: AtomicU64 = AtomicU64::new(0x3000);

static DECODER_MAP: OnceLock<DashMap<u64, NetworkHandle>> = OnceLock::new();
static PARSER_MAP: OnceLock<DashMap<u64, NetworkHandle>> = OnceLock::new();
static CTX_LOCK_MAP: OnceLock<DashMap<u64, NetworkHandle>> = OnceLock::new();
static MAPPED_FRAME_MAP: OnceLock<DashMap<u64, NetworkHandle>> = OnceLock::new();

fn decoder_map() -> &'static DashMap<u64, NetworkHandle> {
    DECODER_MAP.get_or_init(DashMap::new)
}
fn parser_map() -> &'static DashMap<u64, NetworkHandle> {
    PARSER_MAP.get_or_init(DashMap::new)
}
fn ctx_lock_map() -> &'static DashMap<u64, NetworkHandle> {
    CTX_LOCK_MAP.get_or_init(DashMap::new)
}
fn mapped_frame_map() -> &'static DashMap<u64, NetworkHandle> {
    MAPPED_FRAME_MAP.get_or_init(DashMap::new)
}

fn alloc_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

// ── Decoder ─────────────────────────────────────────────────────
pub fn store_decoder(handle: NetworkHandle) -> u64 {
    let id = alloc_id();
    decoder_map().insert(id, handle);
    id
}
pub fn get_decoder(id: u64) -> Option<NetworkHandle> {
    decoder_map().get(&id).map(|v| *v)
}
pub fn remove_decoder(id: u64) {
    decoder_map().remove(&id);
}

// ── Parser ──────────────────────────────────────────────────────
pub fn store_parser(handle: NetworkHandle) -> u64 {
    let id = alloc_id();
    parser_map().insert(id, handle);
    id
}
pub fn get_parser(id: u64) -> Option<NetworkHandle> {
    parser_map().get(&id).map(|v| *v)
}
pub fn remove_parser(id: u64) {
    parser_map().remove(&id);
}

// ── Context Lock ────────────────────────────────────────────────
pub fn store_ctx_lock(handle: NetworkHandle) -> u64 {
    let id = alloc_id();
    ctx_lock_map().insert(id, handle);
    id
}
pub fn get_ctx_lock(id: u64) -> Option<NetworkHandle> {
    ctx_lock_map().get(&id).map(|v| *v)
}
pub fn remove_ctx_lock(id: u64) {
    ctx_lock_map().remove(&id);
}

// ── Mapped Frame ────────────────────────────────────────────────
pub fn store_mapped_frame(handle: NetworkHandle) -> u64 {
    let id = alloc_id();
    mapped_frame_map().insert(id, handle);
    id
}
pub fn get_mapped_frame(id: u64) -> Option<NetworkHandle> {
    mapped_frame_map().get(&id).map(|v| *v)
}
pub fn remove_mapped_frame(id: u64) {
    mapped_frame_map().remove(&id);
}
