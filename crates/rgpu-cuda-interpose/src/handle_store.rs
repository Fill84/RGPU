//! Thread-safe handle-to-NetworkHandle mapping for the CUDA interpose library.
//!
//! Maps local opaque IDs (returned to the application as CUdevice, CUcontext, etc.)
//! to NetworkHandles used for IPC communication with the RGPU daemon.

use dashmap::DashMap;
use rgpu_protocol::handle::NetworkHandle;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

static NEXT_ID: AtomicU64 = AtomicU64::new(0x1000);

static DEVICE_MAP: OnceLock<DashMap<u64, NetworkHandle>> = OnceLock::new();
static CTX_MAP: OnceLock<DashMap<u64, NetworkHandle>> = OnceLock::new();
static MOD_MAP: OnceLock<DashMap<u64, NetworkHandle>> = OnceLock::new();
static FUNC_MAP: OnceLock<DashMap<u64, NetworkHandle>> = OnceLock::new();
static MEM_MAP: OnceLock<DashMap<u64, NetworkHandle>> = OnceLock::new();
static STREAM_MAP: OnceLock<DashMap<u64, NetworkHandle>> = OnceLock::new();
static EVENT_MAP: OnceLock<DashMap<u64, NetworkHandle>> = OnceLock::new();
static MEMPOOL_MAP: OnceLock<DashMap<u64, NetworkHandle>> = OnceLock::new();
static LINKER_MAP: OnceLock<DashMap<u64, NetworkHandle>> = OnceLock::new();
static HOST_MEM_MAP: OnceLock<DashMap<u64, NetworkHandle>> = OnceLock::new();

fn device_map() -> &'static DashMap<u64, NetworkHandle> {
    DEVICE_MAP.get_or_init(DashMap::new)
}
fn ctx_map() -> &'static DashMap<u64, NetworkHandle> {
    CTX_MAP.get_or_init(DashMap::new)
}
fn mod_map() -> &'static DashMap<u64, NetworkHandle> {
    MOD_MAP.get_or_init(DashMap::new)
}
fn func_map() -> &'static DashMap<u64, NetworkHandle> {
    FUNC_MAP.get_or_init(DashMap::new)
}
fn mem_map() -> &'static DashMap<u64, NetworkHandle> {
    MEM_MAP.get_or_init(DashMap::new)
}
fn stream_map() -> &'static DashMap<u64, NetworkHandle> {
    STREAM_MAP.get_or_init(DashMap::new)
}
fn event_map() -> &'static DashMap<u64, NetworkHandle> {
    EVENT_MAP.get_or_init(DashMap::new)
}
fn mempool_map() -> &'static DashMap<u64, NetworkHandle> {
    MEMPOOL_MAP.get_or_init(DashMap::new)
}
fn linker_map() -> &'static DashMap<u64, NetworkHandle> {
    LINKER_MAP.get_or_init(DashMap::new)
}
fn host_mem_map() -> &'static DashMap<u64, NetworkHandle> {
    HOST_MEM_MAP.get_or_init(DashMap::new)
}

fn alloc_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

// ── Device ──────────────────────────────────────────────────────
pub fn store_device(handle: NetworkHandle) -> u64 {
    let id = alloc_id();
    device_map().insert(id, handle);
    id
}
pub fn get_device(id: u64) -> Option<NetworkHandle> {
    device_map().get(&id).map(|v| *v)
}

// ── Context ─────────────────────────────────────────────────────
pub fn store_ctx(handle: NetworkHandle) -> u64 {
    let id = alloc_id();
    ctx_map().insert(id, handle);
    id
}
pub fn get_ctx(id: u64) -> Option<NetworkHandle> {
    ctx_map().get(&id).map(|v| *v)
}
pub fn remove_ctx(id: u64) {
    ctx_map().remove(&id);
}

// ── Module ──────────────────────────────────────────────────────
pub fn store_mod(handle: NetworkHandle) -> u64 {
    let id = alloc_id();
    mod_map().insert(id, handle);
    id
}
pub fn get_mod(id: u64) -> Option<NetworkHandle> {
    mod_map().get(&id).map(|v| *v)
}
pub fn remove_mod(id: u64) {
    mod_map().remove(&id);
}

// ── Function ────────────────────────────────────────────────────
pub fn store_func(handle: NetworkHandle) -> u64 {
    let id = alloc_id();
    func_map().insert(id, handle);
    id
}
pub fn get_func(id: u64) -> Option<NetworkHandle> {
    func_map().get(&id).map(|v| *v)
}

// ── Memory ──────────────────────────────────────────────────────
pub fn store_mem(handle: NetworkHandle) -> u64 {
    let id = alloc_id();
    mem_map().insert(id, handle);
    id
}
pub fn get_mem(id: u64) -> Option<NetworkHandle> {
    mem_map().get(&id).map(|v| *v)
}
pub fn remove_mem(id: u64) {
    mem_map().remove(&id);
}
pub fn get_mem_by_ptr(ptr: u64) -> Option<NetworkHandle> {
    get_mem(ptr)
}

// ── Stream ──────────────────────────────────────────────────────
pub fn store_stream(handle: NetworkHandle) -> u64 {
    let id = alloc_id();
    stream_map().insert(id, handle);
    id
}
pub fn get_stream(id: u64) -> Option<NetworkHandle> {
    stream_map().get(&id).map(|v| *v)
}
pub fn remove_stream(id: u64) {
    stream_map().remove(&id);
}

// ── Event ───────────────────────────────────────────────────────
pub fn store_event(handle: NetworkHandle) -> u64 {
    let id = alloc_id();
    event_map().insert(id, handle);
    id
}
pub fn get_event(id: u64) -> Option<NetworkHandle> {
    event_map().get(&id).map(|v| *v)
}
pub fn remove_event(id: u64) {
    event_map().remove(&id);
}

// ── Memory Pool ─────────────────────────────────────────────────
pub fn store_mempool(handle: NetworkHandle) -> u64 {
    let id = alloc_id();
    mempool_map().insert(id, handle);
    id
}
pub fn get_mempool(id: u64) -> Option<NetworkHandle> {
    mempool_map().get(&id).map(|v| *v)
}
pub fn remove_mempool(id: u64) {
    mempool_map().remove(&id);
}

// ── Linker ──────────────────────────────────────────────────────
pub fn store_linker(handle: NetworkHandle) -> u64 {
    let id = alloc_id();
    linker_map().insert(id, handle);
    id
}
pub fn get_linker(id: u64) -> Option<NetworkHandle> {
    linker_map().get(&id).map(|v| *v)
}
pub fn remove_linker(id: u64) {
    linker_map().remove(&id);
}

// ── Host Memory ─────────────────────────────────────────────────
pub fn store_host_mem(handle: NetworkHandle) -> u64 {
    let id = alloc_id();
    host_mem_map().insert(id, handle);
    id
}
pub fn get_host_mem(id: u64) -> Option<NetworkHandle> {
    host_mem_map().get(&id).map(|v| *v)
}
pub fn remove_host_mem(id: u64) {
    host_mem_map().remove(&id);
}
