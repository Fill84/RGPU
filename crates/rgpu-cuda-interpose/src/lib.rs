//! CUDA Driver API interception library.
//!
//! This cdylib replaces the standard CUDA driver library (libcuda.so / nvcuda.dll).
//! It intercepts CUDA driver API calls and forwards them to the RGPU client daemon
//! via IPC, which in turn sends them to a remote RGPU server.
//!
//! Usage:
//! - Linux: LD_PRELOAD=librgpu_cuda_interpose.so <application>
//! - Windows: Place as nvcuda.dll in the application's directory

mod ipc_client;
pub mod handle_store;
pub mod error;
pub mod proc_address;
pub mod stubs;

pub(crate) mod device;
pub(crate) mod context;
pub(crate) mod memory;
pub(crate) mod module;
pub(crate) mod execution;
pub(crate) mod stream;
pub(crate) mod event;

use std::cell::RefCell;
use std::ffi::{c_char, c_int, c_void};
use std::sync::OnceLock;

use tracing::error;

use rgpu_protocol::cuda_commands::{CudaCommand, CudaResponse};
use rgpu_protocol::handle::{NetworkHandle, ResourceType};

use ipc_client::IpcClient;

// CUDA types
pub(crate) type CUresult = c_int;
pub(crate) type CUdevice = c_int;
pub(crate) type CUcontext = *mut c_void;
pub(crate) type CUmodule = *mut c_void;
pub(crate) type CUfunction = *mut c_void;
pub(crate) type CUdeviceptr = u64;
pub(crate) type CUstream = *mut c_void;
pub(crate) type CUevent = *mut c_void;
pub(crate) type CUlinkState = *mut c_void;
pub(crate) type CUmemoryPool = *mut c_void;

pub(crate) const CUDA_SUCCESS: CUresult = 0;
pub(crate) const CUDA_ERROR_INVALID_VALUE: CUresult = 1;
const _CUDA_ERROR_NOT_INITIALIZED: CUresult = 3;
pub(crate) const CUDA_ERROR_INVALID_CONTEXT: CUresult = 201;
pub(crate) const CUDA_ERROR_NOT_READY: CUresult = 600;
const _CUDA_ERROR_NOT_SUPPORTED: CUresult = 801;
pub(crate) const CUDA_ERROR_UNKNOWN: CUresult = 999;

// ── Thread-local CUDA context stack ─────────────────────────────────
// CUDA maintains a per-thread context stack. We track it locally to avoid
// relying on the server's thread-local state (which breaks with async runtimes).
thread_local! {
    pub(crate) static CTX_STACK: RefCell<Vec<u64>> = RefCell::new(Vec::new());
}

pub(crate) fn ctx_stack_push(local_id: u64) {
    CTX_STACK.with(|s| s.borrow_mut().push(local_id));
}

pub(crate) fn ctx_stack_pop() -> Option<u64> {
    CTX_STACK.with(|s| s.borrow_mut().pop())
}

pub(crate) fn ctx_stack_top() -> Option<u64> {
    CTX_STACK.with(|s| s.borrow().last().copied())
}

/// Send CtxSetCurrent to the server for the given local context id.
/// If local_id is None, sends a null context (detach).
pub(crate) fn sync_server_context(local_id: Option<u64>) {
    let net_handle = match local_id {
        Some(id) => match handle_store::get_ctx(id) {
            Some(h) => h,
            None => NetworkHandle::null(),
        },
        None => NetworkHandle::null(),
    };
    let _ = send_cuda_command(CudaCommand::CtxSetCurrent { ctx: net_handle });
}

// ── Real CUDA driver loading (hybrid passthrough mode) ─────────────
// When we don't intercept a function, we forward to the real CUDA driver.
// This allows apps to use cublas, cudnn, etc. transparently through the
// real local GPU while known functions go through our IPC interpose.

static REAL_CUDA: OnceLock<Option<libloading::Library>> = OnceLock::new();

/// Get the real CUDA driver library.
/// Tries to load a renamed original first, then falls back to known system paths.
/// Includes anti-recursion: skips any library that exports `rgpu_interpose_marker`.
fn get_real_cuda() -> Option<&'static libloading::Library> {
    REAL_CUDA.get_or_init(|| {
        #[cfg(target_os = "linux")]
        let names: &[&str] = &[
            "libcuda_real.so.1",
            "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
            "/usr/lib/aarch64-linux-gnu/libcuda.so.1",
        ];
        #[cfg(target_os = "windows")]
        let names: &[&str] = &["nvcuda_real.dll"];
        #[cfg(target_os = "macos")]
        let names: &[&str] = &["libcuda_real.dylib"];

        for name in names {
            match unsafe { libloading::Library::new(name) } {
                Ok(lib) => {
                    // Anti-recursion: check if this is our own interpose library
                    let is_us = unsafe {
                        lib.get::<unsafe extern "C" fn() -> c_int>(b"rgpu_interpose_marker").is_ok()
                    };
                    if is_us {
                        tracing::debug!("skipping {} — it's our own interpose", name);
                        continue;
                    }
                    tracing::info!("loaded real CUDA driver from: {}", name);
                    return Some(lib);
                }
                Err(e) => {
                    tracing::debug!("could not load real CUDA from {}: {}", name, e);
                }
            }
        }
        tracing::warn!("could not load real CUDA driver — only remote GPUs will be available");
        None
    }).as_ref()
}

/// Look up a function pointer in the real CUDA driver.
pub(crate) fn real_cuda_proc_address(name: &str) -> Option<*mut c_void> {
    let lib = get_real_cuda()?;
    unsafe {
        lib.get::<*mut c_void>(name.as_bytes())
            .ok()
            .map(|sym| *sym)
    }
}

static IPC_CLIENT: OnceLock<IpcClient> = OnceLock::new();

fn get_client() -> &'static IpcClient {
    IPC_CLIENT.get_or_init(|| {
        let path = rgpu_common::platform::resolve_ipc_address();
        IpcClient::new(&path)
    })
}

pub(crate) fn send_cuda_command(cmd: CudaCommand) -> CudaResponse {
    let client = get_client();
    match client.send_command(cmd) {
        Ok(resp) => resp,
        Err(e) => {
            error!("IPC error: {}", e);
            CudaResponse::Error {
                code: CUDA_ERROR_UNKNOWN,
                message: e.to_string(),
            }
        }
    }
}

/// Create a null/default NetworkHandle for stream references (NULL stream = default).
pub(crate) fn null_stream_handle() -> NetworkHandle {
    NetworkHandle {
        server_id: 0,
        session_id: 0,
        resource_id: 0,
        resource_type: ResourceType::CuStream,
    }
}

// ── Marker function for RGPU interpose DLL detection ────────────────
// Used by the installer/daemon to verify that nvcuda.dll in System32
// is the RGPU interpose library and not the real NVIDIA driver.
#[no_mangle]
pub extern "C" fn rgpu_interpose_marker() -> c_int {
    1
}

// ── Cross-DLL handle resolution ─────────────────────────────────────
// These functions allow other RGPU interpose DLLs (NVENC, NVDEC) in the
// same process to resolve local handle IDs to proper NetworkHandles.

#[allow(non_snake_case)]
unsafe fn rgpu_cuda_resolve_ctx_impl(
    local_id: u64,
    out_server_id: *mut u16,
    out_session_id: *mut u32,
    out_resource_id: *mut u64,
) -> c_int {
    match handle_store::get_ctx(local_id) {
        Some(h) => {
            if !out_server_id.is_null() { *out_server_id = h.server_id; }
            if !out_session_id.is_null() { *out_session_id = h.session_id; }
            if !out_resource_id.is_null() { *out_resource_id = h.resource_id; }
            1
        }
        None => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn rgpu_cuda_resolve_ctx(
    local_id: u64,
    out_server_id: *mut u16,
    out_session_id: *mut u32,
    out_resource_id: *mut u64,
) -> c_int {
    rgpu_common::ffi::catch_panic(0, || rgpu_cuda_resolve_ctx_impl(local_id, out_server_id, out_session_id, out_resource_id))
}

#[allow(non_snake_case)]
unsafe fn rgpu_cuda_resolve_mem_impl(
    local_id: u64,
    out_server_id: *mut u16,
    out_session_id: *mut u32,
    out_resource_id: *mut u64,
) -> c_int {
    match handle_store::get_mem(local_id) {
        Some(h) => {
            if !out_server_id.is_null() { *out_server_id = h.server_id; }
            if !out_session_id.is_null() { *out_session_id = h.session_id; }
            if !out_resource_id.is_null() { *out_resource_id = h.resource_id; }
            1
        }
        None => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn rgpu_cuda_resolve_mem(
    local_id: u64,
    out_server_id: *mut u16,
    out_session_id: *mut u32,
    out_resource_id: *mut u64,
) -> c_int {
    rgpu_common::ffi::catch_panic(0, || rgpu_cuda_resolve_mem_impl(local_id, out_server_id, out_session_id, out_resource_id))
}

// ── Helper Functions ────────────────────────────────────────────────

/// Detect whether a module image is PTX (text) or cubin (binary) and return the data.
pub(crate) unsafe fn detect_and_read_module_image(image: *const c_void) -> Vec<u8> {
    let ptr = image as *const u8;

    // Check for ELF magic (cubin files)
    let elf_magic = [0x7f, b'E', b'L', b'F'];
    let first_four = std::slice::from_raw_parts(ptr, 4);

    if first_four == elf_magic {
        let max_size = 64 * 1024 * 1024; // 64 MB max
        if let Some(size) = parse_elf_size(ptr) {
            return std::slice::from_raw_parts(ptr, size).to_vec();
        }
        return std::slice::from_raw_parts(ptr, max_size).to_vec();
    }

    // Assume PTX (null-terminated string)
    let c_str = std::ffi::CStr::from_ptr(image as *const c_char);
    c_str.to_bytes_with_nul().to_vec()
}

/// Try to parse the total size of an ELF binary.
unsafe fn parse_elf_size(ptr: *const u8) -> Option<usize> {
    if *ptr.add(4) != 2 {
        return None;
    }

    let e_shoff = *(ptr.add(40) as *const u64);
    let e_shentsize = *(ptr.add(58) as *const u16) as u64;
    let e_shnum = *(ptr.add(60) as *const u16) as u64;

    let total = e_shoff + (e_shentsize * e_shnum);
    Some(total as usize)
}
