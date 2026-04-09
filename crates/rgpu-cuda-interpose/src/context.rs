//! CUDA Context Management API functions.

use std::ffi::{c_int, c_uint};
use rgpu_protocol::cuda_commands::{CudaCommand, CudaResponse};
use rgpu_protocol::handle::NetworkHandle;
use tracing::debug;

use crate::{
    CUresult, CUdevice, CUcontext,
    CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_UNKNOWN,
    CTX_STACK, send_cuda_command, handle_store,
    ctx_stack_push, ctx_stack_pop, ctx_stack_top, sync_server_context,
};

// ── Context Management ──────────────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn cuCtxCreate_v2_impl(
    pctx: *mut CUcontext,
    flags: c_uint,
    dev: CUdevice,
) -> CUresult {
    if pctx.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }

    debug!("cuCtxCreate_v2(flags={}, dev={})", flags, dev);

    let dev_handle = match handle_store::get_device(dev as u64) {
        Some(h) => h,
        None => return CUDA_ERROR_INVALID_VALUE,
    };

    match send_cuda_command(CudaCommand::CtxCreate {
        flags: flags as u32,
        device: dev_handle,
    }) {
        CudaResponse::Context(handle) => {
            let local_id = handle_store::store_ctx(handle);
            // cuCtxCreate pushes the new context onto the thread's context stack
            ctx_stack_push(local_id);
            *pctx = local_id as CUcontext;
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuCtxCreate_v2(
    pctx: *mut CUcontext,
    flags: c_uint,
    dev: CUdevice,
) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuCtxCreate_v2_impl(pctx, flags, dev))
}

#[allow(non_snake_case)]
unsafe fn cuCtxDestroy_v2_impl(ctx: CUcontext) -> CUresult {
    let local_id = ctx as u64;
    let net_handle = match handle_store::get_ctx(local_id) {
        Some(h) => h,
        None => {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    match send_cuda_command(CudaCommand::CtxDestroy { ctx: net_handle }) {
        CudaResponse::Success => {
            handle_store::remove_ctx(local_id);
            // Remove from context stack if present
            CTX_STACK.with(|s| {
                let mut stack = s.borrow_mut();
                stack.retain(|&id| id != local_id);
            });
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuCtxDestroy_v2(ctx: CUcontext) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuCtxDestroy_v2_impl(ctx))
}

#[allow(non_snake_case)]
unsafe fn cuCtxSetCurrent_impl(ctx: CUcontext) -> CUresult {
    let local_id = ctx as u64;

    if ctx.is_null() {
        // Setting null context — pop the stack top without removing lower entries
        CTX_STACK.with(|s| {
            let mut stack = s.borrow_mut();
            if !stack.is_empty() {
                let len = stack.len();
                stack[len - 1] = 0; // mark top as null
            }
        });
        let _ = send_cuda_command(CudaCommand::CtxSetCurrent { ctx: NetworkHandle::null() });
        return CUDA_SUCCESS;
    }

    let net_handle = match handle_store::get_ctx(local_id) {
        Some(h) => h,
        None => {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    // Replace the top of the context stack
    CTX_STACK.with(|s| {
        let mut stack = s.borrow_mut();
        if let Some(last) = stack.last_mut() {
            *last = local_id;
        } else {
            stack.push(local_id);
        }
    });

    match send_cuda_command(CudaCommand::CtxSetCurrent { ctx: net_handle }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuCtxSetCurrent_impl(ctx))
}

#[allow(non_snake_case)]
unsafe fn cuCtxGetCurrent_impl(pctx: *mut CUcontext) -> CUresult {
    if pctx.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }

    // Return from local context stack — no IPC round-trip needed
    match ctx_stack_top() {
        Some(id) if id != 0 => {
            *pctx = id as CUcontext;
        }
        _ => {
            *pctx = std::ptr::null_mut();
        }
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cuCtxGetCurrent(pctx: *mut CUcontext) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuCtxGetCurrent_impl(pctx))
}

#[allow(non_snake_case)]
unsafe fn cuCtxSynchronize_impl() -> CUresult {
    match send_cuda_command(CudaCommand::CtxSynchronize) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuCtxSynchronize() -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuCtxSynchronize_impl())
}

#[allow(non_snake_case)]
unsafe fn cuCtxPushCurrent_v2_impl(ctx: CUcontext) -> CUresult {
    let local_id = ctx as u64;
    let net_h = match handle_store::get_ctx(local_id) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    // Push onto local context stack
    ctx_stack_push(local_id);
    // Tell the server to set this as the current context
    match send_cuda_command(CudaCommand::CtxSetCurrent { ctx: net_h }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuCtxPushCurrent_v2(ctx: CUcontext) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuCtxPushCurrent_v2_impl(ctx))
}

#[allow(non_snake_case)]
unsafe fn cuCtxPopCurrent_v2_impl(pctx: *mut CUcontext) -> CUresult {
    // Pop from local context stack
    let popped_id = match ctx_stack_pop() {
        Some(id) => id,
        None => return CUDA_ERROR_INVALID_CONTEXT,
    };

    if !pctx.is_null() {
        if popped_id != 0 {
            *pctx = popped_id as CUcontext;
        } else {
            *pctx = std::ptr::null_mut();
        }
    }

    // Tell the server about the new current context (the new top of stack)
    sync_server_context(ctx_stack_top().filter(|id| *id != 0));
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cuCtxPopCurrent_v2(pctx: *mut CUcontext) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuCtxPopCurrent_v2_impl(pctx))
}

#[allow(non_snake_case)]
unsafe fn cuCtxGetDevice_impl(device: *mut CUdevice) -> CUresult {
    if device.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    match send_cuda_command(CudaCommand::CtxGetDevice) {
        CudaResponse::ContextDevice(handle) => {
            let id = handle_store::store_device(handle);
            *device = id as CUdevice;
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuCtxGetDevice(device: *mut CUdevice) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuCtxGetDevice_impl(device))
}

#[allow(non_snake_case)]
unsafe fn cuCtxSetCacheConfig_impl(config: c_int) -> CUresult {
    match send_cuda_command(CudaCommand::CtxSetCacheConfig { config }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuCtxSetCacheConfig(config: c_int) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuCtxSetCacheConfig_impl(config))
}

#[allow(non_snake_case)]
unsafe fn cuCtxGetCacheConfig_impl(config: *mut c_int) -> CUresult {
    if config.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    match send_cuda_command(CudaCommand::CtxGetCacheConfig) {
        CudaResponse::CacheConfig(c) => { *config = c; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuCtxGetCacheConfig(config: *mut c_int) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuCtxGetCacheConfig_impl(config))
}

#[allow(non_snake_case)]
unsafe fn cuCtxSetLimit_impl(limit: c_int, value: usize) -> CUresult {
    match send_cuda_command(CudaCommand::CtxSetLimit { limit, value: value as u64 }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuCtxSetLimit(limit: c_int, value: usize) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuCtxSetLimit_impl(limit, value))
}

#[allow(non_snake_case)]
unsafe fn cuCtxGetLimit_impl(pvalue: *mut usize, limit: c_int) -> CUresult {
    if pvalue.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    match send_cuda_command(CudaCommand::CtxGetLimit { limit }) {
        CudaResponse::ContextLimit(v) => { *pvalue = v as usize; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuCtxGetLimit(pvalue: *mut usize, limit: c_int) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuCtxGetLimit_impl(pvalue, limit))
}

#[allow(non_snake_case)]
unsafe fn cuCtxGetStreamPriorityRange_impl(least: *mut c_int, greatest: *mut c_int) -> CUresult {
    match send_cuda_command(CudaCommand::CtxGetStreamPriorityRange) {
        CudaResponse::StreamPriorityRange { least: l, greatest: g } => {
            if !least.is_null() { *least = l; }
            if !greatest.is_null() { *greatest = g; }
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuCtxGetStreamPriorityRange(least: *mut c_int, greatest: *mut c_int) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuCtxGetStreamPriorityRange_impl(least, greatest))
}

#[allow(non_snake_case)]
unsafe fn cuCtxGetApiVersion_impl(ctx: CUcontext, version: *mut c_uint) -> CUresult {
    if version.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let net_h = match handle_store::get_ctx(ctx as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::CtxGetApiVersion { ctx: net_h }) {
        CudaResponse::ContextApiVersion(v) => { *version = v; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuCtxGetApiVersion(ctx: CUcontext, version: *mut c_uint) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuCtxGetApiVersion_impl(ctx, version))
}

#[allow(non_snake_case)]
unsafe fn cuCtxGetFlags_impl(flags: *mut c_uint) -> CUresult {
    if flags.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    match send_cuda_command(CudaCommand::CtxGetFlags) {
        CudaResponse::ContextFlags(f) => { *flags = f; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuCtxGetFlags(flags: *mut c_uint) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuCtxGetFlags_impl(flags))
}

#[allow(non_snake_case)]
unsafe fn cuCtxSetFlags_impl(flags: c_uint) -> CUresult {
    match send_cuda_command(CudaCommand::CtxSetFlags { flags: flags as u32 }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuCtxSetFlags(flags: c_uint) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuCtxSetFlags_impl(flags))
}

#[allow(non_snake_case)]
unsafe fn cuCtxResetPersistingL2Cache_impl() -> CUresult {
    match send_cuda_command(CudaCommand::CtxResetPersistingL2Cache) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuCtxResetPersistingL2Cache() -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuCtxResetPersistingL2Cache_impl())
}

// ── Peer Access ─────────────────────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn cuCtxEnablePeerAccess_impl(peer_ctx: CUcontext, flags: c_uint) -> CUresult {
    let net_h = match handle_store::get_ctx(peer_ctx as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::CtxEnablePeerAccess { peer_ctx: net_h, flags: flags as u32 }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuCtxEnablePeerAccess(peer_ctx: CUcontext, flags: c_uint) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuCtxEnablePeerAccess_impl(peer_ctx, flags))
}

#[allow(non_snake_case)]
unsafe fn cuCtxDisablePeerAccess_impl(peer_ctx: CUcontext) -> CUresult {
    let net_h = match handle_store::get_ctx(peer_ctx as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::CtxDisablePeerAccess { peer_ctx: net_h }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuCtxDisablePeerAccess(peer_ctx: CUcontext) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuCtxDisablePeerAccess_impl(peer_ctx))
}

// ── Primary Context ─────────────────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn cuDevicePrimaryCtxRetain_impl(pctx: *mut CUcontext, dev: CUdevice) -> CUresult {
    if pctx.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let dev_h = match handle_store::get_device(dev as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::DevicePrimaryCtxRetain { device: dev_h }) {
        CudaResponse::Context(handle) => { let id = handle_store::store_ctx(handle); *pctx = id as CUcontext; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuDevicePrimaryCtxRetain(pctx: *mut CUcontext, dev: CUdevice) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuDevicePrimaryCtxRetain_impl(pctx, dev))
}

#[allow(non_snake_case)]
unsafe fn cuDevicePrimaryCtxRelease_v2_impl(dev: CUdevice) -> CUresult {
    let dev_h = match handle_store::get_device(dev as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::DevicePrimaryCtxRelease { device: dev_h }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuDevicePrimaryCtxRelease_v2(dev: CUdevice) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuDevicePrimaryCtxRelease_v2_impl(dev))
}

#[allow(non_snake_case)]
unsafe fn cuDevicePrimaryCtxReset_v2_impl(dev: CUdevice) -> CUresult {
    let dev_h = match handle_store::get_device(dev as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::DevicePrimaryCtxReset { device: dev_h }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuDevicePrimaryCtxReset_v2(dev: CUdevice) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuDevicePrimaryCtxReset_v2_impl(dev))
}

#[allow(non_snake_case)]
unsafe fn cuDevicePrimaryCtxGetState_impl(dev: CUdevice, flags: *mut c_uint, active: *mut c_int) -> CUresult {
    if flags.is_null() || active.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let dev_h = match handle_store::get_device(dev as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::DevicePrimaryCtxGetState { device: dev_h }) {
        CudaResponse::PrimaryCtxState { flags: f, active: a } => { *flags = f; *active = if a { 1 } else { 0 }; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuDevicePrimaryCtxGetState(dev: CUdevice, flags: *mut c_uint, active: *mut c_int) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuDevicePrimaryCtxGetState_impl(dev, flags, active))
}

#[allow(non_snake_case)]
unsafe fn cuDevicePrimaryCtxSetFlags_v2_impl(dev: CUdevice, flags: c_uint) -> CUresult {
    let dev_h = match handle_store::get_device(dev as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::DevicePrimaryCtxSetFlags { device: dev_h, flags: flags as u32 }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuDevicePrimaryCtxSetFlags_v2(dev: CUdevice, flags: c_uint) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuDevicePrimaryCtxSetFlags_v2_impl(dev, flags))
}

// ── Unversioned Export Aliases ──────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn cuDevicePrimaryCtxRelease_impl(dev: CUdevice) -> CUresult {
    cuDevicePrimaryCtxRelease_v2(dev)
}

#[no_mangle]
pub unsafe extern "C" fn cuDevicePrimaryCtxRelease(dev: CUdevice) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuDevicePrimaryCtxRelease_impl(dev))
}

#[allow(non_snake_case)]
unsafe fn cuDevicePrimaryCtxReset_impl(dev: CUdevice) -> CUresult {
    cuDevicePrimaryCtxReset_v2(dev)
}

#[no_mangle]
pub unsafe extern "C" fn cuDevicePrimaryCtxReset(dev: CUdevice) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuDevicePrimaryCtxReset_impl(dev))
}

#[allow(non_snake_case)]
unsafe fn cuDevicePrimaryCtxSetFlags_impl(dev: CUdevice, flags: c_uint) -> CUresult {
    cuDevicePrimaryCtxSetFlags_v2(dev, flags)
}

#[no_mangle]
pub unsafe extern "C" fn cuDevicePrimaryCtxSetFlags(dev: CUdevice, flags: c_uint) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuDevicePrimaryCtxSetFlags_impl(dev, flags))
}

#[allow(non_snake_case)]
unsafe fn cuCtxCreate_impl(pctx: *mut CUcontext, flags: c_uint, dev: CUdevice) -> CUresult {
    cuCtxCreate_v2(pctx, flags, dev)
}

#[no_mangle]
pub unsafe extern "C" fn cuCtxCreate(pctx: *mut CUcontext, flags: c_uint, dev: CUdevice) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuCtxCreate_impl(pctx, flags, dev))
}

#[allow(non_snake_case)]
unsafe fn cuCtxDestroy_impl(ctx: CUcontext) -> CUresult {
    cuCtxDestroy_v2(ctx)
}

#[no_mangle]
pub unsafe extern "C" fn cuCtxDestroy(ctx: CUcontext) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuCtxDestroy_impl(ctx))
}

#[allow(non_snake_case)]
unsafe fn cuCtxPushCurrent_impl(ctx: CUcontext) -> CUresult {
    cuCtxPushCurrent_v2(ctx)
}

#[no_mangle]
pub unsafe extern "C" fn cuCtxPushCurrent(ctx: CUcontext) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuCtxPushCurrent_impl(ctx))
}

#[allow(non_snake_case)]
unsafe fn cuCtxPopCurrent_impl(pctx: *mut CUcontext) -> CUresult {
    cuCtxPopCurrent_v2(pctx)
}

#[no_mangle]
pub unsafe extern "C" fn cuCtxPopCurrent(pctx: *mut CUcontext) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuCtxPopCurrent_impl(pctx))
}
