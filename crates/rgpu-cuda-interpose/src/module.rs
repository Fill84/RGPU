//! CUDA Module and Linker Management API functions.

use std::ffi::{c_char, c_int, c_uint, c_void};
use tracing::debug;
use rgpu_protocol::cuda_commands::{CudaCommand, CudaResponse};

use crate::{
    CUresult, CUmodule, CUdeviceptr, CUlinkState,
    CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_UNKNOWN,
    send_cuda_command, handle_store,
    detect_and_read_module_image,
};

// ── Module Management ───────────────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn cuModuleLoadData_impl(
    module: *mut CUmodule,
    image: *const c_void,
) -> CUresult {
    if module.is_null() || image.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }

    let image_data = detect_and_read_module_image(image);

    debug!("cuModuleLoadData({} bytes)", image_data.len());

    match send_cuda_command(CudaCommand::ModuleLoadData { image: image_data }) {
        CudaResponse::Module(handle) => {
            let local_id = handle_store::store_mod(handle);
            *module = local_id as CUmodule;
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuModuleLoadData(
    module: *mut CUmodule,
    image: *const c_void,
) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuModuleLoadData_impl(module, image))
}

#[allow(non_snake_case)]
unsafe fn cuModuleUnload_impl(hmod: CUmodule) -> CUresult {
    let local_id = hmod as u64;
    let net_handle = match handle_store::get_mod(local_id) {
        Some(h) => h,
        None => {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    match send_cuda_command(CudaCommand::ModuleUnload { module: net_handle }) {
        CudaResponse::Success => {
            handle_store::remove_mod(local_id);
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuModuleUnload(hmod: CUmodule) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuModuleUnload_impl(hmod))
}

#[allow(non_snake_case)]
unsafe fn cuModuleGetFunction_impl(
    hfunc: *mut crate::CUfunction,
    hmod: CUmodule,
    name: *const c_char,
) -> CUresult {
    if hfunc.is_null() || name.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }

    let local_mod_id = hmod as u64;
    let net_module = match handle_store::get_mod(local_mod_id) {
        Some(h) => h,
        None => {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    let func_name = std::ffi::CStr::from_ptr(name).to_string_lossy().into_owned();
    debug!("cuModuleGetFunction('{}')", func_name);

    match send_cuda_command(CudaCommand::ModuleGetFunction {
        module: net_module,
        name: func_name,
    }) {
        CudaResponse::Function(handle) => {
            let local_id = handle_store::store_func(handle);
            *hfunc = local_id as crate::CUfunction;
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuModuleGetFunction(
    hfunc: *mut crate::CUfunction,
    hmod: CUmodule,
    name: *const c_char,
) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuModuleGetFunction_impl(hfunc, hmod, name))
}

// ── Module Management Extended ──────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn cuModuleLoad_impl(module: *mut CUmodule, fname: *const c_char) -> CUresult {
    if module.is_null() || fname.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let name = std::ffi::CStr::from_ptr(fname).to_string_lossy().into_owned();
    match send_cuda_command(CudaCommand::ModuleLoad { fname: name }) {
        CudaResponse::Module(handle) => { let id = handle_store::store_mod(handle); *module = id as CUmodule; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuModuleLoad(module: *mut CUmodule, fname: *const c_char) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuModuleLoad_impl(module, fname))
}

#[allow(non_snake_case)]
unsafe fn cuModuleLoadDataEx_impl(
    module: *mut CUmodule, image: *const c_void,
    _num_options: c_uint, _options: *mut c_int, _option_values: *mut *mut c_void,
) -> CUresult {
    if module.is_null() || image.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let image_data = detect_and_read_module_image(image);
    match send_cuda_command(CudaCommand::ModuleLoadDataEx { image: image_data, num_options: 0, options: vec![], option_values: vec![] }) {
        CudaResponse::Module(handle) => { let id = handle_store::store_mod(handle); *module = id as CUmodule; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuModuleLoadDataEx(
    module: *mut CUmodule, image: *const c_void,
    _num_options: c_uint, _options: *mut c_int, _option_values: *mut *mut c_void,
) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuModuleLoadDataEx_impl(module, image, _num_options, _options, _option_values))
}

#[allow(non_snake_case)]
unsafe fn cuModuleLoadFatBinary_impl(module: *mut CUmodule, fat_cubin: *const c_void) -> CUresult {
    if module.is_null() || fat_cubin.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let image_data = detect_and_read_module_image(fat_cubin);
    match send_cuda_command(CudaCommand::ModuleLoadFatBinary { fat_cubin: image_data }) {
        CudaResponse::Module(handle) => { let id = handle_store::store_mod(handle); *module = id as CUmodule; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuModuleLoadFatBinary(module: *mut CUmodule, fat_cubin: *const c_void) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuModuleLoadFatBinary_impl(module, fat_cubin))
}

#[allow(non_snake_case)]
unsafe fn cuModuleGetGlobal_v2_impl(dptr: *mut CUdeviceptr, bytes: *mut usize, hmod: CUmodule, name: *const c_char) -> CUresult {
    if name.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let net_mod = match handle_store::get_mod(hmod as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    let func_name = std::ffi::CStr::from_ptr(name).to_string_lossy().into_owned();
    match send_cuda_command(CudaCommand::ModuleGetGlobal { module: net_mod, name: func_name }) {
        CudaResponse::GlobalPtr { ptr, size } => {
            if !dptr.is_null() { let id = handle_store::store_mem(ptr); *dptr = id; }
            if !bytes.is_null() { *bytes = size as usize; }
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuModuleGetGlobal_v2(dptr: *mut CUdeviceptr, bytes: *mut usize, hmod: CUmodule, name: *const c_char) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuModuleGetGlobal_v2_impl(dptr, bytes, hmod, name))
}

// ── Linker ──────────────────────────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn cuLinkCreate_v2_impl(
    _num_options: c_uint, _options: *mut c_int, _option_values: *mut *mut c_void,
    state: *mut CUlinkState,
) -> CUresult {
    if state.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    match send_cuda_command(CudaCommand::LinkCreate { num_options: 0, options: vec![], option_values: vec![] }) {
        CudaResponse::Linker(handle) => { let id = handle_store::store_linker(handle); *state = id as CUlinkState; CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuLinkCreate_v2(
    _num_options: c_uint, _options: *mut c_int, _option_values: *mut *mut c_void,
    state: *mut CUlinkState,
) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuLinkCreate_v2_impl(_num_options, _options, _option_values, state))
}

#[allow(non_snake_case)]
unsafe fn cuLinkAddData_v2_impl(
    state: CUlinkState, jit_type: c_int, data: *mut c_void, size: usize,
    name: *const c_char, _num_options: c_uint, _options: *mut c_int, _option_values: *mut *mut c_void,
) -> CUresult {
    let net_link = match handle_store::get_linker(state as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    let data_vec = if !data.is_null() && size > 0 {
        std::slice::from_raw_parts(data as *const u8, size).to_vec()
    } else { vec![] };
    let name_str = if !name.is_null() { std::ffi::CStr::from_ptr(name).to_string_lossy().into_owned() } else { String::new() };
    match send_cuda_command(CudaCommand::LinkAddData { link: net_link, jit_type, data: data_vec, name: name_str, num_options: 0, options: vec![], option_values: vec![] }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuLinkAddData_v2(
    state: CUlinkState, jit_type: c_int, data: *mut c_void, size: usize,
    name: *const c_char, _num_options: c_uint, _options: *mut c_int, _option_values: *mut *mut c_void,
) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuLinkAddData_v2_impl(state, jit_type, data, size, name, _num_options, _options, _option_values))
}

#[allow(non_snake_case)]
unsafe fn cuLinkAddFile_v2_impl(
    state: CUlinkState, jit_type: c_int, path: *const c_char,
    _num_options: c_uint, _options: *mut c_int, _option_values: *mut *mut c_void,
) -> CUresult {
    if path.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let net_link = match handle_store::get_linker(state as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    let path_str = std::ffi::CStr::from_ptr(path).to_string_lossy().into_owned();
    match send_cuda_command(CudaCommand::LinkAddFile { link: net_link, jit_type, path: path_str, num_options: 0, options: vec![], option_values: vec![] }) {
        CudaResponse::Success => CUDA_SUCCESS,
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuLinkAddFile_v2(
    state: CUlinkState, jit_type: c_int, path: *const c_char,
    _num_options: c_uint, _options: *mut c_int, _option_values: *mut *mut c_void,
) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuLinkAddFile_v2_impl(state, jit_type, path, _num_options, _options, _option_values))
}

#[allow(non_snake_case)]
unsafe fn cuLinkComplete_impl(state: CUlinkState, cubin_out: *mut *mut c_void, size_out: *mut usize) -> CUresult {
    let net_link = match handle_store::get_linker(state as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::LinkComplete { link: net_link }) {
        CudaResponse::LinkCompleted { cubin_data } => {
            // Leak the data so the pointer stays valid
            let boxed = cubin_data.into_boxed_slice();
            let len = boxed.len();
            let ptr = Box::into_raw(boxed) as *mut c_void;
            if !cubin_out.is_null() { *cubin_out = ptr; }
            if !size_out.is_null() { *size_out = len; }
            CUDA_SUCCESS
        }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuLinkComplete(state: CUlinkState, cubin_out: *mut *mut c_void, size_out: *mut usize) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuLinkComplete_impl(state, cubin_out, size_out))
}

#[allow(non_snake_case)]
unsafe fn cuLinkDestroy_impl(state: CUlinkState) -> CUresult {
    let net_link = match handle_store::get_linker(state as u64) { Some(h) => h, None => return CUDA_ERROR_INVALID_VALUE };
    match send_cuda_command(CudaCommand::LinkDestroy { link: net_link }) {
        CudaResponse::Success => { handle_store::remove_linker(state as u64); CUDA_SUCCESS }
        CudaResponse::Error { code, .. } => code,
        _ => CUDA_ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuLinkDestroy(state: CUlinkState) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuLinkDestroy_impl(state))
}

// ── Unversioned Export Aliases ──────────────────────────────────────

#[allow(non_snake_case)]
unsafe fn cuModuleGetGlobal_impl(dptr: *mut CUdeviceptr, bytes: *mut usize, hmod: CUmodule, name: *const c_char) -> CUresult {
    cuModuleGetGlobal_v2(dptr, bytes, hmod, name)
}

#[no_mangle]
pub unsafe extern "C" fn cuModuleGetGlobal(dptr: *mut CUdeviceptr, bytes: *mut usize, hmod: CUmodule, name: *const c_char) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuModuleGetGlobal_impl(dptr, bytes, hmod, name))
}

#[allow(non_snake_case)]
unsafe fn cuLinkCreate_impl(num_options: c_uint, option_keys: *mut c_int, option_values: *mut *mut c_void, state_out: *mut CUlinkState) -> CUresult {
    cuLinkCreate_v2(num_options, option_keys, option_values, state_out)
}

#[no_mangle]
pub unsafe extern "C" fn cuLinkCreate(num_options: c_uint, option_keys: *mut c_int, option_values: *mut *mut c_void, state_out: *mut CUlinkState) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuLinkCreate_impl(num_options, option_keys, option_values, state_out))
}

#[allow(non_snake_case)]
unsafe fn cuLinkAddData_impl(state: CUlinkState, jit_type: c_int, data: *mut c_void, size: usize, name: *const c_char, num_options: c_uint, options: *mut c_int, option_values: *mut *mut c_void) -> CUresult {
    cuLinkAddData_v2(state, jit_type, data, size, name, num_options, options, option_values)
}

#[no_mangle]
pub unsafe extern "C" fn cuLinkAddData(state: CUlinkState, jit_type: c_int, data: *mut c_void, size: usize, name: *const c_char, num_options: c_uint, options: *mut c_int, option_values: *mut *mut c_void) -> CUresult {
    rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || cuLinkAddData_impl(state, jit_type, data, size, name, num_options, options, option_values))
}
