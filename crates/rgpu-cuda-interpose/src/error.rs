//! CUDA error string functions (client-side only, no IPC needed).

use std::ffi::{c_char, c_int, CStr};

type CUresult = c_int;
const CUDA_SUCCESS: CUresult = 0;
const CUDA_ERROR_INVALID_VALUE: CUresult = 1;

macro_rules! cstr {
    ($s:literal) => {
        unsafe { CStr::from_bytes_with_nul_unchecked(concat!($s, "\0").as_bytes()).as_ptr() }
    };
}

/// Return a null-terminated C string pointer for the error name.
fn error_name_ptr(error: CUresult) -> *const c_char {
    match error {
        0 => cstr!("CUDA_SUCCESS"),
        1 => cstr!("CUDA_ERROR_INVALID_VALUE"),
        2 => cstr!("CUDA_ERROR_OUT_OF_MEMORY"),
        3 => cstr!("CUDA_ERROR_NOT_INITIALIZED"),
        4 => cstr!("CUDA_ERROR_DEINITIALIZED"),
        5 => cstr!("CUDA_ERROR_PROFILER_DISABLED"),
        6 => cstr!("CUDA_ERROR_PROFILER_NOT_INITIALIZED"),
        7 => cstr!("CUDA_ERROR_PROFILER_ALREADY_STARTED"),
        8 => cstr!("CUDA_ERROR_PROFILER_ALREADY_STOPPED"),
        34 => cstr!("CUDA_ERROR_STUB_LIBRARY"),
        46 => cstr!("CUDA_ERROR_DEVICE_UNAVAILABLE"),
        100 => cstr!("CUDA_ERROR_NO_DEVICE"),
        101 => cstr!("CUDA_ERROR_INVALID_DEVICE"),
        102 => cstr!("CUDA_ERROR_DEVICE_NOT_LICENSED"),
        200 => cstr!("CUDA_ERROR_INVALID_IMAGE"),
        201 => cstr!("CUDA_ERROR_INVALID_CONTEXT"),
        202 => cstr!("CUDA_ERROR_CONTEXT_ALREADY_CURRENT"),
        205 => cstr!("CUDA_ERROR_MAP_FAILED"),
        206 => cstr!("CUDA_ERROR_UNMAP_FAILED"),
        207 => cstr!("CUDA_ERROR_ARRAY_IS_MAPPED"),
        208 => cstr!("CUDA_ERROR_ALREADY_MAPPED"),
        209 => cstr!("CUDA_ERROR_NO_BINARY_FOR_GPU"),
        210 => cstr!("CUDA_ERROR_ALREADY_ACQUIRED"),
        211 => cstr!("CUDA_ERROR_NOT_MAPPED"),
        212 => cstr!("CUDA_ERROR_NOT_MAPPED_AS_ARRAY"),
        213 => cstr!("CUDA_ERROR_NOT_MAPPED_AS_POINTER"),
        214 => cstr!("CUDA_ERROR_ECC_UNCORRECTABLE"),
        215 => cstr!("CUDA_ERROR_UNSUPPORTED_LIMIT"),
        216 => cstr!("CUDA_ERROR_CONTEXT_ALREADY_IN_USE"),
        217 => cstr!("CUDA_ERROR_PEER_ACCESS_UNSUPPORTED"),
        218 => cstr!("CUDA_ERROR_INVALID_PTX"),
        219 => cstr!("CUDA_ERROR_INVALID_GRAPHICS_CONTEXT"),
        220 => cstr!("CUDA_ERROR_NVLINK_UNCORRECTABLE"),
        221 => cstr!("CUDA_ERROR_JIT_COMPILER_NOT_FOUND"),
        222 => cstr!("CUDA_ERROR_UNSUPPORTED_PTX_VERSION"),
        223 => cstr!("CUDA_ERROR_JIT_COMPILATION_DISABLED"),
        224 => cstr!("CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY"),
        300 => cstr!("CUDA_ERROR_INVALID_SOURCE"),
        301 => cstr!("CUDA_ERROR_FILE_NOT_FOUND"),
        302 => cstr!("CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND"),
        303 => cstr!("CUDA_ERROR_SHARED_OBJECT_INIT_FAILED"),
        304 => cstr!("CUDA_ERROR_OPERATING_SYSTEM"),
        400 => cstr!("CUDA_ERROR_INVALID_HANDLE"),
        401 => cstr!("CUDA_ERROR_ILLEGAL_STATE"),
        500 => cstr!("CUDA_ERROR_NOT_FOUND"),
        600 => cstr!("CUDA_ERROR_NOT_READY"),
        700 => cstr!("CUDA_ERROR_ILLEGAL_ADDRESS"),
        701 => cstr!("CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES"),
        702 => cstr!("CUDA_ERROR_LAUNCH_TIMEOUT"),
        703 => cstr!("CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING"),
        704 => cstr!("CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED"),
        705 => cstr!("CUDA_ERROR_PEER_ACCESS_NOT_ENABLED"),
        708 => cstr!("CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE"),
        709 => cstr!("CUDA_ERROR_CONTEXT_IS_DESTROYED"),
        710 => cstr!("CUDA_ERROR_ASSERT"),
        711 => cstr!("CUDA_ERROR_TOO_MANY_PEERS"),
        712 => cstr!("CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED"),
        713 => cstr!("CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED"),
        714 => cstr!("CUDA_ERROR_HARDWARE_STACK_ERROR"),
        715 => cstr!("CUDA_ERROR_ILLEGAL_INSTRUCTION"),
        716 => cstr!("CUDA_ERROR_MISALIGNED_ADDRESS"),
        717 => cstr!("CUDA_ERROR_INVALID_ADDRESS_SPACE"),
        718 => cstr!("CUDA_ERROR_INVALID_PC"),
        719 => cstr!("CUDA_ERROR_LAUNCH_FAILED"),
        720 => cstr!("CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE"),
        800 => cstr!("CUDA_ERROR_NOT_PERMITTED"),
        801 => cstr!("CUDA_ERROR_NOT_SUPPORTED"),
        802 => cstr!("CUDA_ERROR_SYSTEM_NOT_READY"),
        803 => cstr!("CUDA_ERROR_SYSTEM_DRIVER_MISMATCH"),
        804 => cstr!("CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE"),
        805 => cstr!("CUDA_ERROR_MPS_CONNECTION_FAILED"),
        806 => cstr!("CUDA_ERROR_MPS_RPC_FAILURE"),
        807 => cstr!("CUDA_ERROR_MPS_SERVER_NOT_READY"),
        808 => cstr!("CUDA_ERROR_MPS_MAX_CLIENTS_REACHED"),
        809 => cstr!("CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED"),
        810 => cstr!("CUDA_ERROR_MPS_CLIENT_TERMINATED"),
        900 => cstr!("CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED"),
        901 => cstr!("CUDA_ERROR_STREAM_CAPTURE_INVALIDATED"),
        902 => cstr!("CUDA_ERROR_STREAM_CAPTURE_MERGE"),
        903 => cstr!("CUDA_ERROR_STREAM_CAPTURE_UNMATCHED"),
        904 => cstr!("CUDA_ERROR_STREAM_CAPTURE_UNJOINED"),
        905 => cstr!("CUDA_ERROR_STREAM_CAPTURE_ISOLATION"),
        906 => cstr!("CUDA_ERROR_STREAM_CAPTURE_IMPLICIT"),
        907 => cstr!("CUDA_ERROR_CAPTURED_EVENT"),
        908 => cstr!("CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD"),
        999 => cstr!("CUDA_ERROR_UNKNOWN"),
        _ => cstr!("CUDA_ERROR_UNKNOWN"),
    }
}

/// Return a null-terminated C string pointer for the error description.
fn error_string_ptr(error: CUresult) -> *const c_char {
    match error {
        0 => cstr!("no error"),
        1 => cstr!("invalid argument"),
        2 => cstr!("out of memory"),
        3 => cstr!("driver not initialized"),
        4 => cstr!("driver deinitialized"),
        100 => cstr!("no CUDA-capable device is detected"),
        101 => cstr!("invalid device ordinal"),
        200 => cstr!("device kernel image is invalid"),
        201 => cstr!("invalid context"),
        209 => cstr!("no kernel image is available for execution on the device"),
        300 => cstr!("invalid source"),
        301 => cstr!("file not found"),
        400 => cstr!("invalid resource handle"),
        401 => cstr!("an illegal state was encountered"),
        500 => cstr!("named symbol not found"),
        600 => cstr!("not ready"),
        700 => cstr!("an illegal memory access was encountered"),
        701 => cstr!("too many resources requested for launch"),
        702 => cstr!("the launch timed out and was terminated"),
        719 => cstr!("unspecified launch failure"),
        800 => cstr!("operation not permitted"),
        801 => cstr!("operation not supported"),
        999 => cstr!("unknown error"),
        _ => cstr!("unknown error"),
    }
}

#[no_mangle]
pub unsafe extern "C" fn cuGetErrorString(
    error: CUresult,
    p_str: *mut *const c_char,
) -> CUresult {
    if p_str.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *p_str = error_string_ptr(error);
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cuGetErrorName(
    error: CUresult,
    p_str: *mut *const c_char,
) -> CUresult {
    if p_str.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *p_str = error_name_ptr(error);
    CUDA_SUCCESS
}
