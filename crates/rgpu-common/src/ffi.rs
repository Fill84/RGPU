//! FFI safety utilities for interpose libraries.
//!
//! Panics crossing FFI boundaries are undefined behavior in Rust.
//! All `extern "C"` functions in interpose crates must catch panics
//! and convert them to appropriate error codes.

use std::panic::{catch_unwind, AssertUnwindSafe};

/// Wraps a closure in `catch_unwind` to prevent panics from crossing FFI boundaries.
///
/// Returns `error_value` if the closure panics.
///
/// # Usage
/// ```ignore
/// #[no_mangle]
/// pub unsafe extern "C" fn cuInit(flags: u32) -> i32 {
///     rgpu_common::ffi::catch_panic(CUDA_ERROR_UNKNOWN, || {
///         // function body
///     })
/// }
/// ```
#[inline]
pub fn catch_panic<F, R>(error_value: R, f: F) -> R
where
    F: FnOnce() -> R,
{
    catch_unwind(AssertUnwindSafe(f)).unwrap_or(error_value)
}

/// Wraps a closure in `catch_unwind` for FFI functions that return `Option`.
///
/// Returns `None` if the closure panics.
#[inline]
pub fn catch_panic_option<F, R>(f: F) -> Option<R>
where
    F: FnOnce() -> Option<R>,
{
    catch_unwind(AssertUnwindSafe(f)).unwrap_or(None)
}

/// Validates a raw pointer is non-null before dereferencing.
///
/// # Safety
/// The pointer must point to valid memory if non-null.
#[inline]
pub unsafe fn validate_ptr<T>(ptr: *const T) -> Option<&'static T> {
    if ptr.is_null() {
        None
    } else {
        Some(&*ptr)
    }
}

/// Validates a mutable raw pointer is non-null before dereferencing.
///
/// # Safety
/// The pointer must point to valid, mutable memory if non-null.
#[inline]
pub unsafe fn validate_ptr_mut<T>(ptr: *mut T) -> Option<&'static mut T> {
    if ptr.is_null() {
        None
    } else {
        Some(&mut *ptr)
    }
}

/// Safely converts a C string pointer to a Rust string slice.
///
/// Returns `None` if the pointer is null or the string is not valid UTF-8.
///
/// # Safety
/// The pointer must point to a null-terminated C string if non-null.
#[inline]
pub unsafe fn safe_cstr_to_str(ptr: *const std::ffi::c_char) -> Option<&'static str> {
    if ptr.is_null() {
        return None;
    }
    std::ffi::CStr::from_ptr(ptr).to_str().ok()
}

/// Safely converts a C string pointer to a `CStr`.
///
/// Returns `None` if the pointer is null.
///
/// # Safety
/// The pointer must point to a null-terminated C string if non-null.
#[inline]
pub unsafe fn safe_cstr(ptr: *const std::ffi::c_char) -> Option<&'static std::ffi::CStr> {
    if ptr.is_null() {
        None
    } else {
        Some(std::ffi::CStr::from_ptr(ptr))
    }
}
