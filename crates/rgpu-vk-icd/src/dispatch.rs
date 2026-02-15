//! Dispatchable handle management for the Vulkan ICD.
//!
//! The Vulkan loader requires that dispatchable handles (VkInstance, VkDevice,
//! VkQueue, VkCommandBuffer) have their first `sizeof(void*)` bytes point to
//! a dispatch table. The loader writes this after the ICD returns the handle.

/// The ICD loader magic value. The loader expects this in new dispatchable handles.
pub const ICD_LOADER_MAGIC: usize = 0x01CDC0DE;

/// A dispatchable handle wrapper. The Vulkan loader writes its dispatch
/// table pointer into the first pointer-sized slot.
#[repr(C)]
pub struct DispatchableHandle {
    /// The loader will overwrite this with its dispatch table pointer.
    pub loader_data: usize,
    /// Our internal handle identifier (maps to NetworkHandle via handle_store).
    pub local_id: u64,
}

impl DispatchableHandle {
    /// Allocate a new dispatchable handle on the heap.
    pub fn new(local_id: u64) -> *mut Self {
        Box::into_raw(Box::new(Self {
            loader_data: ICD_LOADER_MAGIC,
            local_id,
        }))
    }

    /// Get the local_id from a dispatchable handle pointer.
    ///
    /// # Safety
    /// The pointer must point to a valid DispatchableHandle.
    pub unsafe fn get_id(ptr: *const Self) -> u64 {
        (*ptr).local_id
    }

    /// Free a dispatchable handle.
    ///
    /// # Safety
    /// The pointer must have been created by `DispatchableHandle::new`.
    pub unsafe fn destroy(ptr: *mut Self) {
        drop(Box::from_raw(ptr));
    }
}
