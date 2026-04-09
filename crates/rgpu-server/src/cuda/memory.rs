use tracing::debug;

use rgpu_protocol::cuda_commands::CudaResponse;
use rgpu_protocol::handle::{NetworkHandle, ResourceType};

use crate::cuda_driver::{CUDA_ERROR_NOT_SUPPORTED, CUDA_SUCCESS};
use crate::session::Session;

use super::CudaExecutor;

impl CudaExecutor {
    pub(crate) fn handle_mem_alloc(&self, session: &Session, byte_size: u64) -> CudaResponse {
        let d = match self.driver() {
            Ok(d) => d,
            Err(e) => return e,
        };

        match d.mem_alloc(byte_size as usize) {
            Ok(dptr) => {
                let handle = session.alloc_handle(ResourceType::CuDevicePtr);
                self.memory_handles.insert(handle, dptr);
                self.memory_sizes.insert(handle, byte_size);
                debug!(
                    session_id = session.session_id,
                    "MemAlloc({} bytes) -> {:?} (dptr=0x{:x})", byte_size, handle, dptr
                );
                CudaResponse::MemAllocated(handle)
            }
            Err(e) => Self::cuda_err(e),
        }
    }

    pub(crate) fn handle_mem_free(&self, session: &Session, dptr: NetworkHandle) -> CudaResponse {
        let d = match self.driver() {
            Ok(d) => d,
            Err(e) => return e,
        };

        match self.memory_handles.remove(&dptr) {
            Some((_, real_ptr)) => {
                let res = d.mem_free(real_ptr);
                self.memory_sizes.remove(&dptr);
                session.remove_handle(&dptr);
                if res == CUDA_SUCCESS {
                    debug!(
                        session_id = session.session_id,
                        "MemFree({:?})", dptr
                    );
                    CudaResponse::Success
                } else {
                    Self::cuda_err(res)
                }
            }
            None => CudaResponse::Error {
                code: 400,
                message: "invalid memory handle".to_string(),
            },
        }
    }

    pub(crate) fn handle_memcpy_htod(&self, session: &Session, dst: NetworkHandle, src_data: Vec<u8>, byte_count: u64) -> CudaResponse {
        let d = match self.driver() {
            Ok(d) => d,
            Err(e) => return e,
        };

        let real_ptr = match self.memory_handles.get(&dst) {
            Some(p) => *p,
            None => {
                return CudaResponse::Error {
                    code: 400,
                    message: "invalid destination memory handle".to_string(),
                }
            }
        };

        let data = &src_data[..byte_count as usize];
        let res = d.memcpy_htod(real_ptr, data);
        if res == CUDA_SUCCESS {
            debug!(
                session_id = session.session_id,
                "MemcpyHtoD({:?}, {} bytes)", dst, byte_count
            );
            CudaResponse::Success
        } else {
            Self::cuda_err(res)
        }
    }

    pub(crate) fn handle_memcpy_dtoh(&self, session: &Session, src: NetworkHandle, byte_count: u64) -> CudaResponse {
        let d = match self.driver() {
            Ok(d) => d,
            Err(e) => return e,
        };

        let real_ptr = match self.memory_handles.get(&src) {
            Some(p) => *p,
            None => {
                return CudaResponse::Error {
                    code: 400,
                    message: "invalid source memory handle".to_string(),
                }
            }
        };

        let mut buf = vec![0u8; byte_count as usize];
        let res = d.memcpy_dtoh(&mut buf, real_ptr);
        if res == CUDA_SUCCESS {
            debug!(
                session_id = session.session_id,
                "MemcpyDtoH({:?}, {} bytes)", src, byte_count
            );
            CudaResponse::MemoryData(buf)
        } else {
            Self::cuda_err(res)
        }
    }

    pub(crate) fn handle_memcpy_dtod(&self, dst: NetworkHandle, src: NetworkHandle, byte_count: u64) -> CudaResponse {
        let d = match self.driver() {
            Ok(d) => d,
            Err(e) => return e,
        };

        let real_dst = match self.memory_handles.get(&dst) {
            Some(p) => *p,
            None => {
                return CudaResponse::Error {
                    code: 400,
                    message: "invalid destination memory handle".to_string(),
                }
            }
        };
        let real_src = match self.memory_handles.get(&src) {
            Some(p) => *p,
            None => {
                return CudaResponse::Error {
                    code: 400,
                    message: "invalid source memory handle".to_string(),
                }
            }
        };

        let res = d.memcpy_dtod(real_dst, real_src, byte_count as usize);
        if res == CUDA_SUCCESS {
            CudaResponse::Success
        } else {
            Self::cuda_err(res)
        }
    }

    pub(crate) fn handle_memcpy_htod_async(&self, session: &Session, dst: NetworkHandle, src_data: Vec<u8>, byte_count: u64) -> CudaResponse {
        // For async memcpy over network, we use sync version since
        // the network is the bottleneck anyway
        let d = match self.driver() {
            Ok(d) => d,
            Err(e) => return e,
        };

        let real_ptr = match self.memory_handles.get(&dst) {
            Some(p) => *p,
            None => {
                return CudaResponse::Error {
                    code: 400,
                    message: "invalid destination memory handle".to_string(),
                }
            }
        };

        let data = &src_data[..byte_count as usize];
        let res = d.memcpy_htod(real_ptr, data);
        if res == CUDA_SUCCESS {
            debug!(
                session_id = session.session_id,
                "MemcpyHtoDAsync({:?}, {} bytes) [sync]", dst, byte_count
            );
            CudaResponse::Success
        } else {
            Self::cuda_err(res)
        }
    }

    pub(crate) fn handle_memcpy_dtoh_async(&self, session: &Session, src: NetworkHandle, byte_count: u64) -> CudaResponse {
        // Use sync version - network is the bottleneck
        let d = match self.driver() {
            Ok(d) => d,
            Err(e) => return e,
        };

        let real_ptr = match self.memory_handles.get(&src) {
            Some(p) => *p,
            None => {
                return CudaResponse::Error {
                    code: 400,
                    message: "invalid source memory handle".to_string(),
                }
            }
        };

        let mut buf = vec![0u8; byte_count as usize];
        let res = d.memcpy_dtoh(&mut buf, real_ptr);
        if res == CUDA_SUCCESS {
            debug!(
                session_id = session.session_id,
                "MemcpyDtoHAsync({:?}, {} bytes) [sync]", src, byte_count
            );
            CudaResponse::MemoryData(buf)
        } else {
            Self::cuda_err(res)
        }
    }

    pub(crate) fn handle_memcpy_dtod_async(&self, dst: NetworkHandle, src: NetworkHandle, byte_count: u64) -> CudaResponse {
        // Use sync version - network is the bottleneck
        let d = match self.driver() {
            Ok(d) => d,
            Err(e) => return e,
        };

        let real_dst = match self.memory_handles.get(&dst) {
            Some(p) => *p,
            None => {
                return CudaResponse::Error {
                    code: 400,
                    message: "invalid destination memory handle".to_string(),
                }
            }
        };
        let real_src = match self.memory_handles.get(&src) {
            Some(p) => *p,
            None => {
                return CudaResponse::Error {
                    code: 400,
                    message: "invalid source memory handle".to_string(),
                }
            }
        };

        let res = d.memcpy_dtod(real_dst, real_src, byte_count as usize);
        if res == CUDA_SUCCESS {
            CudaResponse::Success
        } else {
            Self::cuda_err(res)
        }
    }

    pub(crate) fn handle_memcpy_2d_htod(
        &self,
        dst: NetworkHandle,
        dst_x_in_bytes: u64,
        dst_y: u64,
        dst_pitch: u64,
        src_data: Vec<u8>,
        width_in_bytes: u64,
        height: u64,
    ) -> CudaResponse {
        let d = match self.driver() {
            Ok(d) => d,
            Err(e) => return e,
        };
        let real_dst = match self.memory_handles.get(&dst) {
            Some(p) => *p,
            None => return CudaResponse::Error { code: 400, message: "invalid dst handle".into() },
        };
        // Row-by-row copy from packed src_data to pitched device memory
        let w = width_in_bytes as usize;
        let h = height as usize;
        for row in 0..h {
            let dst_offset = (dst_y as usize + row) * dst_pitch as usize + dst_x_in_bytes as usize;
            let src_offset = row * w;
            let res = d.memcpy_htod(real_dst + dst_offset as u64, &src_data[src_offset..src_offset + w]);
            if res != CUDA_SUCCESS {
                return Self::cuda_err(res);
            }
        }
        CudaResponse::Success
    }

    pub(crate) fn handle_memcpy_2d_dtoh(
        &self,
        src: NetworkHandle,
        src_x_in_bytes: u64,
        src_y: u64,
        src_pitch: u64,
        width_in_bytes: u64,
        height: u64,
    ) -> CudaResponse {
        let d = match self.driver() {
            Ok(d) => d,
            Err(e) => return e,
        };
        let real_src = match self.memory_handles.get(&src) {
            Some(p) => *p,
            None => return CudaResponse::Error { code: 400, message: "invalid src handle".into() },
        };
        let w = width_in_bytes as usize;
        let h = height as usize;
        let mut packed = vec![0u8; w * h];
        for row in 0..h {
            let src_offset = (src_y as usize + row) * src_pitch as usize + src_x_in_bytes as usize;
            let dst_offset = row * w;
            let res = d.memcpy_dtoh(&mut packed[dst_offset..dst_offset + w], real_src + src_offset as u64);
            if res != CUDA_SUCCESS {
                return Self::cuda_err(res);
            }
        }
        CudaResponse::MemoryData(packed)
    }

    pub(crate) fn handle_memcpy_2d_dtod(
        &self,
        dst: NetworkHandle,
        dst_x_in_bytes: u64,
        dst_y: u64,
        dst_pitch: u64,
        src: NetworkHandle,
        src_x_in_bytes: u64,
        src_y: u64,
        src_pitch: u64,
        width_in_bytes: u64,
        height: u64,
    ) -> CudaResponse {
        let d = match self.driver() {
            Ok(d) => d,
            Err(e) => return e,
        };
        let real_dst = match self.memory_handles.get(&dst) {
            Some(p) => *p,
            None => return CudaResponse::Error { code: 400, message: "invalid dst handle".into() },
        };
        let real_src = match self.memory_handles.get(&src) {
            Some(p) => *p,
            None => return CudaResponse::Error { code: 400, message: "invalid src handle".into() },
        };
        let w = width_in_bytes as usize;
        let h = height as usize;
        for row in 0..h {
            let s = real_src + (src_y as usize + row) as u64 * src_pitch + src_x_in_bytes;
            let d_off = real_dst + (dst_y as usize + row) as u64 * dst_pitch + dst_x_in_bytes;
            let res = d.memcpy_dtod(d_off, s, w);
            if res != CUDA_SUCCESS {
                return Self::cuda_err(res);
            }
        }
        CudaResponse::Success
    }

    pub(crate) fn handle_memset_d8(&self, dst: NetworkHandle, value: u8, count: u64) -> CudaResponse {
        let d = match self.driver() {
            Ok(d) => d,
            Err(e) => return e,
        };

        let real_ptr = match self.memory_handles.get(&dst) {
            Some(p) => *p,
            None => {
                return CudaResponse::Error {
                    code: 400,
                    message: "invalid memory handle".to_string(),
                }
            }
        };

        let res = d.memset_d8(real_ptr, value, count as usize);
        if res == CUDA_SUCCESS {
            CudaResponse::Success
        } else {
            Self::cuda_err(res)
        }
    }

    pub(crate) fn handle_memset_d16(&self, dst: NetworkHandle, value: u16, count: u64) -> CudaResponse {
        let d = match self.driver() {
            Ok(d) => d,
            Err(e) => return e,
        };

        let real_ptr = match self.memory_handles.get(&dst) {
            Some(p) => *p,
            None => {
                return CudaResponse::Error {
                    code: 400,
                    message: "invalid memory handle".to_string(),
                }
            }
        };

        let res = d.memset_d16(real_ptr, value, count as usize);
        if res == CUDA_SUCCESS {
            CudaResponse::Success
        } else {
            Self::cuda_err(res)
        }
    }

    pub(crate) fn handle_memset_d32(&self, dst: NetworkHandle, value: u32, count: u64) -> CudaResponse {
        let d = match self.driver() {
            Ok(d) => d,
            Err(e) => return e,
        };

        let real_ptr = match self.memory_handles.get(&dst) {
            Some(p) => *p,
            None => {
                return CudaResponse::Error {
                    code: 400,
                    message: "invalid memory handle".to_string(),
                }
            }
        };

        let res = d.memset_d32(real_ptr, value, count as usize);
        if res == CUDA_SUCCESS {
            CudaResponse::Success
        } else {
            Self::cuda_err(res)
        }
    }

    pub(crate) fn handle_memset_d8_async(&self, dst: NetworkHandle, value: u8, count: u64) -> CudaResponse {
        // Use sync version - network is the bottleneck
        self.handle_memset_d8(dst, value, count)
    }

    pub(crate) fn handle_memset_d16_async(&self, dst: NetworkHandle, value: u16, count: u64) -> CudaResponse {
        // Use sync version - network is the bottleneck
        self.handle_memset_d16(dst, value, count)
    }

    pub(crate) fn handle_memset_d32_async(&self, dst: NetworkHandle, value: u32, count: u64) -> CudaResponse {
        // Use sync version - network is the bottleneck
        self.handle_memset_d32(dst, value, count)
    }

    pub(crate) fn handle_mem_get_info(&self) -> CudaResponse {
        let d = match self.driver() {
            Ok(d) => d,
            Err(e) => return e,
        };
        match d.mem_get_info() {
            Ok((free, total)) => CudaResponse::MemInfo {
                free: free as u64,
                total: total as u64,
            },
            Err(e) => Self::cuda_err(e),
        }
    }

    pub(crate) fn handle_mem_get_address_range(&self, dptr: NetworkHandle) -> CudaResponse {
        let d = match self.driver() {
            Ok(d) => d,
            Err(e) => return e,
        };
        let real_ptr = match self.memory_handles.get(&dptr) {
            Some(p) => *p,
            None => {
                return CudaResponse::Error {
                    code: 400,
                    message: "invalid memory handle".to_string(),
                }
            }
        };
        match d.mem_get_address_range(real_ptr) {
            Ok((base, size)) => {
                // Find or create handle for base
                let base_handle = {
                    let mut found = None;
                    for entry in self.memory_handles.iter() {
                        if *entry.value() == base {
                            found = Some(*entry.key());
                            break;
                        }
                    }
                    found.unwrap_or(dptr)
                };
                CudaResponse::MemAddressRange {
                    base: base_handle,
                    size: size as u64,
                }
            }
            Err(e) => Self::cuda_err(e),
        }
    }

    pub(crate) fn handle_mem_alloc_host(&self, session: &Session, byte_size: u64) -> CudaResponse {
        let d = match self.driver() {
            Ok(d) => d,
            Err(e) => return e,
        };
        match d.mem_alloc_host(byte_size as usize) {
            Ok(ptr) => {
                let handle = session.alloc_handle(ResourceType::CuHostPtr);
                self.host_memory_handles.insert(handle, ptr);
                debug!(
                    session_id = session.session_id,
                    "MemAllocHost({} bytes) -> {:?}", byte_size, handle
                );
                CudaResponse::HostPtr(handle)
            }
            Err(e) => Self::cuda_err(e),
        }
    }

    pub(crate) fn handle_mem_free_host(&self, session: &Session, ptr: NetworkHandle) -> CudaResponse {
        let d = match self.driver() {
            Ok(d) => d,
            Err(e) => return e,
        };
        match self.host_memory_handles.remove(&ptr) {
            Some((_, real_ptr)) => {
                let res = d.mem_free_host(real_ptr);
                session.remove_handle(&ptr);
                if res == CUDA_SUCCESS {
                    CudaResponse::Success
                } else {
                    Self::cuda_err(res)
                }
            }
            None => CudaResponse::Error {
                code: 400,
                message: "invalid host memory handle".to_string(),
            },
        }
    }

    pub(crate) fn handle_mem_host_alloc(&self, session: &Session, byte_size: u64, flags: u32) -> CudaResponse {
        let d = match self.driver() {
            Ok(d) => d,
            Err(e) => return e,
        };
        match d.mem_host_alloc(byte_size as usize, flags) {
            Ok(ptr) => {
                let handle = session.alloc_handle(ResourceType::CuHostPtr);
                self.host_memory_handles.insert(handle, ptr);
                debug!(
                    session_id = session.session_id,
                    "MemHostAlloc({} bytes, flags={}) -> {:?}", byte_size, flags, handle
                );
                CudaResponse::HostPtr(handle)
            }
            Err(e) => Self::cuda_err(e),
        }
    }

    pub(crate) fn handle_mem_host_get_device_pointer(&self, session: &Session, host_ptr: NetworkHandle, flags: u32) -> CudaResponse {
        let d = match self.driver() {
            Ok(d) => d,
            Err(e) => return e,
        };
        let real_host_ptr = match self.host_memory_handles.get(&host_ptr) {
            Some(p) => *p,
            None => {
                return CudaResponse::Error {
                    code: 400,
                    message: "invalid host memory handle".to_string(),
                }
            }
        };
        match d.mem_host_get_device_pointer(real_host_ptr, flags) {
            Ok(dptr) => {
                let handle = session.alloc_handle(ResourceType::CuDevicePtr);
                self.memory_handles.insert(handle, dptr);
                CudaResponse::HostDevicePtr(handle)
            }
            Err(e) => Self::cuda_err(e),
        }
    }

    pub(crate) fn handle_mem_host_get_flags(&self, host_ptr: NetworkHandle) -> CudaResponse {
        let d = match self.driver() {
            Ok(d) => d,
            Err(e) => return e,
        };
        let real_host_ptr = match self.host_memory_handles.get(&host_ptr) {
            Some(p) => *p,
            None => {
                return CudaResponse::Error {
                    code: 400,
                    message: "invalid host memory handle".to_string(),
                }
            }
        };
        match d.mem_host_get_flags(real_host_ptr) {
            Ok(flags) => CudaResponse::HostFlags(flags),
            Err(e) => Self::cuda_err(e),
        }
    }

    pub(crate) fn handle_mem_alloc_managed(&self, session: &Session, byte_size: u64, flags: u32) -> CudaResponse {
        let d = match self.driver() {
            Ok(d) => d,
            Err(e) => return e,
        };
        match d.mem_alloc_managed(byte_size as usize, flags) {
            Ok(dptr) => {
                let handle = session.alloc_handle(ResourceType::CuDevicePtr);
                self.memory_handles.insert(handle, dptr);
                self.memory_sizes.insert(handle, byte_size);
                debug!(
                    session_id = session.session_id,
                    "MemAllocManaged({} bytes) -> {:?}", byte_size, handle
                );
                CudaResponse::MemAllocated(handle)
            }
            Err(e) => Self::cuda_err(e),
        }
    }

    pub(crate) fn handle_mem_alloc_pitch(&self, session: &Session, width: u64, height: u64, element_size: u32) -> CudaResponse {
        let d = match self.driver() {
            Ok(d) => d,
            Err(e) => return e,
        };
        match d.mem_alloc_pitch(width as usize, height as usize, element_size) {
            Ok((dptr, pitch)) => {
                let handle = session.alloc_handle(ResourceType::CuDevicePtr);
                self.memory_handles.insert(handle, dptr);
                self.memory_sizes.insert(handle, pitch as u64 * height);
                CudaResponse::MemAllocPitch {
                    dptr: handle,
                    pitch: pitch as u64,
                }
            }
            Err(e) => Self::cuda_err(e),
        }
    }

    pub(crate) fn handle_mem_host_register(&self) -> CudaResponse {
        // Cannot register client host memory on the server - not supported over network
        Self::cuda_err(CUDA_ERROR_NOT_SUPPORTED)
    }

    pub(crate) fn handle_mem_host_unregister(&self) -> CudaResponse {
        // Cannot unregister client host memory on the server - not supported over network
        Self::cuda_err(CUDA_ERROR_NOT_SUPPORTED)
    }

    pub(crate) fn handle_mem_prefetch_async(&self) -> CudaResponse {
        // No-op over network - prefetch hints are meaningless remotely
        CudaResponse::Success
    }

    pub(crate) fn handle_mem_advise(&self) -> CudaResponse {
        // No-op over network - memory advice hints are meaningless remotely
        CudaResponse::Success
    }

    pub(crate) fn handle_mem_range_get_attribute(&self) -> CudaResponse {
        // No-op over network - return empty attribute data
        CudaResponse::Success
    }

    pub(crate) fn handle_pointer_get_attribute(&self, attribute: i32, ptr: NetworkHandle) -> CudaResponse {
        let d = match self.driver() {
            Ok(d) => d,
            Err(e) => return e,
        };
        let real_ptr = match self.memory_handles.get(&ptr) {
            Some(p) => *p,
            None => return CudaResponse::Error {
                code: 400,
                message: "invalid memory handle".to_string(),
            },
        };
        match d.pointer_get_attribute(attribute, real_ptr) {
            Ok(val) => CudaResponse::PointerAttribute(val),
            Err(e) => Self::cuda_err(e),
        }
    }

    pub(crate) fn handle_pointer_get_attributes(&self, attributes: Vec<i32>, ptr: NetworkHandle) -> CudaResponse {
        let d = match self.driver() {
            Ok(d) => d,
            Err(e) => return e,
        };
        let real_ptr = match self.memory_handles.get(&ptr) {
            Some(p) => *p,
            None => return CudaResponse::Error {
                code: 400,
                message: "invalid memory handle".to_string(),
            },
        };
        let mut results = Vec::with_capacity(attributes.len());
        for attr in &attributes {
            match d.pointer_get_attribute(*attr, real_ptr) {
                Ok(val) => results.push(val),
                Err(_) => results.push(0),
            }
        }
        CudaResponse::PointerAttributes(results)
    }

    pub(crate) fn handle_pointer_set_attribute(&self, attribute: i32, ptr: NetworkHandle, value: u64) -> CudaResponse {
        let d = match self.driver() {
            Ok(d) => d,
            Err(e) => return e,
        };
        let real_ptr = match self.memory_handles.get(&ptr) {
            Some(p) => *p,
            None => return CudaResponse::Error {
                code: 400,
                message: "invalid memory handle".to_string(),
            },
        };
        let res = d.pointer_set_attribute(attribute, real_ptr, value);
        if res == CUDA_SUCCESS {
            CudaResponse::Success
        } else {
            Self::cuda_err(res)
        }
    }

    pub(crate) fn handle_mem_pool_create(&self, session: &Session, device: NetworkHandle) -> CudaResponse {
        // MemPoolCreate requires a CUmemPoolProps struct which is complex
        // to construct over the network. Use the default pool instead.
        let d = match self.driver() {
            Ok(d) => d,
            Err(e) => return e,
        };
        let real_dev = match self.device_handles.get(&device) {
            Some(dev) => *dev,
            None => return CudaResponse::Error {
                code: 101,
                message: "invalid device handle".to_string(),
            },
        };
        // Fall back to getting the default mem pool since creating custom pools
        // requires full CUmemPoolProps serialization
        match d.device_get_default_mem_pool(real_dev) {
            Ok(pool) => {
                let handle = session.alloc_handle(ResourceType::CuMemPool);
                self.mempool_handles.insert(handle, pool);
                CudaResponse::MemPool(handle)
            }
            Err(e) => Self::cuda_err(e),
        }
    }

    pub(crate) fn handle_mem_pool_destroy(&self, session: &Session, pool: NetworkHandle) -> CudaResponse {
        let d = match self.driver() {
            Ok(d) => d,
            Err(e) => return e,
        };
        match self.mempool_handles.remove(&pool) {
            Some((_, real_pool)) => {
                let res = d.mem_pool_destroy(real_pool);
                session.remove_handle(&pool);
                if res == CUDA_SUCCESS {
                    CudaResponse::Success
                } else {
                    Self::cuda_err(res)
                }
            }
            None => CudaResponse::Error {
                code: 400,
                message: "invalid mempool handle".to_string(),
            },
        }
    }

    pub(crate) fn handle_mem_pool_trim_to(&self, pool: NetworkHandle, min_bytes_to_keep: u64) -> CudaResponse {
        let d = match self.driver() {
            Ok(d) => d,
            Err(e) => return e,
        };
        let real_pool = match self.mempool_handles.get(&pool) {
            Some(p) => *p,
            None => return CudaResponse::Error {
                code: 400,
                message: "invalid mempool handle".to_string(),
            },
        };
        let res = d.mem_pool_trim_to(real_pool, min_bytes_to_keep as usize);
        if res == CUDA_SUCCESS {
            CudaResponse::Success
        } else {
            Self::cuda_err(res)
        }
    }

    pub(crate) fn handle_mem_pool_set_attribute(&self, pool: NetworkHandle) -> CudaResponse {
        // MemPoolSetAttribute requires raw pointer manipulation
        // Return not supported for now - most applications work without it
        let _real_pool = match self.mempool_handles.get(&pool) {
            Some(p) => *p,
            None => return CudaResponse::Error {
                code: 400,
                message: "invalid mempool handle".to_string(),
            },
        };
        // No-op success - most pool attributes are hints
        CudaResponse::Success
    }

    pub(crate) fn handle_mem_pool_get_attribute(&self, pool: NetworkHandle) -> CudaResponse {
        let _real_pool = match self.mempool_handles.get(&pool) {
            Some(p) => *p,
            None => return CudaResponse::Error {
                code: 400,
                message: "invalid mempool handle".to_string(),
            },
        };
        // Return 0 as default attribute value
        CudaResponse::MemPoolAttribute(0)
    }

    pub(crate) fn handle_mem_alloc_async(&self, session: &Session, byte_size: u64, stream: NetworkHandle) -> CudaResponse {
        let d = match self.driver() {
            Ok(d) => d,
            Err(e) => return e,
        };
        let real_stream = self
            .stream_handles
            .get(&stream)
            .map(|s| *s)
            .unwrap_or(std::ptr::null_mut());
        match d.mem_alloc_async(byte_size as usize, real_stream) {
            Ok(dptr) => {
                let handle = session.alloc_handle(ResourceType::CuDevicePtr);
                self.memory_handles.insert(handle, dptr);
                self.memory_sizes.insert(handle, byte_size);
                debug!(
                    session_id = session.session_id,
                    "MemAllocAsync({} bytes) -> {:?}", byte_size, handle
                );
                CudaResponse::MemAllocated(handle)
            }
            Err(e) => Self::cuda_err(e),
        }
    }

    pub(crate) fn handle_mem_free_async(&self, session: &Session, dptr: NetworkHandle, stream: NetworkHandle) -> CudaResponse {
        let d = match self.driver() {
            Ok(d) => d,
            Err(e) => return e,
        };
        let real_stream = self
            .stream_handles
            .get(&stream)
            .map(|s| *s)
            .unwrap_or(std::ptr::null_mut());
        match self.memory_handles.remove(&dptr) {
            Some((_, real_ptr)) => {
                let res = d.mem_free_async(real_ptr, real_stream);
                self.memory_sizes.remove(&dptr);
                session.remove_handle(&dptr);
                if res == CUDA_SUCCESS {
                    CudaResponse::Success
                } else {
                    Self::cuda_err(res)
                }
            }
            None => CudaResponse::Error {
                code: 400,
                message: "invalid memory handle".to_string(),
            },
        }
    }

    pub(crate) fn handle_mem_alloc_from_pool_async(&self, session: &Session, byte_size: u64, pool: NetworkHandle, stream: NetworkHandle) -> CudaResponse {
        let d = match self.driver() {
            Ok(d) => d,
            Err(e) => return e,
        };
        let real_pool = match self.mempool_handles.get(&pool) {
            Some(p) => *p,
            None => return CudaResponse::Error {
                code: 400,
                message: "invalid mempool handle".to_string(),
            },
        };
        let real_stream = self
            .stream_handles
            .get(&stream)
            .map(|s| *s)
            .unwrap_or(std::ptr::null_mut());
        match d.mem_alloc_from_pool_async(byte_size as usize, real_pool, real_stream) {
            Ok(dptr) => {
                let handle = session.alloc_handle(ResourceType::CuDevicePtr);
                self.memory_handles.insert(handle, dptr);
                self.memory_sizes.insert(handle, byte_size);
                debug!(
                    session_id = session.session_id,
                    "MemAllocFromPoolAsync({} bytes) -> {:?}", byte_size, handle
                );
                CudaResponse::MemAllocated(handle)
            }
            Err(e) => Self::cuda_err(e),
        }
    }
}
