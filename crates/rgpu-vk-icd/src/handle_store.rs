//! Client-side handle mapping for Vulkan resources.
//! Maps local opaque IDs to NetworkHandles via DashMap.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use dashmap::DashMap;
use rgpu_protocol::handle::NetworkHandle;

static NEXT_ID: AtomicU64 = AtomicU64::new(0x2000);

fn alloc_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

macro_rules! handle_map {
    ($map_name:ident, $fn_map:ident, $fn_store:ident, $fn_get:ident, $fn_remove:ident) => {
        static $map_name: OnceLock<DashMap<u64, NetworkHandle>> = OnceLock::new();

        fn $fn_map() -> &'static DashMap<u64, NetworkHandle> {
            $map_name.get_or_init(DashMap::new)
        }

        pub fn $fn_store(handle: NetworkHandle) -> u64 {
            let id = alloc_id();
            $fn_map().insert(id, handle);
            id
        }

        pub fn $fn_get(id: u64) -> Option<NetworkHandle> {
            $fn_map().get(&id).map(|v| *v)
        }

        pub fn $fn_remove(id: u64) -> Option<NetworkHandle> {
            $fn_map().remove(&id).map(|(_, v)| v)
        }
    };
}

handle_map!(INSTANCE_MAP, instance_map, store_instance, get_instance, remove_instance);
handle_map!(PHYS_DEV_MAP, phys_dev_map, store_physical_device, get_physical_device, remove_physical_device);
handle_map!(DEVICE_MAP, device_map, store_device, get_device, remove_device);
handle_map!(QUEUE_MAP, queue_map, store_queue, get_queue, remove_queue);
handle_map!(MEMORY_MAP, memory_map, store_memory, get_memory, remove_memory);
handle_map!(BUFFER_MAP, buffer_map, store_buffer, get_buffer, remove_buffer);
handle_map!(SHADER_MAP, shader_map, store_shader_module, get_shader_module, remove_shader_module);
handle_map!(DESC_SET_LAYOUT_MAP, desc_set_layout_map, store_desc_set_layout, get_desc_set_layout, remove_desc_set_layout);
handle_map!(PIPELINE_LAYOUT_MAP, pipeline_layout_map, store_pipeline_layout, get_pipeline_layout, remove_pipeline_layout);
handle_map!(PIPELINE_MAP, pipeline_map, store_pipeline, get_pipeline, remove_pipeline);
handle_map!(DESC_POOL_MAP, desc_pool_map, store_desc_pool, get_desc_pool, remove_desc_pool);
handle_map!(DESC_SET_MAP, desc_set_map, store_desc_set, get_desc_set, remove_desc_set);
handle_map!(CMD_POOL_MAP, cmd_pool_map, store_cmd_pool, get_cmd_pool, remove_cmd_pool);
handle_map!(CMD_BUF_MAP, cmd_buf_map, store_cmd_buffer, get_cmd_buffer, remove_cmd_buffer);
handle_map!(FENCE_MAP, fence_map, store_fence, get_fence, remove_fence);
handle_map!(IMAGE_MAP, image_map, store_image, get_image, remove_image);
handle_map!(IMAGE_VIEW_MAP, image_view_map, store_image_view, get_image_view, remove_image_view);
handle_map!(RENDER_PASS_MAP, render_pass_map, store_render_pass, get_render_pass, remove_render_pass);
handle_map!(FRAMEBUFFER_MAP, framebuffer_map, store_framebuffer, get_framebuffer, remove_framebuffer);
handle_map!(SEMAPHORE_MAP, semaphore_map, store_semaphore, get_semaphore, remove_semaphore);
