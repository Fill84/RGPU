use ash::vk;

use rgpu_protocol::handle::ResourceType;
use rgpu_protocol::vulkan_commands::*;

use crate::session::Session;
use super::VulkanExecutor;

impl VulkanExecutor {
    pub(crate) fn handle_create_descriptor_pool(
        &self,
        session: &Session,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let (device, max_sets, pool_sizes, flags) = match cmd {
            VulkanCommand::CreateDescriptorPool {
                device,
                max_sets,
                pool_sizes,
                flags,
            } => (device, max_sets, pool_sizes, flags),
            _ => unreachable!(),
        };

        let dev = match self.device_wrappers.get(&device) {
            Some(d) => d,
            None => {
                return VulkanResponse::Error {
                    code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                    message: "invalid device handle".to_string(),
                }
            }
        };

        let vk_pool_sizes: Vec<vk::DescriptorPoolSize> = pool_sizes
            .iter()
            .map(|ps| {
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::from_raw(ps.descriptor_type))
                    .descriptor_count(ps.descriptor_count)
            })
            .collect();

        let create_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(max_sets)
            .pool_sizes(&vk_pool_sizes)
            .flags(vk::DescriptorPoolCreateFlags::from_raw(flags));

        match unsafe { dev.create_descriptor_pool(&create_info, None) } {
            Ok(pool) => {
                let handle = session.alloc_handle(ResourceType::VkDescriptorPool);
                self.desc_pool_handles.insert(handle, pool);
                self.desc_pool_to_device.insert(handle, device);
                VulkanResponse::DescriptorPoolCreated { handle }
            }
            Err(e) => Self::vk_err(e),
        }
    }

    pub(crate) fn handle_destroy_descriptor_pool(
        &self,
        session: &Session,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let (device, pool) = match cmd {
            VulkanCommand::DestroyDescriptorPool { device, pool } => (device, pool),
            _ => unreachable!(),
        };

        let dev = match self.device_wrappers.get(&device) {
            Some(d) => d,
            None => return VulkanResponse::Success,
        };
        if let Some((_, p)) = self.desc_pool_handles.remove(&pool) {
            unsafe { dev.destroy_descriptor_pool(p, None) };
            self.desc_pool_to_device.remove(&pool);
            session.remove_handle(&pool);
        }
        VulkanResponse::Success
    }

    pub(crate) fn handle_allocate_descriptor_sets(
        &self,
        session: &Session,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let (device, descriptor_pool, set_layouts) = match cmd {
            VulkanCommand::AllocateDescriptorSets {
                device,
                descriptor_pool,
                set_layouts,
            } => (device, descriptor_pool, set_layouts),
            _ => unreachable!(),
        };

        let dev = match self.device_wrappers.get(&device) {
            Some(d) => d,
            None => {
                return VulkanResponse::Error {
                    code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                    message: "invalid device handle".to_string(),
                }
            }
        };
        let pool = match self.desc_pool_handles.get(&descriptor_pool) {
            Some(p) => *p.value(),
            None => {
                return VulkanResponse::Error {
                    code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                    message: "invalid descriptor pool handle".to_string(),
                }
            }
        };

        let vk_layouts: Vec<vk::DescriptorSetLayout> = set_layouts
            .iter()
            .filter_map(|h| {
                self.desc_set_layout_handles
                    .get(h)
                    .map(|v| *v.value())
            })
            .collect();

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&vk_layouts);

        match unsafe { dev.allocate_descriptor_sets(&alloc_info) } {
            Ok(sets) => {
                let mut handles = Vec::new();
                for set in sets {
                    let handle = session.alloc_handle(ResourceType::VkDescriptorSet);
                    self.desc_set_handles.insert(handle, set);
                    handles.push(handle);
                }
                VulkanResponse::DescriptorSetsAllocated { handles }
            }
            Err(e) => Self::vk_err(e),
        }
    }

    pub(crate) fn handle_free_descriptor_sets(
        &self,
        session: &Session,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let (device, descriptor_pool, descriptor_sets) = match cmd {
            VulkanCommand::FreeDescriptorSets {
                device,
                descriptor_pool,
                descriptor_sets,
            } => (device, descriptor_pool, descriptor_sets),
            _ => unreachable!(),
        };

        let dev = match self.device_wrappers.get(&device) {
            Some(d) => d,
            None => return VulkanResponse::Success,
        };
        let pool = match self.desc_pool_handles.get(&descriptor_pool) {
            Some(p) => *p.value(),
            None => return VulkanResponse::Success,
        };

        let vk_sets: Vec<vk::DescriptorSet> = descriptor_sets
            .iter()
            .filter_map(|h| {
                self.desc_set_handles
                    .remove(h)
                    .map(|(_, s)| {
                        session.remove_handle(h);
                        s
                    })
            })
            .collect();

        if !vk_sets.is_empty() {
            let _ = unsafe { dev.free_descriptor_sets(pool, &vk_sets) };
        }
        VulkanResponse::Success
    }

    pub(crate) fn handle_update_descriptor_sets(
        &self,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let (device, writes) = match cmd {
            VulkanCommand::UpdateDescriptorSets { device, writes } => (device, writes),
            _ => unreachable!(),
        };

        let dev = match self.device_wrappers.get(&device) {
            Some(d) => d,
            None => {
                return VulkanResponse::Error {
                    code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                    message: "invalid device handle".to_string(),
                }
            }
        };

        // Build buffer info arrays (must live long enough)
        let buffer_info_vecs: Vec<Vec<vk::DescriptorBufferInfo>> = writes
            .iter()
            .map(|w| {
                w.buffer_infos
                    .iter()
                    .map(|bi| {
                        let buffer = self
                            .buffer_handles
                            .get(&bi.buffer)
                            .map(|v| *v.value())
                            .unwrap_or(vk::Buffer::null());
                        vk::DescriptorBufferInfo::default()
                            .buffer(buffer)
                            .offset(bi.offset)
                            .range(bi.range)
                    })
                    .collect()
            })
            .collect();

        let vk_writes: Vec<vk::WriteDescriptorSet> = writes
            .iter()
            .enumerate()
            .map(|(i, w)| {
                let dst_set = self
                    .desc_set_handles
                    .get(&w.dst_set)
                    .map(|v| *v.value())
                    .unwrap_or(vk::DescriptorSet::null());

                vk::WriteDescriptorSet::default()
                    .dst_set(dst_set)
                    .dst_binding(w.dst_binding)
                    .dst_array_element(w.dst_array_element)
                    .descriptor_type(vk::DescriptorType::from_raw(w.descriptor_type))
                    .buffer_info(&buffer_info_vecs[i])
            })
            .collect();

        unsafe { dev.update_descriptor_sets(&vk_writes, &[]) };
        VulkanResponse::Success
    }
}
