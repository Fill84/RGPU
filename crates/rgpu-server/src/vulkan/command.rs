use ash::vk;

use rgpu_protocol::handle::ResourceType;
use rgpu_protocol::vulkan_commands::*;

use crate::session::Session;
use super::VulkanExecutor;

impl VulkanExecutor {
    pub(crate) fn handle_create_command_pool(
        &self,
        session: &Session,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let (device, queue_family_index, flags) = match cmd {
            VulkanCommand::CreateCommandPool {
                device,
                queue_family_index,
                flags,
            } => (device, queue_family_index, flags),
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

        let create_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::from_raw(flags));

        match unsafe { dev.create_command_pool(&create_info, None) } {
            Ok(pool) => {
                let handle = session.alloc_handle(ResourceType::VkCommandPool);
                self.command_pool_handles.insert(handle, pool);
                self.command_pool_to_device.insert(handle, device);
                VulkanResponse::CommandPoolCreated { handle }
            }
            Err(e) => Self::vk_err(e),
        }
    }

    pub(crate) fn handle_destroy_command_pool(
        &self,
        session: &Session,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let (device, command_pool) = match cmd {
            VulkanCommand::DestroyCommandPool {
                device,
                command_pool,
            } => (device, command_pool),
            _ => unreachable!(),
        };

        let dev = match self.device_wrappers.get(&device) {
            Some(d) => d,
            None => return VulkanResponse::Success,
        };
        if let Some((_, pool)) = self.command_pool_handles.remove(&command_pool) {
            unsafe { dev.destroy_command_pool(pool, None) };
            self.command_pool_to_device.remove(&command_pool);
            session.remove_handle(&command_pool);
        }
        VulkanResponse::Success
    }

    pub(crate) fn handle_reset_command_pool(&self, cmd: VulkanCommand) -> VulkanResponse {
        let (device, command_pool, flags) = match cmd {
            VulkanCommand::ResetCommandPool {
                device,
                command_pool,
                flags,
            } => (device, command_pool, flags),
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
        let pool = match self.command_pool_handles.get(&command_pool) {
            Some(p) => *p.value(),
            None => {
                return VulkanResponse::Error {
                    code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                    message: "invalid command pool handle".to_string(),
                }
            }
        };
        match unsafe {
            dev.reset_command_pool(pool, vk::CommandPoolResetFlags::from_raw(flags))
        } {
            Ok(()) => VulkanResponse::Success,
            Err(e) => Self::vk_err(e),
        }
    }

    pub(crate) fn handle_allocate_command_buffers(
        &self,
        session: &Session,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let (device, command_pool, level, count) = match cmd {
            VulkanCommand::AllocateCommandBuffers {
                device,
                command_pool,
                level,
                count,
            } => (device, command_pool, level, count),
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
        let pool = match self.command_pool_handles.get(&command_pool) {
            Some(p) => *p.value(),
            None => {
                return VulkanResponse::Error {
                    code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                    message: "invalid command pool handle".to_string(),
                }
            }
        };

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::from_raw(level as i32))
            .command_buffer_count(count);

        match unsafe { dev.allocate_command_buffers(&alloc_info) } {
            Ok(cmd_bufs) => {
                let mut handles = Vec::new();
                for cb in cmd_bufs {
                    let handle = session.alloc_handle(ResourceType::VkCommandBuffer);
                    self.command_buffer_handles.insert(handle, cb);
                    self.command_buffer_to_device.insert(handle, device);
                    handles.push(handle);
                }
                VulkanResponse::CommandBuffersAllocated { handles }
            }
            Err(e) => Self::vk_err(e),
        }
    }

    pub(crate) fn handle_free_command_buffers(
        &self,
        session: &Session,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let (device, command_pool, command_buffers) = match cmd {
            VulkanCommand::FreeCommandBuffers {
                device,
                command_pool,
                command_buffers,
            } => (device, command_pool, command_buffers),
            _ => unreachable!(),
        };

        let dev = match self.device_wrappers.get(&device) {
            Some(d) => d,
            None => return VulkanResponse::Success,
        };
        let pool = match self.command_pool_handles.get(&command_pool) {
            Some(p) => *p.value(),
            None => return VulkanResponse::Success,
        };

        let vk_bufs: Vec<vk::CommandBuffer> = command_buffers
            .iter()
            .filter_map(|h| {
                self.command_buffer_handles
                    .remove(h)
                    .map(|(_, cb)| {
                        self.command_buffer_to_device.remove(h);
                        session.remove_handle(h);
                        cb
                    })
            })
            .collect();

        if !vk_bufs.is_empty() {
            unsafe { dev.free_command_buffers(pool, &vk_bufs) };
        }
        VulkanResponse::Success
    }

    pub(crate) fn handle_submit_recorded_commands(
        &self,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let (command_buffer, commands) = match cmd {
            VulkanCommand::SubmitRecordedCommands {
                command_buffer,
                commands,
            } => (command_buffer, commands),
            _ => unreachable!(),
        };

        let cb = match self.command_buffer_handles.get(&command_buffer) {
            Some(c) => *c.value(),
            None => {
                return VulkanResponse::Error {
                    code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                    message: "invalid command buffer handle".to_string(),
                }
            }
        };

        // Find the device for this command buffer
        let dev_handle = match self.command_buffer_to_device.get(&command_buffer) {
            Some(d) => *d.value(),
            None => {
                return VulkanResponse::Error {
                    code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                    message: "command buffer has no associated device".to_string(),
                }
            }
        };
        let dev = match self.device_wrappers.get(&dev_handle) {
            Some(d) => d,
            None => {
                return VulkanResponse::Error {
                    code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                    message: "device not found".to_string(),
                }
            }
        };

        // Begin command buffer
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        if let Err(e) = unsafe { dev.begin_command_buffer(cb, &begin_info) } {
            return Self::vk_err(e);
        }

        // Replay all recorded commands
        for recorded_cmd in &commands {
            match recorded_cmd {
                RecordedCommand::BindPipeline {
                    pipeline_bind_point,
                    pipeline,
                } => {
                    let p = match self.pipeline_handles.get(pipeline) {
                        Some(p) => *p.value(),
                        None => continue,
                    };
                    unsafe {
                        dev.cmd_bind_pipeline(
                            cb,
                            vk::PipelineBindPoint::from_raw(*pipeline_bind_point as i32),
                            p,
                        );
                    }
                }

                RecordedCommand::BindDescriptorSets {
                    pipeline_bind_point,
                    layout,
                    first_set,
                    descriptor_sets,
                    dynamic_offsets,
                } => {
                    let pl = match self.pipeline_layout_handles.get(layout) {
                        Some(l) => *l.value(),
                        None => continue,
                    };
                    let sets: Vec<vk::DescriptorSet> = descriptor_sets
                        .iter()
                        .filter_map(|h| {
                            self.desc_set_handles.get(h).map(|v| *v.value())
                        })
                        .collect();
                    unsafe {
                        dev.cmd_bind_descriptor_sets(
                            cb,
                            vk::PipelineBindPoint::from_raw(
                                *pipeline_bind_point as i32,
                            ),
                            pl,
                            *first_set,
                            &sets,
                            dynamic_offsets,
                        );
                    }
                }

                RecordedCommand::Dispatch {
                    group_count_x,
                    group_count_y,
                    group_count_z,
                } => unsafe {
                    dev.cmd_dispatch(
                        cb,
                        *group_count_x,
                        *group_count_y,
                        *group_count_z,
                    );
                },

                RecordedCommand::PipelineBarrier {
                    src_stage_mask,
                    dst_stage_mask,
                    dependency_flags,
                    memory_barriers,
                    buffer_memory_barriers,
                    image_memory_barriers,
                } => {
                    let vk_mem_barriers: Vec<vk::MemoryBarrier> = memory_barriers
                        .iter()
                        .map(|mb| {
                            vk::MemoryBarrier::default()
                                .src_access_mask(vk::AccessFlags::from_raw(
                                    mb.src_access_mask,
                                ))
                                .dst_access_mask(vk::AccessFlags::from_raw(
                                    mb.dst_access_mask,
                                ))
                        })
                        .collect();

                    let vk_buf_barriers: Vec<vk::BufferMemoryBarrier> =
                        buffer_memory_barriers
                            .iter()
                            .map(|bmb| {
                                let buffer = self
                                    .buffer_handles
                                    .get(&bmb.buffer)
                                    .map(|v| *v.value())
                                    .unwrap_or(vk::Buffer::null());
                                vk::BufferMemoryBarrier::default()
                                    .src_access_mask(vk::AccessFlags::from_raw(
                                        bmb.src_access_mask,
                                    ))
                                    .dst_access_mask(vk::AccessFlags::from_raw(
                                        bmb.dst_access_mask,
                                    ))
                                    .src_queue_family_index(bmb.src_queue_family_index)
                                    .dst_queue_family_index(bmb.dst_queue_family_index)
                                    .buffer(buffer)
                                    .offset(bmb.offset)
                                    .size(bmb.size)
                            })
                            .collect();

                    let vk_img_barriers: Vec<vk::ImageMemoryBarrier> =
                        image_memory_barriers
                            .iter()
                            .map(|imb| {
                                let image = self
                                    .image_handles
                                    .get(&imb.image)
                                    .map(|v| *v.value())
                                    .unwrap_or(vk::Image::null());
                                vk::ImageMemoryBarrier::default()
                                    .src_access_mask(vk::AccessFlags::from_raw(
                                        imb.src_access_mask,
                                    ))
                                    .dst_access_mask(vk::AccessFlags::from_raw(
                                        imb.dst_access_mask,
                                    ))
                                    .old_layout(vk::ImageLayout::from_raw(imb.old_layout))
                                    .new_layout(vk::ImageLayout::from_raw(imb.new_layout))
                                    .src_queue_family_index(imb.src_queue_family_index)
                                    .dst_queue_family_index(imb.dst_queue_family_index)
                                    .image(image)
                                    .subresource_range(vk::ImageSubresourceRange {
                                        aspect_mask: vk::ImageAspectFlags::from_raw(
                                            imb.subresource_range.aspect_mask,
                                        ),
                                        base_mip_level: imb.subresource_range.base_mip_level,
                                        level_count: imb.subresource_range.level_count,
                                        base_array_layer: imb
                                            .subresource_range
                                            .base_array_layer,
                                        layer_count: imb.subresource_range.layer_count,
                                    })
                            })
                            .collect();

                    unsafe {
                        dev.cmd_pipeline_barrier(
                            cb,
                            vk::PipelineStageFlags::from_raw(*src_stage_mask),
                            vk::PipelineStageFlags::from_raw(*dst_stage_mask),
                            vk::DependencyFlags::from_raw(*dependency_flags),
                            &vk_mem_barriers,
                            &vk_buf_barriers,
                            &vk_img_barriers,
                        );
                    }
                }

                RecordedCommand::CopyBuffer { src, dst, regions } => {
                    let src_buf = match self.buffer_handles.get(src) {
                        Some(b) => *b.value(),
                        None => continue,
                    };
                    let dst_buf = match self.buffer_handles.get(dst) {
                        Some(b) => *b.value(),
                        None => continue,
                    };
                    let vk_regions: Vec<vk::BufferCopy> = regions
                        .iter()
                        .map(|r| {
                            vk::BufferCopy::default()
                                .src_offset(r.src_offset)
                                .dst_offset(r.dst_offset)
                                .size(r.size)
                        })
                        .collect();
                    unsafe { dev.cmd_copy_buffer(cb, src_buf, dst_buf, &vk_regions) };
                }

                RecordedCommand::FillBuffer {
                    buffer,
                    offset,
                    size,
                    data,
                } => {
                    let buf = match self.buffer_handles.get(buffer) {
                        Some(b) => *b.value(),
                        None => continue,
                    };
                    unsafe { dev.cmd_fill_buffer(cb, buf, *offset, *size, *data) };
                }

                RecordedCommand::UpdateBuffer {
                    buffer,
                    offset,
                    data,
                } => {
                    let buf = match self.buffer_handles.get(buffer) {
                        Some(b) => *b.value(),
                        None => continue,
                    };
                    unsafe { dev.cmd_update_buffer(cb, buf, *offset, data) };
                }

                RecordedCommand::BeginRenderPass {
                    render_pass,
                    framebuffer,
                    render_area,
                    clear_values,
                    contents,
                } => {
                    let rp = match self.render_pass_handles.get(render_pass) {
                        Some(r) => *r.value(),
                        None => continue,
                    };
                    let fb = match self.framebuffer_handles.get(framebuffer) {
                        Some(f) => *f.value(),
                        None => continue,
                    };
                    let vk_clear_values: Vec<vk::ClearValue> = clear_values
                        .iter()
                        .map(|cv| unsafe {
                            std::mem::transmute::<[u8; 16], vk::ClearValue>(cv.data)
                        })
                        .collect();
                    let begin_info = vk::RenderPassBeginInfo::default()
                        .render_pass(rp)
                        .framebuffer(fb)
                        .render_area(vk::Rect2D {
                            offset: vk::Offset2D {
                                x: render_area.offset[0],
                                y: render_area.offset[1],
                            },
                            extent: vk::Extent2D {
                                width: render_area.extent[0],
                                height: render_area.extent[1],
                            },
                        })
                        .clear_values(&vk_clear_values);
                    unsafe {
                        dev.cmd_begin_render_pass(
                            cb,
                            &begin_info,
                            vk::SubpassContents::from_raw(*contents as i32),
                        );
                    }
                }

                RecordedCommand::EndRenderPass => unsafe {
                    dev.cmd_end_render_pass(cb);
                },

                RecordedCommand::Draw {
                    vertex_count,
                    instance_count,
                    first_vertex,
                    first_instance,
                } => unsafe {
                    dev.cmd_draw(
                        cb,
                        *vertex_count,
                        *instance_count,
                        *first_vertex,
                        *first_instance,
                    );
                },

                RecordedCommand::DrawIndexed {
                    index_count,
                    instance_count,
                    first_index,
                    vertex_offset,
                    first_instance,
                } => unsafe {
                    dev.cmd_draw_indexed(
                        cb,
                        *index_count,
                        *instance_count,
                        *first_index,
                        *vertex_offset,
                        *first_instance,
                    );
                },

                RecordedCommand::BindVertexBuffers {
                    first_binding,
                    buffers,
                    offsets,
                } => {
                    let vk_bufs: Vec<vk::Buffer> = buffers
                        .iter()
                        .filter_map(|h| self.buffer_handles.get(h).map(|v| *v.value()))
                        .collect();
                    unsafe {
                        dev.cmd_bind_vertex_buffers(cb, *first_binding, &vk_bufs, offsets);
                    }
                }

                RecordedCommand::BindIndexBuffer {
                    buffer,
                    offset,
                    index_type,
                } => {
                    let buf = match self.buffer_handles.get(buffer) {
                        Some(b) => *b.value(),
                        None => continue,
                    };
                    unsafe {
                        dev.cmd_bind_index_buffer(
                            cb,
                            buf,
                            *offset,
                            vk::IndexType::from_raw(*index_type as i32),
                        );
                    }
                }

                RecordedCommand::SetViewport {
                    first_viewport,
                    viewports,
                } => {
                    let vk_viewports: Vec<vk::Viewport> = viewports
                        .iter()
                        .map(|vp| vk::Viewport {
                            x: vp.x,
                            y: vp.y,
                            width: vp.width,
                            height: vp.height,
                            min_depth: vp.min_depth,
                            max_depth: vp.max_depth,
                        })
                        .collect();
                    unsafe {
                        dev.cmd_set_viewport(cb, *first_viewport, &vk_viewports);
                    }
                }

                RecordedCommand::SetScissor {
                    first_scissor,
                    scissors,
                } => {
                    let vk_scissors: Vec<vk::Rect2D> = scissors
                        .iter()
                        .map(|s| vk::Rect2D {
                            offset: vk::Offset2D {
                                x: s.offset[0],
                                y: s.offset[1],
                            },
                            extent: vk::Extent2D {
                                width: s.extent[0],
                                height: s.extent[1],
                            },
                        })
                        .collect();
                    unsafe {
                        dev.cmd_set_scissor(cb, *first_scissor, &vk_scissors);
                    }
                }

                RecordedCommand::CopyBufferToImage {
                    src_buffer,
                    dst_image,
                    dst_image_layout,
                    regions,
                } => {
                    let buf = match self.buffer_handles.get(src_buffer) {
                        Some(b) => *b.value(),
                        None => continue,
                    };
                    let img = match self.image_handles.get(dst_image) {
                        Some(i) => *i.value(),
                        None => continue,
                    };
                    let vk_regions: Vec<vk::BufferImageCopy> = regions
                        .iter()
                        .map(|r| vk::BufferImageCopy {
                            buffer_offset: r.buffer_offset,
                            buffer_row_length: r.buffer_row_length,
                            buffer_image_height: r.buffer_image_height,
                            image_subresource: vk::ImageSubresourceLayers {
                                aspect_mask: vk::ImageAspectFlags::from_raw(
                                    r.image_subresource.aspect_mask,
                                ),
                                mip_level: r.image_subresource.mip_level,
                                base_array_layer: r.image_subresource.base_array_layer,
                                layer_count: r.image_subresource.layer_count,
                            },
                            image_offset: vk::Offset3D {
                                x: r.image_offset[0],
                                y: r.image_offset[1],
                                z: r.image_offset[2],
                            },
                            image_extent: vk::Extent3D {
                                width: r.image_extent[0],
                                height: r.image_extent[1],
                                depth: r.image_extent[2],
                            },
                        })
                        .collect();
                    unsafe {
                        dev.cmd_copy_buffer_to_image(
                            cb,
                            buf,
                            img,
                            vk::ImageLayout::from_raw(*dst_image_layout),
                            &vk_regions,
                        );
                    }
                }

                RecordedCommand::CopyImageToBuffer {
                    src_image,
                    src_image_layout,
                    dst_buffer,
                    regions,
                } => {
                    let img = match self.image_handles.get(src_image) {
                        Some(i) => *i.value(),
                        None => continue,
                    };
                    let buf = match self.buffer_handles.get(dst_buffer) {
                        Some(b) => *b.value(),
                        None => continue,
                    };
                    let vk_regions: Vec<vk::BufferImageCopy> = regions
                        .iter()
                        .map(|r| vk::BufferImageCopy {
                            buffer_offset: r.buffer_offset,
                            buffer_row_length: r.buffer_row_length,
                            buffer_image_height: r.buffer_image_height,
                            image_subresource: vk::ImageSubresourceLayers {
                                aspect_mask: vk::ImageAspectFlags::from_raw(
                                    r.image_subresource.aspect_mask,
                                ),
                                mip_level: r.image_subresource.mip_level,
                                base_array_layer: r.image_subresource.base_array_layer,
                                layer_count: r.image_subresource.layer_count,
                            },
                            image_offset: vk::Offset3D {
                                x: r.image_offset[0],
                                y: r.image_offset[1],
                                z: r.image_offset[2],
                            },
                            image_extent: vk::Extent3D {
                                width: r.image_extent[0],
                                height: r.image_extent[1],
                                depth: r.image_extent[2],
                            },
                        })
                        .collect();
                    unsafe {
                        dev.cmd_copy_image_to_buffer(
                            cb,
                            img,
                            vk::ImageLayout::from_raw(*src_image_layout),
                            buf,
                            &vk_regions,
                        );
                    }
                }
            }
        }

        // End command buffer
        if let Err(e) = unsafe { dev.end_command_buffer(cb) } {
            return Self::vk_err(e);
        }

        VulkanResponse::Success
    }
}
