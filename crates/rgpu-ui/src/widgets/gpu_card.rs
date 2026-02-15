use egui::{Color32, RichText, Ui};
use rgpu_protocol::gpu_info::{GpuDeviceType, GpuInfo};

/// Render a single GPU info card.
pub fn gpu_card(ui: &mut Ui, gpu: &GpuInfo, server_address: &str) {
    egui::Frame::group(ui.style())
        .inner_margin(egui::Margin::same(8))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.strong(&gpu.device_name);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    device_type_badge(ui, &gpu.device_type);
                });
            });

            ui.add_space(4.0);

            egui::Grid::new(format!("gpu_{}_{}", server_address, gpu.server_device_index))
                .num_columns(2)
                .spacing([16.0, 4.0])
                .show(ui, |ui| {
                    ui.label("VRAM:");
                    ui.label(format_memory(gpu.total_memory));
                    ui.end_row();

                    ui.label("CUDA:");
                    if gpu.supports_cuda {
                        let cc = gpu
                            .cuda_compute_capability
                            .map(|(maj, min)| format!("Yes (SM {}.{})", maj, min))
                            .unwrap_or_else(|| "Yes".to_string());
                        ui.label(RichText::new(cc).color(Color32::from_rgb(100, 200, 100)));
                    } else {
                        ui.label(RichText::new("No").color(Color32::GRAY));
                    }
                    ui.end_row();

                    ui.label("Vulkan:");
                    if gpu.supports_vulkan {
                        let ver = gpu
                            .vulkan_api_version
                            .map(|v| {
                                format!(
                                    "Yes (v{}.{}.{})",
                                    v >> 22,
                                    (v >> 12) & 0x3FF,
                                    v & 0xFFF
                                )
                            })
                            .unwrap_or_else(|| "Yes".to_string());
                        ui.label(RichText::new(ver).color(Color32::from_rgb(100, 200, 100)));
                    } else {
                        ui.label(RichText::new("No").color(Color32::GRAY));
                    }
                    ui.end_row();

                    ui.label("Memory Heaps:");
                    ui.label(format!("{}", gpu.memory_heaps.len()));
                    ui.end_row();

                    ui.label("Queue Families:");
                    ui.label(format!("{}", gpu.queue_family_count));
                    ui.end_row();

                    ui.label("Server:");
                    ui.label(format!(
                        "{} (ID: {})",
                        server_address, gpu.server_id
                    ));
                    ui.end_row();
                });
        });
}

fn device_type_badge(ui: &mut Ui, device_type: &GpuDeviceType) {
    let (text, color) = match device_type {
        GpuDeviceType::DiscreteGpu => ("Discrete", Color32::from_rgb(100, 180, 255)),
        GpuDeviceType::IntegratedGpu => ("Integrated", Color32::from_rgb(255, 180, 100)),
        GpuDeviceType::VirtualGpu => ("Virtual", Color32::from_rgb(180, 100, 255)),
        GpuDeviceType::Cpu => ("CPU", Color32::GRAY),
        GpuDeviceType::Other => ("Other", Color32::GRAY),
    };
    ui.label(
        RichText::new(text)
            .color(color)
            .small(),
    );
}

fn format_memory(bytes: u64) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.1} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else {
        format!("{} MB", bytes / (1024 * 1024))
    }
}
