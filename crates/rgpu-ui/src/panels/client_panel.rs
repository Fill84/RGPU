use egui::{Color32, RichText, Ui};

use crate::state::{ServiceOrigin, ServiceStatus, UiState};
use crate::widgets::gpu_card;

/// Render the Client detail panel.
pub fn show(ui: &mut Ui, state: &mut UiState) {
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            show_status_header(ui, state);
            ui.add_space(12.0);
            show_gpu_pool(ui, state);
            ui.add_space(12.0);
            show_remote_servers(ui, state);
            ui.add_space(12.0);
            show_interpose_status(ui, state);
        });
}

// ── Status header + Start/Stop ──────────────────────────────────────────────

fn show_status_header(ui: &mut Ui, state: &mut UiState) {
    ui.heading("Client");
    ui.add_space(4.0);

    // Origin badge
    let origin_text = match &state.client.origin {
        ServiceOrigin::NotDetected => "Not detected",
        ServiceOrigin::WindowsService => "Windows Service",
        ServiceOrigin::ExternalProcess => "External Process (CLI)",
        ServiceOrigin::Embedded => "Embedded (this UI)",
    };
    ui.horizontal(|ui| {
        ui.label("Mode:");
        ui.label(
            RichText::new(origin_text)
                .small()
                .color(Color32::GRAY),
        );
    });

    // IPC path
    ui.horizontal(|ui| {
        ui.label("IPC:");
        ui.label(
            RichText::new(rgpu_common::platform::default_ipc_path())
                .small()
                .color(Color32::GRAY),
        );
    });

    // Status
    let (status_text, status_color) = status_display(&state.client.status);
    ui.horizontal(|ui| {
        ui.label(RichText::new("Status:").strong());
        ui.label(RichText::new(status_text).color(status_color).strong());
    });

    // Error detail
    if let ServiceStatus::Error(ref msg) = state.client.status {
        ui.label(
            RichText::new(format!("Error: {}", msg))
                .color(Color32::from_rgb(255, 100, 100))
                .small(),
        );
    }

    ui.add_space(4.0);

    // Start/Stop button
    let is_transitioning = matches!(
        state.client.status,
        ServiceStatus::Starting | ServiceStatus::Stopping
    );

    match state.client.status {
        ServiceStatus::Running => {
            let btn = ui.add_enabled(
                !is_transitioning,
                egui::Button::new(
                    RichText::new("Stop Client Daemon").color(Color32::from_rgb(255, 100, 100)),
                ),
            );
            if btn.clicked() {
                state.client.stop_requested = true;
            }
        }
        ServiceStatus::Stopped | ServiceStatus::Unknown | ServiceStatus::Error(_) => {
            let btn = ui.add_enabled(
                !is_transitioning,
                egui::Button::new(
                    RichText::new("Start Client Daemon").color(Color32::from_rgb(100, 255, 100)),
                ),
            );
            if btn.clicked() {
                state.client.start_requested = true;
            }
        }
        _ => {
            ui.add_enabled(false, egui::Button::new("Please wait..."));
        }
    }
}

// ── GPU Pool ────────────────────────────────────────────────────────────────

fn show_gpu_pool(ui: &mut Ui, state: &UiState) {
    ui.separator();
    ui.add_space(4.0);
    ui.strong(format!("GPU Pool ({})", state.client.gpu_pool.len()));
    ui.add_space(4.0);

    if !state.client.status.is_running() {
        ui.label(
            RichText::new("Client daemon is not running. Start the daemon to see available GPUs.")
                .color(Color32::GRAY)
                .italics(),
        );
        return;
    }

    if state.client.gpu_pool.is_empty() {
        ui.label(
            RichText::new("No GPUs in pool. Check server connections and configuration.")
                .color(Color32::GRAY)
                .italics(),
        );
        return;
    }

    for (i, gpu) in state.client.gpu_pool.iter().enumerate() {
        let source = if gpu.server_id == 0 {
            "local".to_string()
        } else {
            format!("server {}", gpu.server_id)
        };
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(format!("#{}", i))
                    .small()
                    .color(Color32::from_rgb(100, 180, 255)),
            );
            ui.label(
                RichText::new(format!("({})", source))
                    .small()
                    .color(Color32::GRAY),
            );
        });
        gpu_card::gpu_card(ui, gpu, &source);
        ui.add_space(4.0);
    }
}

// ── Connected remote servers ────────────────────────────────────────────────

fn show_remote_servers(ui: &mut Ui, state: &UiState) {
    ui.separator();
    ui.add_space(4.0);
    ui.strong("Connected Servers");
    ui.add_space(4.0);

    if state.client.remote_servers.is_empty() {
        ui.label(
            RichText::new("No remote servers configured.")
                .color(Color32::GRAY)
                .italics(),
        );
        return;
    }

    for server in &state.client.remote_servers {
        ui.horizontal(|ui| {
            // Connection dot
            let dot_color = if server.connected {
                Color32::from_rgb(100, 200, 100)
            } else {
                Color32::from_rgb(255, 100, 100)
            };
            let (rect, _) =
                ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
            ui.painter()
                .circle_filled(rect.center(), 5.0, dot_color);

            // Address
            ui.label(RichText::new(&server.address).strong());

            // Status text
            let status = if server.connected {
                format!("Connected ({} GPUs)", server.gpu_count)
            } else {
                "Disconnected".to_string()
            };
            ui.label(
                RichText::new(format!("({})", status))
                    .small()
                    .color(Color32::GRAY),
            );

            // Server ID
            if let Some(sid) = server.server_id {
                ui.label(
                    RichText::new(format!("[ID: {}]", sid))
                        .small()
                        .color(Color32::GRAY),
                );
            }
        });
    }
}

// ── Interpose status ────────────────────────────────────────────────────────

fn show_interpose_status(ui: &mut Ui, state: &UiState) {
    ui.separator();
    ui.add_space(4.0);
    ui.strong("Interpose Libraries");
    ui.add_space(4.0);

    let items = [
        ("CUDA", state.client.interpose.cuda),
        ("Vulkan ICD", state.client.interpose.vulkan),
        ("NVENC", state.client.interpose.nvenc),
        ("NVDEC", state.client.interpose.nvdec),
        ("NVML", state.client.interpose.nvml),
    ];

    let all_unknown = items.iter().all(|(_, v)| v.is_none());
    if all_unknown {
        ui.label(
            RichText::new("Interpose status not yet checked.")
                .color(Color32::GRAY)
                .italics(),
        );
        return;
    }

    egui::Grid::new("interpose_grid")
        .num_columns(2)
        .spacing([16.0, 4.0])
        .show(ui, |ui| {
            for (name, status) in &items {
                ui.label(*name);
                match status {
                    Some(true) => {
                        ui.label(
                            RichText::new("Installed")
                                .color(Color32::from_rgb(100, 200, 100)),
                        );
                    }
                    Some(false) => {
                        ui.label(
                            RichText::new("Not installed")
                                .color(Color32::from_rgb(255, 100, 100)),
                        );
                    }
                    None => {
                        ui.label(
                            RichText::new("Unknown")
                                .color(Color32::GRAY),
                        );
                    }
                }
                ui.end_row();
            }
        });
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn status_display(status: &ServiceStatus) -> (&str, Color32) {
    match status {
        ServiceStatus::Unknown => ("Unknown", Color32::from_rgb(150, 150, 150)),
        ServiceStatus::Stopped => ("Stopped", Color32::from_rgb(150, 150, 150)),
        ServiceStatus::Starting => ("Starting...", Color32::from_rgb(255, 200, 50)),
        ServiceStatus::Running => ("Running", Color32::from_rgb(100, 200, 100)),
        ServiceStatus::Stopping => ("Stopping...", Color32::from_rgb(255, 165, 0)),
        ServiceStatus::Error(_) => ("Error", Color32::from_rgb(255, 100, 100)),
    }
}
