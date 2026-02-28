use egui::{Color32, RichText, Ui};

use crate::state::{ServiceStatus, UiState, UiTab};

/// Render the dashboard (home) panel — overview of both server and client roles.
pub fn show(ui: &mut Ui, state: &mut UiState) {
    ui.heading("RGPU Dashboard");
    ui.add_space(8.0);

    // Two side-by-side cards
    ui.columns(2, |cols| {
        // --- Server Role Card ---
        server_role_card(&mut cols[0], state);

        // --- Client Role Card ---
        client_role_card(&mut cols[1], state);
    });

    ui.add_space(16.0);
    ui.separator();
    ui.add_space(8.0);

    // Quick Actions
    ui.strong("Quick Actions");
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        let server_running = state.server.status.is_running();
        let client_running = state.client.status.is_running();

        if server_running {
            if ui.button("Stop Server").clicked() {
                state.server.stop_requested = true;
            }
        } else {
            if ui.button("Start Server").clicked() {
                state.server.start_requested = true;
            }
        }

        if client_running {
            if ui.button("Stop Client").clicked() {
                state.client.stop_requested = true;
            }
        } else {
            if ui.button("Start Client").clicked() {
                state.client.start_requested = true;
            }
        }

        if ui.button("Open Config").clicked() {
            state.active_tab = UiTab::ConfigEditor;
        }
    });
}

fn server_role_card(ui: &mut Ui, state: &mut UiState) {
    egui::Frame::group(ui.style())
        .inner_margin(egui::Margin::same(12))
        .show(ui, |ui| {
            ui.strong("Server Role");
            ui.add_space(8.0);

            // Status
            let (status_text, status_color) = status_display(&state.server.status);
            ui.horizontal(|ui| {
                ui.label("Status:");
                ui.label(RichText::new(status_text).color(status_color));
            });

            // Origin
            ui.horizontal(|ui| {
                ui.label("Mode:");
                ui.label(
                    RichText::new(format!("{:?}", state.server.origin))
                        .small()
                        .color(Color32::GRAY),
                );
            });

            // Port
            ui.horizontal(|ui| {
                ui.label("Port:");
                ui.label(format!("{}", state.server.config.port));
            });

            // GPUs
            ui.horizontal(|ui| {
                ui.label("GPUs:");
                ui.label(format!("{}", state.server.served_gpus.len()));
            });

            // Connected clients
            if state.server.status.is_running() {
                ui.horizontal(|ui| {
                    ui.label("Clients:");
                    ui.label(format!("{}", state.server.connected_clients));
                });

                // Uptime
                if let Some(ref metrics) = state.server.metrics {
                    ui.horizontal(|ui| {
                        ui.label("Uptime:");
                        ui.label(format_uptime(metrics.uptime_secs));
                    });
                }
            }

            ui.add_space(8.0);
            if ui
                .button(RichText::new("Go to Server >").color(Color32::from_rgb(100, 180, 255)))
                .clicked()
            {
                state.active_tab = UiTab::Server;
            }
        });
}

fn client_role_card(ui: &mut Ui, state: &mut UiState) {
    egui::Frame::group(ui.style())
        .inner_margin(egui::Margin::same(12))
        .show(ui, |ui| {
            ui.strong("Client Role");
            ui.add_space(8.0);

            // Status
            let (status_text, status_color) = status_display(&state.client.status);
            ui.horizontal(|ui| {
                ui.label("Status:");
                ui.label(RichText::new(status_text).color(status_color));
            });

            // Origin
            ui.horizontal(|ui| {
                ui.label("Mode:");
                ui.label(
                    RichText::new(format!("{:?}", state.client.origin))
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

            // GPU Pool
            ui.horizontal(|ui| {
                ui.label("GPU Pool:");
                ui.label(format!("{}", state.client.gpu_pool.len()));
            });

            // Connected servers
            if !state.client.remote_servers.is_empty() {
                let connected = state.client.remote_servers.iter().filter(|s| s.connected).count();
                let total = state.client.remote_servers.len();
                ui.horizontal(|ui| {
                    ui.label("Servers:");
                    ui.label(format!("{}/{} connected", connected, total));
                });
            }

            // Interpose status
            let installed_count = [
                state.client.interpose.cuda,
                state.client.interpose.vulkan,
                state.client.interpose.nvenc,
                state.client.interpose.nvdec,
                state.client.interpose.nvml,
            ]
            .iter()
            .filter(|v| **v == Some(true))
            .count();
            let checked_count = [
                state.client.interpose.cuda,
                state.client.interpose.vulkan,
                state.client.interpose.nvenc,
                state.client.interpose.nvdec,
                state.client.interpose.nvml,
            ]
            .iter()
            .filter(|v| v.is_some())
            .count();
            if checked_count > 0 {
                ui.horizontal(|ui| {
                    ui.label("Interpose:");
                    ui.label(format!("{}/{} installed", installed_count, 5));
                });
            }

            ui.add_space(8.0);
            if ui
                .button(RichText::new("Go to Client >").color(Color32::from_rgb(100, 180, 255)))
                .clicked()
            {
                state.active_tab = UiTab::Client;
            }
        });
}

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

fn format_uptime(secs: u64) -> String {
    let days = secs / 86400;
    let hours = (secs % 86400) / 3600;
    let minutes = (secs % 3600) / 60;
    if days > 0 {
        format!("{}d {}h {}m", days, hours, minutes)
    } else if hours > 0 {
        format!("{}h {}m", hours, minutes)
    } else {
        format!("{}m {}s", minutes, secs % 60)
    }
}
