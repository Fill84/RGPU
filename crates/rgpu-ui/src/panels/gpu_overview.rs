use egui::{Color32, RichText, Ui};

use crate::state::{LocalServerStatus, ServerConnectionState, UiState};
use crate::widgets::gpu_card;

/// Render the GPU overview panel showing all GPUs grouped by server.
pub fn show(ui: &mut Ui, state: &UiState) {
    ui.heading("GPU Overview");
    ui.add_space(4.0);

    let has_embedded = !state.embedded_server_gpus.is_empty();
    let embedded_label = if has_embedded { " (incl. local server)" } else { "" };
    ui.label(format!(
        "{} GPU(s) across {} server(s) ({} connected){}",
        state.total_gpus(),
        state.servers.len() + if has_embedded { 1 } else { 0 },
        state.connected_servers() + if has_embedded { 1 } else { 0 },
        embedded_label
    ));
    ui.add_space(8.0);

    if state.servers.is_empty() && state.embedded_server_gpus.is_empty() {
        ui.label("No servers configured. Add servers via the Config Editor or --server flag.");
        return;
    }

    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            // Show embedded (local) server GPUs first
            if !state.embedded_server_gpus.is_empty() {
                let is_running = matches!(state.local_server_status, LocalServerStatus::Running);
                let status_color = if is_running {
                    Color32::from_rgb(100, 200, 100)
                } else {
                    Color32::from_rgb(180, 180, 180)
                };
                let status_text = if is_running { "Running" } else { "Stopped" };
                let server_label = format!("Local Server ({})", status_text);

                egui::CollapsingHeader::new(
                    RichText::new(&server_label).color(status_color),
                )
                .id_salt("embedded_server_group")
                .default_open(true)
                .show(ui, |ui| {
                    if let Some(ref metrics) = state.embedded_server_metrics {
                        ui.label(
                            RichText::new(format!(
                                "Uptime: {}s | CUDA cmds: {} | Vulkan cmds: {} | Connections: {}",
                                metrics.uptime_secs,
                                metrics.cuda_commands,
                                metrics.vulkan_commands,
                                metrics.connections_active
                            ))
                            .small()
                            .color(Color32::GRAY),
                        );
                    }

                    for gpu in &state.embedded_server_gpus {
                        ui.add_space(4.0);
                        gpu_card::gpu_card(ui, gpu, "local");
                    }
                });

                ui.add_space(8.0);
            }

            // Show remote servers
            for (i, server) in state.servers.iter().enumerate() {
                let header_id = ui.make_persistent_id(format!("server_group_{}", i));

                let status_color = match &server.connection_state {
                    ServerConnectionState::Connected => Color32::from_rgb(100, 200, 100),
                    ServerConnectionState::Connecting => Color32::from_rgb(255, 200, 50),
                    ServerConnectionState::Disconnected => Color32::from_rgb(180, 180, 180),
                    ServerConnectionState::Error(_) => Color32::from_rgb(255, 80, 80),
                };

                let status_text = match &server.connection_state {
                    ServerConnectionState::Connected => "Connected",
                    ServerConnectionState::Connecting => "Connecting...",
                    ServerConnectionState::Disconnected => "Disconnected",
                    ServerConnectionState::Error(_) => "Error",
                };

                let server_label = format!(
                    "Server: {} ({})",
                    server.address, status_text
                );

                egui::CollapsingHeader::new(
                    RichText::new(&server_label).color(status_color),
                )
                .id_salt(header_id)
                .default_open(true)
                .show(ui, |ui| {
                    if let ServerConnectionState::Error(ref e) = server.connection_state {
                        ui.label(
                            RichText::new(format!("Error: {}", e))
                                .color(Color32::from_rgb(255, 100, 100))
                                .small(),
                        );
                        ui.add_space(4.0);
                    }

                    if let Some(sid) = server.server_id {
                        ui.label(
                            RichText::new(format!("Server ID: {}", sid))
                                .small()
                                .color(Color32::GRAY),
                        );
                    }

                    if server.gpus.is_empty() {
                        ui.label("No GPUs available from this server.");
                    } else {
                        for gpu in &server.gpus {
                            ui.add_space(4.0);
                            gpu_card::gpu_card(ui, gpu, &server.address);
                        }
                    }
                });

                ui.add_space(8.0);
            }
        });
}
