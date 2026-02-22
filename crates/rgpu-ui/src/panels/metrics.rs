use egui::{Color32, RichText, Ui, Vec2};

use crate::state::{LocalServerStatus, UiState};
use crate::widgets::metric_chart;

/// Render the live metrics dashboard.
pub fn show(ui: &mut Ui, state: &UiState) {
    ui.heading("Live Metrics");
    ui.add_space(4.0);

    let has_embedded = state.embedded_server_metrics.is_some();
    let connected: Vec<_> = state
        .servers
        .iter()
        .filter(|s| s.connection_state.is_connected())
        .collect();

    if !has_embedded && connected.is_empty() {
        ui.label("No servers running or connected. Metrics will appear once a server is started or a connection is established.");
        return;
    }

    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            // --- Embedded (local) server metrics ---
            if let Some(latest) = &state.embedded_server_metrics {
                let status_label = match &state.local_server_status {
                    LocalServerStatus::Running => "Running",
                    _ => "Active",
                };
                ui.strong(format!(
                    "Local Server (ID: {}) - {}",
                    state.local_server_config.server_id, status_label
                ));
                ui.add_space(4.0);

                // Summary cards row
                ui.horizontal_wrapped(|ui| {
                    summary_card(
                        ui,
                        "Connections",
                        &format!(
                            "{} active / {} total",
                            latest.connections_active, latest.connections_total
                        ),
                        Color32::from_rgb(100, 180, 255),
                    );
                    summary_card(
                        ui,
                        "Requests/s",
                        &format!("{:.1}", state.embedded_server_rates.requests_per_sec),
                        Color32::from_rgb(100, 255, 180),
                    );
                    summary_card(
                        ui,
                        "CUDA/s",
                        &format!("{:.1}", state.embedded_server_rates.cuda_per_sec),
                        Color32::from_rgb(255, 200, 100),
                    );
                    summary_card(
                        ui,
                        "Vulkan/s",
                        &format!("{:.1}", state.embedded_server_rates.vulkan_per_sec),
                        Color32::from_rgb(200, 100, 255),
                    );
                    summary_card(
                        ui,
                        "Errors",
                        &format!(
                            "{} ({:.1}/s)",
                            latest.errors_total, state.embedded_server_rates.errors_per_sec
                        ),
                        Color32::from_rgb(255, 100, 100),
                    );
                    summary_card(
                        ui,
                        "Uptime",
                        &format_uptime(latest.uptime_secs),
                        Color32::from_rgb(180, 180, 180),
                    );
                });

                ui.add_space(8.0);

                // Charts - 2x2 grid
                let chart_size = Vec2::new(
                    (ui.available_width() - 12.0) / 2.0,
                    150.0,
                );

                ui.horizontal(|ui| {
                    metric_chart::metric_line_chart(
                        ui,
                        "req_local_server",
                        "Requests (total)",
                        &state.embedded_server_metrics_history,
                        |s| s.requests_total as f64,
                        Color32::from_rgb(100, 255, 180),
                        chart_size,
                    );
                    ui.add_space(8.0);
                    metric_chart::metric_line_chart(
                        ui,
                        "conn_local_server",
                        "Active Connections",
                        &state.embedded_server_metrics_history,
                        |s| s.connections_active as f64,
                        Color32::from_rgb(100, 180, 255),
                        chart_size,
                    );
                });

                ui.add_space(8.0);

                ui.horizontal(|ui| {
                    metric_chart::metric_line_chart(
                        ui,
                        "cuda_local_server",
                        "CUDA Commands (total)",
                        &state.embedded_server_metrics_history,
                        |s| s.cuda_commands as f64,
                        Color32::from_rgb(255, 200, 100),
                        chart_size,
                    );
                    ui.add_space(8.0);
                    metric_chart::metric_line_chart(
                        ui,
                        "vk_local_server",
                        "Vulkan Commands (total)",
                        &state.embedded_server_metrics_history,
                        |s| s.vulkan_commands as f64,
                        Color32::from_rgb(200, 100, 255),
                        chart_size,
                    );
                });

                ui.add_space(16.0);

                if !connected.is_empty() {
                    ui.separator();
                    ui.add_space(8.0);
                }
            }

            // --- Remote server metrics ---
            for server in &connected {
                ui.strong(format!(
                    "Server: {} (ID: {})",
                    server.address,
                    server.server_id.unwrap_or(0)
                ));
                ui.add_space(4.0);

                // Summary cards row
                if let Some(latest) = server.latest_metrics() {
                    ui.horizontal_wrapped(|ui| {
                        summary_card(
                            ui,
                            "Connections",
                            &format!(
                                "{} active / {} total",
                                latest.connections_active, latest.connections_total
                            ),
                            Color32::from_rgb(100, 180, 255),
                        );
                        summary_card(
                            ui,
                            "Requests/s",
                            &format!("{:.1}", server.current_rates.requests_per_sec),
                            Color32::from_rgb(100, 255, 180),
                        );
                        summary_card(
                            ui,
                            "CUDA/s",
                            &format!("{:.1}", server.current_rates.cuda_per_sec),
                            Color32::from_rgb(255, 200, 100),
                        );
                        summary_card(
                            ui,
                            "Vulkan/s",
                            &format!("{:.1}", server.current_rates.vulkan_per_sec),
                            Color32::from_rgb(200, 100, 255),
                        );
                        summary_card(
                            ui,
                            "Errors",
                            &format!(
                                "{} ({:.1}/s)",
                                latest.errors_total, server.current_rates.errors_per_sec
                            ),
                            Color32::from_rgb(255, 100, 100),
                        );
                        summary_card(
                            ui,
                            "Uptime",
                            &format_uptime(latest.uptime_secs),
                            Color32::from_rgb(180, 180, 180),
                        );
                    });
                }

                ui.add_space(8.0);

                // Charts - 2x2 grid
                let chart_size = Vec2::new(
                    (ui.available_width() - 12.0) / 2.0,
                    150.0,
                );

                ui.horizontal(|ui| {
                    metric_chart::metric_line_chart(
                        ui,
                        &format!("req_{}", server.address),
                        "Requests (total)",
                        &server.metrics_history,
                        |s| s.requests_total as f64,
                        Color32::from_rgb(100, 255, 180),
                        chart_size,
                    );
                    ui.add_space(8.0);
                    metric_chart::metric_line_chart(
                        ui,
                        &format!("conn_{}", server.address),
                        "Active Connections",
                        &server.metrics_history,
                        |s| s.connections_active as f64,
                        Color32::from_rgb(100, 180, 255),
                        chart_size,
                    );
                });

                ui.add_space(8.0);

                ui.horizontal(|ui| {
                    metric_chart::metric_line_chart(
                        ui,
                        &format!("cuda_{}", server.address),
                        "CUDA Commands (total)",
                        &server.metrics_history,
                        |s| s.cuda_commands as f64,
                        Color32::from_rgb(255, 200, 100),
                        chart_size,
                    );
                    ui.add_space(8.0);
                    metric_chart::metric_line_chart(
                        ui,
                        &format!("vk_{}", server.address),
                        "Vulkan Commands (total)",
                        &server.metrics_history,
                        |s| s.vulkan_commands as f64,
                        Color32::from_rgb(200, 100, 255),
                        chart_size,
                    );
                });

                ui.add_space(16.0);
            }
        });
}

fn summary_card(ui: &mut Ui, label: &str, value: &str, color: Color32) {
    egui::Frame::group(ui.style())
        .inner_margin(egui::Margin::same(6))
        .show(ui, |ui| {
            ui.vertical(|ui| {
                ui.label(RichText::new(label).small().color(Color32::GRAY));
                ui.label(RichText::new(value).color(color).strong());
            });
        });
}

fn format_uptime(secs: u64) -> String {
    let days = secs / 86400;
    let hours = (secs % 86400) / 3600;
    let mins = (secs % 3600) / 60;
    if days > 0 {
        format!("{}d {}h {}m", days, hours, mins)
    } else if hours > 0 {
        format!("{}h {}m", hours, mins)
    } else {
        format!("{}m {}s", mins, secs % 60)
    }
}
