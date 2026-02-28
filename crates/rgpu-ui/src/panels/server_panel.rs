use egui::{Color32, RichText, Ui, Vec2};

use rgpu_core::config::TransportMode;

use crate::state::{ServiceOrigin, ServiceStatus, UiState};
use crate::widgets::{gpu_card, metric_chart};

/// Render the Server detail panel.
pub fn show(ui: &mut Ui, state: &mut UiState) {
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            show_status_header(ui, state);
            ui.add_space(12.0);
            show_served_gpus(ui, state);
            ui.add_space(12.0);
            show_server_metrics(ui, state);
            ui.add_space(12.0);
            show_server_config(ui, state);
        });
}

// ── Status header + Start/Stop ──────────────────────────────────────────────

fn show_status_header(ui: &mut Ui, state: &mut UiState) {
    ui.heading("Server");
    ui.add_space(4.0);

    // Origin badge
    let origin_text = match &state.server.origin {
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

    // Status
    let (status_text, status_color) = status_display(&state.server.status);
    ui.horizontal(|ui| {
        ui.label(RichText::new("Status:").strong());
        ui.label(RichText::new(status_text).color(status_color).strong());
    });

    // Error detail
    if let ServiceStatus::Error(ref msg) = state.server.status {
        ui.label(
            RichText::new(format!("Error: {}", msg))
                .color(Color32::from_rgb(255, 100, 100))
                .small(),
        );
    }

    ui.add_space(4.0);

    // Start/Stop button
    let is_transitioning = matches!(
        state.server.status,
        ServiceStatus::Starting | ServiceStatus::Stopping
    );

    match state.server.status {
        ServiceStatus::Running => {
            let btn = ui.add_enabled(
                !is_transitioning,
                egui::Button::new(
                    RichText::new("Stop Server").color(Color32::from_rgb(255, 100, 100)),
                ),
            );
            if btn.clicked() {
                state.server.stop_requested = true;
            }
        }
        ServiceStatus::Stopped | ServiceStatus::Unknown | ServiceStatus::Error(_) => {
            let btn = ui.add_enabled(
                !is_transitioning,
                egui::Button::new(
                    RichText::new("Start Server").color(Color32::from_rgb(100, 255, 100)),
                ),
            );
            if btn.clicked() {
                state.server.start_requested = true;
            }
        }
        _ => {
            ui.add_enabled(false, egui::Button::new("Please wait..."));
        }
    }
}

// ── GPUs being served ───────────────────────────────────────────────────────

fn show_served_gpus(ui: &mut Ui, state: &UiState) {
    ui.separator();
    ui.add_space(4.0);
    ui.strong(format!("GPUs Being Served ({})", state.server.served_gpus.len()));
    ui.add_space(4.0);

    if state.server.served_gpus.is_empty() {
        ui.label(
            RichText::new("No GPUs available. Start the server to discover GPUs.")
                .color(Color32::GRAY)
                .italics(),
        );
        return;
    }

    for gpu in &state.server.served_gpus {
        gpu_card::gpu_card(ui, gpu, "local");
        ui.add_space(4.0);
    }
}

// ── Server metrics ──────────────────────────────────────────────────────────

fn show_server_metrics(ui: &mut Ui, state: &UiState) {
    ui.separator();
    ui.add_space(4.0);
    ui.strong("Server Metrics");
    ui.add_space(4.0);

    if !state.server.status.is_running() {
        ui.label(
            RichText::new("Server is not running. Metrics will appear when the server starts.")
                .color(Color32::GRAY)
                .italics(),
        );
        return;
    }

    // Summary cards
    if let Some(ref latest) = state.server.metrics {
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
                &format!("{:.1}", state.server.rates.requests_per_sec),
                Color32::from_rgb(100, 255, 180),
            );
            summary_card(
                ui,
                "CUDA/s",
                &format!("{:.1}", state.server.rates.cuda_per_sec),
                Color32::from_rgb(255, 200, 100),
            );
            summary_card(
                ui,
                "Vulkan/s",
                &format!("{:.1}", state.server.rates.vulkan_per_sec),
                Color32::from_rgb(200, 100, 255),
            );
            summary_card(
                ui,
                "Errors",
                &format!(
                    "{} ({:.1}/s)",
                    latest.errors_total, state.server.rates.errors_per_sec
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

    // Charts — 2x2 grid
    let chart_size = Vec2::new((ui.available_width() - 12.0) / 2.0, 150.0);

    ui.horizontal(|ui| {
        metric_chart::metric_line_chart(
            ui,
            "srv_req",
            "Requests (total)",
            &state.server.metrics_history,
            |s| s.requests_total as f64,
            Color32::from_rgb(100, 255, 180),
            chart_size,
        );
        ui.add_space(8.0);
        metric_chart::metric_line_chart(
            ui,
            "srv_conn",
            "Active Connections",
            &state.server.metrics_history,
            |s| s.connections_active as f64,
            Color32::from_rgb(100, 180, 255),
            chart_size,
        );
    });

    ui.add_space(8.0);

    ui.horizontal(|ui| {
        metric_chart::metric_line_chart(
            ui,
            "srv_cuda",
            "CUDA Commands (total)",
            &state.server.metrics_history,
            |s| s.cuda_commands as f64,
            Color32::from_rgb(255, 200, 100),
            chart_size,
        );
        ui.add_space(8.0);
        metric_chart::metric_line_chart(
            ui,
            "srv_vk",
            "Vulkan Commands (total)",
            &state.server.metrics_history,
            |s| s.vulkan_commands as f64,
            Color32::from_rgb(200, 100, 255),
            chart_size,
        );
    });
}

// ── Server configuration ────────────────────────────────────────────────────

fn show_server_config(ui: &mut Ui, state: &mut UiState) {
    ui.separator();
    ui.add_space(4.0);
    ui.strong("Server Configuration");
    ui.add_space(4.0);

    let is_running = state.server.status.is_running()
        || matches!(
            state.server.status,
            ServiceStatus::Starting | ServiceStatus::Stopping
        );

    if is_running {
        ui.label(
            RichText::new("Configuration is read-only while the server is running.")
                .small()
                .color(Color32::from_rgb(255, 200, 50)),
        );
        ui.add_space(4.0);
    }

    ui.add_enabled_ui(!is_running, |ui| {
        egui::Grid::new("server_config_grid")
            .num_columns(2)
            .spacing([16.0, 4.0])
            .show(ui, |ui| {
                ui.label("Server ID:");
                let mut sid = state.server.config.server_id as i32;
                if ui
                    .add(egui::DragValue::new(&mut sid).range(0..=65535))
                    .changed()
                {
                    state.server.config.server_id = sid as u16;
                }
                ui.end_row();

                ui.label("Port:");
                let mut port = state.server.config.port as i32;
                if ui
                    .add(egui::DragValue::new(&mut port).range(1..=65535))
                    .changed()
                {
                    state.server.config.port = port as u16;
                }
                ui.end_row();

                ui.label("Bind Address:");
                ui.text_edit_singleline(&mut state.server.config.bind);
                ui.end_row();

                ui.label("Transport:");
                let mut is_quic =
                    matches!(state.server.config.transport, TransportMode::Quic);
                if ui
                    .checkbox(&mut is_quic, "Use QUIC (instead of TCP)")
                    .changed()
                {
                    state.server.config.transport = if is_quic {
                        TransportMode::Quic
                    } else {
                        TransportMode::Tcp
                    };
                }
                ui.end_row();

                ui.label("TLS Certificate:");
                ui.text_edit_singleline(&mut state.server.config.cert_path);
                ui.end_row();

                ui.label("TLS Private Key:");
                ui.text_edit_singleline(&mut state.server.config.key_path);
                ui.end_row();

                ui.label("Max Clients:");
                let mut mc = state.server.config.max_clients as i32;
                if ui
                    .add(egui::DragValue::new(&mut mc).range(1..=1000))
                    .changed()
                {
                    state.server.config.max_clients = mc as u32;
                }
                ui.end_row();
            });

        // Token management
        ui.add_space(8.0);
        ui.label(RichText::new("Authentication Tokens:").strong().small());
        ui.add_space(4.0);

        let mut token_to_remove = None;
        for (idx, token_entry) in state.server.config.tokens.iter().enumerate() {
            ui.horizontal(|ui| {
                ui.label(format!(
                    "  {} ({}...)",
                    token_entry.name,
                    &token_entry.token[..token_entry.token.len().min(8)]
                ));
                if ui.small_button("Remove").clicked() {
                    token_to_remove = Some(idx);
                }
            });
        }
        if let Some(idx) = token_to_remove {
            state.server.config.tokens.remove(idx);
        }

        ui.horizontal(|ui| {
            ui.label("Name:");
            ui.text_edit_singleline(&mut state.server.config.new_token_name);
            ui.label("Token:");
            ui.text_edit_singleline(&mut state.server.config.new_token_value);
            if ui.button("Generate").clicked() {
                state.server.config.new_token_value =
                    rgpu_transport::auth::generate_token(32);
            }
            if ui.button("Add Token").clicked()
                && !state.server.config.new_token_name.is_empty()
                && !state.server.config.new_token_value.is_empty()
            {
                state.server.config.tokens.push(rgpu_core::config::TokenEntry {
                    token: state.server.config.new_token_value.clone(),
                    name: state.server.config.new_token_name.clone(),
                    allowed_gpus: None,
                    max_memory: None,
                });
                state.server.config.new_token_name.clear();
                state.server.config.new_token_value.clear();
            }
        });
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
    let minutes = (secs % 3600) / 60;
    if days > 0 {
        format!("{}d {}h {}m", days, hours, minutes)
    } else if hours > 0 {
        format!("{}h {}m", hours, minutes)
    } else {
        format!("{}m {}s", minutes, secs % 60)
    }
}
