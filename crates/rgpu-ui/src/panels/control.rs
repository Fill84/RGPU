use egui::{Color32, RichText, Ui};

use rgpu_core::config::{TokenEntry, TransportMode};

use crate::state::{LocalServerStatus, PendingConnection, ServerConnectionState, UiState};

/// Render the Control panel — server start/stop and connection management.
pub fn show(ui: &mut Ui, state: &mut UiState) {
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            show_server_control(ui, state);
            ui.add_space(16.0);
            show_connections(ui, state);
        });
}

/// Server control section: configure and start/stop an embedded server.
fn show_server_control(ui: &mut Ui, state: &mut UiState) {
    ui.heading("Server Control");
    ui.add_space(4.0);

    // Status indicator
    let (status_text, status_color) = match &state.local_server_status {
        LocalServerStatus::Stopped => ("Stopped", Color32::from_rgb(150, 150, 150)),
        LocalServerStatus::Starting => ("Starting...", Color32::from_rgb(255, 200, 50)),
        LocalServerStatus::Running => ("Running", Color32::from_rgb(100, 200, 100)),
        LocalServerStatus::Stopping => ("Stopping...", Color32::from_rgb(255, 165, 0)),
        LocalServerStatus::Error(msg) => {
            ui.horizontal(|ui| {
                ui.label(RichText::new("Status:").strong());
                ui.label(
                    RichText::new(format!("Error: {}", msg))
                        .color(Color32::from_rgb(255, 100, 100)),
                );
            });
            // Show start button after error so user can retry
            if ui.button("Start Server").clicked() {
                state.server_start_requested = true;
                state.local_server_status = LocalServerStatus::Starting;
            }
            return show_server_config_form(ui, state, true);
        }
    };

    ui.horizontal(|ui| {
        ui.label(RichText::new("Status:").strong());
        ui.label(RichText::new(status_text).color(status_color).strong());
    });

    ui.add_space(4.0);

    // Start/Stop button
    let is_transitioning = matches!(
        state.local_server_status,
        LocalServerStatus::Starting | LocalServerStatus::Stopping
    );

    match state.local_server_status {
        LocalServerStatus::Stopped | LocalServerStatus::Error(_) => {
            let btn = ui.add_enabled(
                !is_transitioning,
                egui::Button::new(
                    RichText::new("Start Server").color(Color32::from_rgb(100, 255, 100)),
                ),
            );
            if btn.clicked() {
                state.server_start_requested = true;
                state.local_server_status = LocalServerStatus::Starting;
            }
        }
        LocalServerStatus::Running => {
            let btn = ui.add_enabled(
                !is_transitioning,
                egui::Button::new(
                    RichText::new("Stop Server").color(Color32::from_rgb(255, 100, 100)),
                ),
            );
            if btn.clicked() {
                state.server_stop_requested = true;
                state.local_server_status = LocalServerStatus::Stopping;
            }
        }
        _ => {
            ui.add_enabled(false, egui::Button::new("Please wait..."));
        }
    }

    ui.add_space(8.0);

    let is_running = matches!(
        state.local_server_status,
        LocalServerStatus::Running | LocalServerStatus::Starting | LocalServerStatus::Stopping
    );
    show_server_config_form(ui, state, is_running);
}

/// Server configuration form (disabled when server is running).
fn show_server_config_form(ui: &mut Ui, state: &mut UiState, disabled: bool) {
    ui.separator();
    ui.label(RichText::new("Server Configuration").strong().small());
    ui.add_space(4.0);

    ui.add_enabled_ui(!disabled, |ui| {
        egui::Grid::new("server_control_grid")
            .num_columns(2)
            .spacing([16.0, 4.0])
            .show(ui, |ui| {
                ui.label("Server ID:");
                let mut sid = state.local_server_config.server_id as i32;
                if ui
                    .add(egui::DragValue::new(&mut sid).range(0..=65535))
                    .changed()
                {
                    state.local_server_config.server_id = sid as u16;
                }
                ui.end_row();

                ui.label("Port:");
                let mut port = state.local_server_config.port as i32;
                if ui
                    .add(egui::DragValue::new(&mut port).range(1..=65535))
                    .changed()
                {
                    state.local_server_config.port = port as u16;
                }
                ui.end_row();

                ui.label("Bind Address:");
                ui.text_edit_singleline(&mut state.local_server_config.bind);
                ui.end_row();

                ui.label("Transport:");
                let mut is_quic = matches!(
                    state.local_server_config.transport,
                    TransportMode::Quic
                );
                if ui
                    .checkbox(&mut is_quic, "Use QUIC (instead of TCP)")
                    .changed()
                {
                    state.local_server_config.transport = if is_quic {
                        TransportMode::Quic
                    } else {
                        TransportMode::Tcp
                    };
                }
                ui.end_row();

                ui.label("TLS Certificate:");
                ui.text_edit_singleline(&mut state.local_server_config.cert_path);
                ui.end_row();

                ui.label("TLS Private Key:");
                ui.text_edit_singleline(&mut state.local_server_config.key_path);
                ui.end_row();

                ui.label("Max Clients:");
                let mut mc = state.local_server_config.max_clients as i32;
                if ui
                    .add(egui::DragValue::new(&mut mc).range(1..=1000))
                    .changed()
                {
                    state.local_server_config.max_clients = mc as u32;
                }
                ui.end_row();
            });

        // Token management
        ui.add_space(8.0);
        ui.label(RichText::new("Authentication Tokens:").strong().small());
        ui.add_space(4.0);

        let mut token_to_remove = None;
        for (idx, token_entry) in state.local_server_config.tokens.iter().enumerate() {
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
            state.local_server_config.tokens.remove(idx);
        }

        ui.horizontal(|ui| {
            ui.label("Name:");
            ui.text_edit_singleline(&mut state.local_server_config.new_token_name);
            ui.label("Token:");
            ui.text_edit_singleline(&mut state.local_server_config.new_token_value);
            if ui.button("Generate").clicked() {
                state.local_server_config.new_token_value =
                    rgpu_transport::auth::generate_token(32);
            }
            if ui.button("Add Token").clicked()
                && !state.local_server_config.new_token_name.is_empty()
                && !state.local_server_config.new_token_value.is_empty()
            {
                state.local_server_config.tokens.push(TokenEntry {
                    token: state.local_server_config.new_token_value.clone(),
                    name: state.local_server_config.new_token_name.clone(),
                    allowed_gpus: None,
                    max_memory: None,
                });
                state.local_server_config.new_token_name.clear();
                state.local_server_config.new_token_value.clear();
            }
        });
    });
}

/// Connections section: list active connections + add/remove dynamically.
fn show_connections(ui: &mut Ui, state: &mut UiState) {
    ui.heading("Connections");
    ui.add_space(4.0);

    if state.servers.is_empty() {
        ui.label(
            RichText::new("No connections. Add a server below to connect.")
                .color(Color32::GRAY)
                .italics(),
        );
    } else {
        // Connection list
        let mut disconnect_idx = None;
        for (idx, server) in state.servers.iter().enumerate() {
            ui.horizontal(|ui| {
                // Status dot
                let dot_color = match &server.connection_state {
                    ServerConnectionState::Connected => Color32::from_rgb(100, 200, 100),
                    ServerConnectionState::Connecting => Color32::from_rgb(255, 200, 50),
                    ServerConnectionState::Error(_) => Color32::from_rgb(255, 100, 100),
                    ServerConnectionState::Disconnected => Color32::from_rgb(150, 150, 150),
                };
                let (rect, _) = ui.allocate_exact_size(
                    egui::vec2(10.0, 10.0),
                    egui::Sense::hover(),
                );
                ui.painter()
                    .circle_filled(rect.center(), 5.0, dot_color);

                // Address and status
                ui.label(RichText::new(&server.address).strong());

                let status_text = match &server.connection_state {
                    ServerConnectionState::Connected => "Connected".to_string(),
                    ServerConnectionState::Connecting => "Connecting...".to_string(),
                    ServerConnectionState::Error(e) => format!("Error: {}", e),
                    ServerConnectionState::Disconnected => "Disconnected".to_string(),
                };
                ui.label(
                    RichText::new(format!("({})", status_text))
                        .small()
                        .color(Color32::GRAY),
                );

                if let Some(sid) = server.server_id {
                    ui.label(
                        RichText::new(format!("[ID: {}]", sid))
                            .small()
                            .color(Color32::GRAY),
                    );
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .button(RichText::new("Disconnect").color(Color32::from_rgb(255, 100, 100)))
                        .clicked()
                    {
                        disconnect_idx = Some(idx);
                    }
                });
            });
        }

        if let Some(idx) = disconnect_idx {
            state.servers[idx].should_disconnect = true;
        }
    }

    ui.add_space(8.0);
    ui.separator();
    ui.label(RichText::new("Add Connection").strong().small());
    ui.add_space(4.0);

    ui.horizontal(|ui| {
        ui.label("Address:");
        ui.add(
            egui::TextEdit::singleline(&mut state.new_connection_address)
                .hint_text("host:port"),
        );
        ui.label("Token:");
        ui.add(
            egui::TextEdit::singleline(&mut state.new_connection_token)
                .hint_text("auth token"),
        );
        if ui
            .button(RichText::new("Connect").color(Color32::from_rgb(100, 200, 255)))
            .clicked()
            && !state.new_connection_address.is_empty()
        {
            state.pending_connections.push(PendingConnection {
                address: state.new_connection_address.clone(),
                token: state.new_connection_token.clone(),
            });
            state.new_connection_address.clear();
            state.new_connection_token.clear();
        }
    });
}
