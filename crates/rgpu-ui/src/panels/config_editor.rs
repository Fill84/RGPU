use egui::{Color32, RichText, Ui};

use rgpu_core::config::{
    GpuOrdering, RgpuConfig, ServerEndpoint, TokenEntry, TransportMode,
};

use crate::state::{ConfigEditorState, UiState};

/// Render the configuration editor panel.
pub fn show(ui: &mut Ui, state: &mut UiState) {
    ui.heading("Configuration Editor");
    ui.add_space(4.0);
    ui.label(
        RichText::new(format!("File: {}", state.config_path))
            .small()
            .color(Color32::GRAY),
    );
    ui.add_space(8.0);

    // Action buttons
    ui.horizontal(|ui| {
        if ui.button("Reload from disk").clicked() {
            match RgpuConfig::load(&state.config_path) {
                Ok(config) => {
                    state.config_editor = Some(ConfigEditorState::from_config(config));
                }
                Err(e) => {
                    state.push_error(format!("Failed to load config: {}", e));
                }
            }
        }

        if let Some(ref editor) = state.config_editor {
            if editor.dirty {
                if ui
                    .button(RichText::new("Save").color(Color32::from_rgb(100, 255, 100)))
                    .clicked()
                {
                    match toml::to_string_pretty(&editor.config) {
                        Ok(content) => {
                            if let Err(e) = std::fs::write(&state.config_path, &content) {
                                state.push_error(format!("Failed to save config: {}", e));
                            } else {
                                if let Some(ref mut ed) = state.config_editor {
                                    ed.dirty = false;
                                }
                            }
                        }
                        Err(e) => {
                            state.push_error(format!("Failed to serialize config: {}", e));
                        }
                    }
                }

                ui.label(
                    RichText::new("(unsaved changes)")
                        .color(Color32::from_rgb(255, 200, 50))
                        .small(),
                );
            }
        }
    });

    ui.add_space(8.0);

    if state.config_editor.is_none() {
        ui.label("No config file loaded. Click 'Reload from disk' or create a new rgpu.toml.");
        if ui.button("Create default config").clicked() {
            let config = RgpuConfig::default();
            state.config_editor = Some(ConfigEditorState::from_config(config));
            if let Some(ref mut ed) = state.config_editor {
                ed.dirty = true;
            }
        }
        return;
    }

    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            // We need to work with the editor mutably
            let editor = state.config_editor.as_mut().unwrap();

            // --- Server Config ---
            ui.separator();
            ui.strong("Server Configuration");
            ui.add_space(4.0);

            egui::Grid::new("server_config_grid")
                .num_columns(2)
                .spacing([16.0, 4.0])
                .show(ui, |ui| {
                    ui.label("Server ID:");
                    let mut sid = editor.config.server.server_id as i32;
                    if ui
                        .add(egui::DragValue::new(&mut sid).range(0..=65535))
                        .changed()
                    {
                        editor.config.server.server_id = sid as u16;
                        editor.dirty = true;
                    }
                    ui.end_row();

                    ui.label("Port:");
                    let mut port = editor.config.server.port as i32;
                    if ui
                        .add(egui::DragValue::new(&mut port).range(1..=65535))
                        .changed()
                    {
                        editor.config.server.port = port as u16;
                        editor.dirty = true;
                    }
                    ui.end_row();

                    ui.label("Bind Address:");
                    if ui
                        .text_edit_singleline(&mut editor.config.server.bind)
                        .changed()
                    {
                        editor.dirty = true;
                    }
                    ui.end_row();

                    ui.label("Transport:");
                    let mut is_quic =
                        matches!(editor.config.server.transport, TransportMode::Quic);
                    if ui
                        .checkbox(&mut is_quic, "Use QUIC (instead of TCP)")
                        .changed()
                    {
                        editor.config.server.transport = if is_quic {
                            TransportMode::Quic
                        } else {
                            TransportMode::Tcp
                        };
                        editor.dirty = true;
                    }
                    ui.end_row();

                    ui.label("Max Clients:");
                    let mut mc = editor.config.server.max_clients as i32;
                    if ui
                        .add(egui::DragValue::new(&mut mc).range(1..=1000))
                        .changed()
                    {
                        editor.config.server.max_clients = mc as u32;
                        editor.dirty = true;
                    }
                    ui.end_row();

                    ui.label("TLS Certificate:");
                    let mut cert = editor
                        .config
                        .server
                        .cert_path
                        .clone()
                        .unwrap_or_default();
                    if ui.text_edit_singleline(&mut cert).changed() {
                        editor.config.server.cert_path = if cert.is_empty() {
                            None
                        } else {
                            Some(cert)
                        };
                        editor.dirty = true;
                    }
                    ui.end_row();

                    ui.label("TLS Private Key:");
                    let mut key_path = editor
                        .config
                        .server
                        .key_path
                        .clone()
                        .unwrap_or_default();
                    if ui.text_edit_singleline(&mut key_path).changed() {
                        editor.config.server.key_path = if key_path.is_empty() {
                            None
                        } else {
                            Some(key_path)
                        };
                        editor.dirty = true;
                    }
                    ui.end_row();
                });

            ui.add_space(12.0);

            // --- Client Config ---
            ui.separator();
            ui.strong("Client Configuration");
            ui.add_space(4.0);

            egui::Grid::new("client_config_grid")
                .num_columns(2)
                .spacing([16.0, 4.0])
                .show(ui, |ui| {
                    ui.label("Include Local GPUs:");
                    if ui
                        .checkbox(
                            &mut editor.config.client.include_local_gpus,
                            "",
                        )
                        .changed()
                    {
                        editor.dirty = true;
                    }
                    ui.end_row();

                    ui.label("GPU Ordering:");
                    egui::ComboBox::from_id_salt("gpu_ordering")
                        .selected_text(format!("{:?}", editor.config.client.gpu_ordering))
                        .show_ui(ui, |ui| {
                            if ui
                                .selectable_value(
                                    &mut editor.config.client.gpu_ordering,
                                    GpuOrdering::LocalFirst,
                                    "LocalFirst",
                                )
                                .changed()
                            {
                                editor.dirty = true;
                            }
                            if ui
                                .selectable_value(
                                    &mut editor.config.client.gpu_ordering,
                                    GpuOrdering::RemoteFirst,
                                    "RemoteFirst",
                                )
                                .changed()
                            {
                                editor.dirty = true;
                            }
                            if ui
                                .selectable_value(
                                    &mut editor.config.client.gpu_ordering,
                                    GpuOrdering::ByCapability,
                                    "ByCapability",
                                )
                                .changed()
                            {
                                editor.dirty = true;
                            }
                        });
                    ui.end_row();
                });

            ui.add_space(8.0);

            // Server endpoints list
            ui.label(RichText::new("Server Endpoints:").strong());
            ui.add_space(4.0);

            let mut to_remove = None;
            for (idx, endpoint) in editor.config.client.servers.iter().enumerate() {
                ui.horizontal(|ui| {
                    ui.label(format!("  {} (token: {}...)", endpoint.address, &endpoint.token[..endpoint.token.len().min(8)]));
                    if ui.small_button("Remove").clicked() {
                        to_remove = Some(idx);
                    }
                });
            }
            if let Some(idx) = to_remove {
                editor.config.client.servers.remove(idx);
                editor.dirty = true;
            }

            ui.horizontal(|ui| {
                ui.label("Address:");
                ui.text_edit_singleline(&mut editor.new_server_address);
                ui.label("Token:");
                ui.text_edit_singleline(&mut editor.new_server_token);
                if ui.button("Add Server").clicked()
                    && !editor.new_server_address.is_empty()
                {
                    editor.config.client.servers.push(ServerEndpoint {
                        address: editor.new_server_address.clone(),
                        token: editor.new_server_token.clone(),
                        ca_cert: None,
                        transport: TransportMode::default(),
                    });
                    editor.new_server_address.clear();
                    editor.new_server_token.clear();
                    editor.dirty = true;
                }
            });

            ui.add_space(12.0);

            // --- Security Config ---
            ui.separator();
            ui.strong("Security - Tokens");
            ui.add_space(4.0);

            let mut token_to_remove = None;
            for (idx, token_entry) in editor.config.security.tokens.iter().enumerate() {
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
                editor.config.security.tokens.remove(idx);
                editor.dirty = true;
            }

            ui.horizontal(|ui| {
                ui.label("Name:");
                ui.text_edit_singleline(&mut editor.new_token_name);
                ui.label("Token:");
                ui.text_edit_singleline(&mut editor.new_token_value);
                if ui.button("Generate").clicked() {
                    editor.new_token_value =
                        rgpu_transport::auth::generate_token(32);
                }
                if ui.button("Add Token").clicked()
                    && !editor.new_token_name.is_empty()
                    && !editor.new_token_value.is_empty()
                {
                    editor.config.security.tokens.push(TokenEntry {
                        token: editor.new_token_value.clone(),
                        name: editor.new_token_name.clone(),
                        allowed_gpus: None,
                        max_memory: None,
                    });
                    editor.new_token_name.clear();
                    editor.new_token_value.clear();
                    editor.dirty = true;
                }
            });
        });
}
