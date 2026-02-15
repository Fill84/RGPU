use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use egui::{Color32, RichText};

use crate::data_fetcher;
use crate::panels;
use crate::state::{LocalServerStatus, UiState, UiTab};

/// Main RGPU UI application.
pub struct RgpuApp {
    state: Arc<Mutex<UiState>>,
    _fetcher_handle: Option<JoinHandle<()>>,
}

impl RgpuApp {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        servers: Vec<(String, String)>,
        config_path: String,
        poll_interval: u64,
    ) -> Self {
        // Set dark mode by default
        cc.egui_ctx.set_visuals(egui::Visuals::dark());

        let state = Arc::new(Mutex::new(UiState::new(
            servers,
            config_path,
            poll_interval,
        )));

        // Start the background data fetcher
        let fetcher_handle =
            data_fetcher::start_data_fetcher(state.clone(), cc.egui_ctx.clone());

        Self {
            state,
            _fetcher_handle: Some(fetcher_handle),
        }
    }
}

impl eframe::App for RgpuApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Top panel with tabs
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("RGPU")
                        .strong()
                        .color(Color32::from_rgb(100, 180, 255)),
                );
                ui.separator();

                let mut st = self.state.lock().unwrap();
                let active = st.active_tab;

                if ui
                    .selectable_label(active == UiTab::Control, "Control")
                    .clicked()
                {
                    st.active_tab = UiTab::Control;
                }
                if ui
                    .selectable_label(active == UiTab::GpuOverview, "GPU Overview")
                    .clicked()
                {
                    st.active_tab = UiTab::GpuOverview;
                }
                if ui
                    .selectable_label(active == UiTab::Metrics, "Metrics")
                    .clicked()
                {
                    st.active_tab = UiTab::Metrics;
                }
                if ui
                    .selectable_label(active == UiTab::ConfigEditor, "Config Editor")
                    .clicked()
                {
                    st.active_tab = UiTab::ConfigEditor;
                }
            });
        });

        // Bottom status bar
        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                let st = self.state.lock().unwrap();
                let connected = st.connected_servers();
                let total = st.servers.len();
                let total_gpus = st.total_gpus();

                // Embedded server status
                let (server_text, server_color) = match &st.local_server_status {
                    LocalServerStatus::Stopped => ("Server: Off", Color32::from_rgb(150, 150, 150)),
                    LocalServerStatus::Starting => {
                        ("Server: Starting", Color32::from_rgb(255, 200, 50))
                    }
                    LocalServerStatus::Running => {
                        ("Server: Running", Color32::from_rgb(100, 200, 100))
                    }
                    LocalServerStatus::Stopping => {
                        ("Server: Stopping", Color32::from_rgb(255, 165, 0))
                    }
                    LocalServerStatus::Error(_) => {
                        ("Server: Error", Color32::from_rgb(255, 100, 100))
                    }
                };

                ui.label(RichText::new(server_text).color(server_color).small());

                ui.separator();

                let status_color = if connected == total && total > 0 {
                    Color32::from_rgb(100, 200, 100)
                } else if connected > 0 {
                    Color32::from_rgb(255, 200, 50)
                } else {
                    Color32::from_rgb(255, 100, 100)
                };

                ui.label(
                    RichText::new(format!(
                        "Servers: {}/{} connected",
                        connected, total
                    ))
                    .color(status_color)
                    .small(),
                );

                ui.separator();

                ui.label(
                    RichText::new(format!("{} GPU(s)", total_gpus))
                        .small()
                        .color(Color32::GRAY),
                );

                // Show latest error if any
                if let Some(last_error) = st.error_log.back() {
                    ui.separator();
                    ui.label(
                        RichText::new(last_error)
                            .small()
                            .color(Color32::from_rgb(255, 100, 100)),
                    );
                }
            });
        });

        // Central panel with active tab content
        egui::CentralPanel::default().show(ctx, |ui| {
            let active_tab = {
                let st = self.state.lock().unwrap();
                st.active_tab
            };

            match active_tab {
                UiTab::Control => {
                    let mut st = self.state.lock().unwrap();
                    panels::control::show(ui, &mut st);
                }
                UiTab::GpuOverview => {
                    let st = self.state.lock().unwrap();
                    panels::gpu_overview::show(ui, &st);
                }
                UiTab::Metrics => {
                    let st = self.state.lock().unwrap();
                    panels::metrics::show(ui, &st);
                }
                UiTab::ConfigEditor => {
                    let mut st = self.state.lock().unwrap();
                    panels::config_editor::show(ui, &mut st);
                }
            }
        });
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        // Signal the fetcher to stop (and stop the embedded server if running)
        if let Ok(mut st) = self.state.lock() {
            st.should_stop = true;
            st.server_stop_requested = true;
        }
    }
}
