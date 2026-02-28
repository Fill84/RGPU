use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use egui::{Color32, RichText};

use crate::data_fetcher;
use crate::panels;
use crate::state::{ServiceStatus, UiState, UiTab};

/// Main RGPU UI application.
pub struct RgpuApp {
    state: Arc<Mutex<UiState>>,
    _fetcher_handle: Option<JoinHandle<()>>,
}

impl RgpuApp {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        config_path: String,
        poll_interval: u64,
    ) -> Self {
        // Set dark mode by default
        cc.egui_ctx.set_visuals(egui::Visuals::dark());

        let state = Arc::new(Mutex::new(UiState::new(config_path, poll_interval)));

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
                    .selectable_label(active == UiTab::Dashboard, "Dashboard")
                    .clicked()
                {
                    st.active_tab = UiTab::Dashboard;
                }
                if ui
                    .selectable_label(active == UiTab::Server, "Server")
                    .clicked()
                {
                    st.active_tab = UiTab::Server;
                }
                if ui
                    .selectable_label(active == UiTab::Client, "Client")
                    .clicked()
                {
                    st.active_tab = UiTab::Client;
                }
                if ui
                    .selectable_label(active == UiTab::ConfigEditor, "Config")
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

                // Server status
                let (server_text, server_color) = status_label("Server", &st.server.status);
                ui.label(RichText::new(server_text).color(server_color).small());

                ui.separator();

                // Client status
                let (client_text, client_color) = status_label("Client", &st.client.status);
                ui.label(RichText::new(client_text).color(client_color).small());

                ui.separator();

                // GPU counts
                let server_gpus = st.server.served_gpus.len();
                let client_gpus = st.client.gpu_pool.len();
                ui.label(
                    RichText::new(format!(
                        "GPUs: {} server, {} client pool",
                        server_gpus, client_gpus
                    ))
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
                UiTab::Dashboard => {
                    let mut st = self.state.lock().unwrap();
                    panels::dashboard::show(ui, &mut st);
                }
                UiTab::Server => {
                    let mut st = self.state.lock().unwrap();
                    panels::server_panel::show(ui, &mut st);
                }
                UiTab::Client => {
                    let mut st = self.state.lock().unwrap();
                    panels::client_panel::show(ui, &mut st);
                }
                UiTab::ConfigEditor => {
                    let mut st = self.state.lock().unwrap();
                    panels::config_editor::show(ui, &mut st);
                }
            }
        });
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        // Signal the fetcher to stop (and stop services if running embedded)
        if let Ok(mut st) = self.state.lock() {
            st.should_stop = true;
            st.server.stop_requested = true;
            st.client.stop_requested = true;
        }
    }
}

/// Format a status label for the status bar.
fn status_label(role: &str, status: &ServiceStatus) -> (String, Color32) {
    match status {
        ServiceStatus::Unknown => (format!("{}: --", role), Color32::from_rgb(150, 150, 150)),
        ServiceStatus::Stopped => (format!("{}: Off", role), Color32::from_rgb(150, 150, 150)),
        ServiceStatus::Starting => {
            (format!("{}: Starting", role), Color32::from_rgb(255, 200, 50))
        }
        ServiceStatus::Running => {
            (format!("{}: Running", role), Color32::from_rgb(100, 200, 100))
        }
        ServiceStatus::Stopping => {
            (format!("{}: Stopping", role), Color32::from_rgb(255, 165, 0))
        }
        ServiceStatus::Error(_) => {
            (format!("{}: Error", role), Color32::from_rgb(255, 100, 100))
        }
    }
}
