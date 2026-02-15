mod app;
mod data_fetcher;
pub mod panels;
pub mod state;
pub mod widgets;

use app::RgpuApp;

/// Launch the RGPU desktop GUI.
///
/// This function blocks until the window is closed.
///
/// # Arguments
/// * `servers` - List of (address, token) pairs to connect to
/// * `config_path` - Path to rgpu.toml for the config editor
/// * `poll_interval` - Metrics poll interval in seconds
pub fn launch_ui(
    servers: Vec<(String, String)>,
    config_path: String,
    poll_interval: u64,
) -> anyhow::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_min_inner_size([800.0, 600.0])
            .with_title("RGPU - Remote GPU Manager"),
        ..Default::default()
    };

    eframe::run_native(
        "RGPU",
        options,
        Box::new(move |cc| {
            Ok(Box::new(RgpuApp::new(cc, servers, config_path, poll_interval)))
        }),
    )
    .map_err(|e| anyhow::anyhow!("eframe error: {}", e))
}
