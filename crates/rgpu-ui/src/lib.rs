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
    let icon_image = image::load_from_memory(include_bytes!("../../../icon.png"))
        .expect("failed to load RGPU icon")
        .to_rgba8();
    let (w, h) = icon_image.dimensions();
    let icon_data = egui::IconData {
        rgba: icon_image.into_raw(),
        width: w,
        height: h,
    };

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_min_inner_size([800.0, 600.0])
            .with_title("RGPU - Remote GPU Manager")
            .with_icon(std::sync::Arc::new(icon_data)),
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
