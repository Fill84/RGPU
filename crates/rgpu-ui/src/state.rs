use std::collections::VecDeque;

use rgpu_core::config::{RgpuConfig, TokenEntry, TransportMode};
use rgpu_protocol::gpu_info::GpuInfo;

/// Maximum number of metrics history entries (ring buffer).
/// At 2s poll interval, 300 entries = 10 minutes of history.
pub const MAX_METRICS_HISTORY: usize = 300;

/// Maximum number of error log entries.
pub const MAX_ERROR_LOG: usize = 50;

/// Metrics snapshot from a single server at a point in time.
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    pub timestamp: std::time::Instant,
    pub connections_total: u64,
    pub connections_active: u32,
    pub requests_total: u64,
    pub errors_total: u64,
    pub cuda_commands: u64,
    pub vulkan_commands: u64,
    pub uptime_secs: u64,
}

/// Derived per-second rates computed between successive snapshots.
#[derive(Debug, Clone, Default)]
pub struct MetricsRates {
    pub requests_per_sec: f64,
    pub cuda_per_sec: f64,
    pub vulkan_per_sec: f64,
    pub errors_per_sec: f64,
}

/// Connection state for a server target.
#[derive(Debug, Clone)]
pub enum ServerConnectionState {
    Disconnected,
    Connecting,
    Connected,
    Error(String),
}

impl ServerConnectionState {
    pub fn is_connected(&self) -> bool {
        matches!(self, Self::Connected)
    }
}

/// Per-server state tracked by the UI.
#[derive(Debug, Clone)]
pub struct ServerState {
    pub address: String,
    pub token: String,
    pub server_id: Option<u16>,
    pub connection_state: ServerConnectionState,
    pub gpus: Vec<GpuInfo>,
    pub metrics_history: VecDeque<MetricsSnapshot>,
    pub current_rates: MetricsRates,
    /// Set by UI to request disconnect; fetcher will drop the connection.
    pub should_disconnect: bool,
}

impl ServerState {
    pub fn new(address: String, token: String) -> Self {
        Self {
            address,
            token,
            server_id: None,
            connection_state: ServerConnectionState::Disconnected,
            gpus: Vec::new(),
            metrics_history: VecDeque::with_capacity(MAX_METRICS_HISTORY),
            current_rates: MetricsRates::default(),
            should_disconnect: false,
        }
    }

    pub fn latest_metrics(&self) -> Option<&MetricsSnapshot> {
        self.metrics_history.back()
    }

    pub fn push_metrics(&mut self, snapshot: MetricsSnapshot) {
        // Compute rates from previous snapshot
        if let Some(prev) = self.metrics_history.back() {
            let dt = snapshot.timestamp.duration_since(prev.timestamp).as_secs_f64();
            if dt > 0.0 {
                self.current_rates = MetricsRates {
                    requests_per_sec: (snapshot.requests_total.saturating_sub(prev.requests_total))
                        as f64
                        / dt,
                    cuda_per_sec: (snapshot.cuda_commands.saturating_sub(prev.cuda_commands))
                        as f64
                        / dt,
                    vulkan_per_sec: (snapshot.vulkan_commands.saturating_sub(prev.vulkan_commands))
                        as f64
                        / dt,
                    errors_per_sec: (snapshot.errors_total.saturating_sub(prev.errors_total))
                        as f64
                        / dt,
                };
            }
        }

        if self.metrics_history.len() >= MAX_METRICS_HISTORY {
            self.metrics_history.pop_front();
        }
        self.metrics_history.push_back(snapshot);
    }
}

/// Active tab in the UI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UiTab {
    Control,
    GpuOverview,
    Metrics,
    ConfigEditor,
}

/// Status of the embedded server running inside the UI process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LocalServerStatus {
    Stopped,
    Starting,
    Running,
    Stopping,
    Error(String),
}

/// Configuration for the embedded server, edited in the Control panel.
#[derive(Debug, Clone)]
pub struct LocalServerConfig {
    pub server_id: u16,
    pub port: u16,
    pub bind: String,
    pub transport: TransportMode,
    pub cert_path: String,
    pub key_path: String,
    pub max_clients: u32,
    pub tokens: Vec<TokenEntry>,
    // Editing helpers for the "add token" form
    pub new_token_name: String,
    pub new_token_value: String,
}

impl Default for LocalServerConfig {
    fn default() -> Self {
        Self {
            server_id: 0,
            port: 9876,
            bind: "0.0.0.0".to_string(),
            transport: TransportMode::default(),
            cert_path: String::new(),
            key_path: String::new(),
            max_clients: 16,
            tokens: Vec::new(),
            new_token_name: String::new(),
            new_token_value: String::new(),
        }
    }
}

/// A pending connection request from the UI to be processed by the fetcher.
#[derive(Debug, Clone)]
pub struct PendingConnection {
    pub address: String,
    pub token: String,
}

/// Editable configuration state for the config editor.
#[derive(Debug, Clone)]
pub struct ConfigEditorState {
    pub config: RgpuConfig,
    pub dirty: bool,
    /// New server endpoint fields being edited
    pub new_server_address: String,
    pub new_server_token: String,
    /// New token fields being edited
    pub new_token_value: String,
    pub new_token_name: String,
}

impl ConfigEditorState {
    pub fn from_config(config: RgpuConfig) -> Self {
        Self {
            config,
            dirty: false,
            new_server_address: String::new(),
            new_server_token: String::new(),
            new_token_value: String::new(),
            new_token_name: String::new(),
        }
    }
}

/// State shared between the eframe UI thread and the background data-fetching thread.
pub struct UiState {
    pub servers: Vec<ServerState>,
    pub config_path: String,
    pub config_editor: Option<ConfigEditorState>,
    pub active_tab: UiTab,
    pub poll_interval_secs: u64,
    pub error_log: VecDeque<String>,
    /// Signal the fetcher to stop
    pub should_stop: bool,

    // --- Embedded server control ---
    pub local_server_status: LocalServerStatus,
    pub local_server_config: LocalServerConfig,
    /// Set by UI to request server start; cleared by fetcher after processing.
    pub server_start_requested: bool,
    /// Set by UI to request server stop; cleared by fetcher after processing.
    pub server_stop_requested: bool,

    // --- Dynamic connections ---
    /// Pending connections to be established by the fetcher.
    pub pending_connections: Vec<PendingConnection>,

    // --- Connection form state ---
    pub new_connection_address: String,
    pub new_connection_token: String,

    // --- Embedded server monitoring (direct, no TCP) ---
    pub embedded_server_gpus: Vec<GpuInfo>,
    pub embedded_server_metrics: Option<MetricsSnapshot>,
}

impl UiState {
    pub fn new(
        servers: Vec<(String, String)>,
        config_path: String,
        poll_interval_secs: u64,
    ) -> Self {
        let server_states = servers
            .into_iter()
            .map(|(addr, token)| ServerState::new(addr, token))
            .collect();

        // Try to load config
        let config_editor = RgpuConfig::load(&config_path)
            .ok()
            .map(ConfigEditorState::from_config);

        Self {
            servers: server_states,
            config_path,
            config_editor,
            active_tab: UiTab::Control,
            poll_interval_secs,
            error_log: VecDeque::with_capacity(MAX_ERROR_LOG),
            should_stop: false,
            local_server_status: LocalServerStatus::Stopped,
            local_server_config: LocalServerConfig::default(),
            server_start_requested: false,
            server_stop_requested: false,
            pending_connections: Vec::new(),
            new_connection_address: String::new(),
            new_connection_token: String::new(),
            embedded_server_gpus: Vec::new(),
            embedded_server_metrics: None,
        }
    }

    pub fn push_error(&mut self, msg: String) {
        if self.error_log.len() >= MAX_ERROR_LOG {
            self.error_log.pop_front();
        }
        self.error_log.push_back(msg);
    }

    pub fn total_gpus(&self) -> usize {
        let remote: usize = self.servers.iter().map(|s| s.gpus.len()).sum();
        remote + self.embedded_server_gpus.len()
    }

    pub fn connected_servers(&self) -> usize {
        self.servers
            .iter()
            .filter(|s| s.connection_state.is_connected())
            .count()
    }
}
