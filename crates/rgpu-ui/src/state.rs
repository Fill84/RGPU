use std::collections::VecDeque;

use rgpu_core::config::{RgpuConfig, TokenEntry, TransportMode};
use rgpu_protocol::gpu_info::GpuInfo;

/// Maximum number of metrics history entries (ring buffer).
/// At 2s poll interval, 300 entries = 10 minutes of history.
pub const MAX_METRICS_HISTORY: usize = 300;

/// Maximum number of error log entries.
pub const MAX_ERROR_LOG: usize = 50;

// ============================================================================
// Shared types (used by both server and client roles)
// ============================================================================

/// How a service was detected or is being managed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServiceOrigin {
    /// Not detected / not running
    NotDetected,
    /// Running as a Windows service managed by SCM
    WindowsService,
    /// Running as an external process (started via CLI)
    ExternalProcess,
    /// Embedded in this UI process
    Embedded,
}

/// Generic service status.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServiceStatus {
    /// Not yet probed
    Unknown,
    Stopped,
    Starting,
    Running,
    Stopping,
    Error(String),
}

impl ServiceStatus {
    pub fn is_running(&self) -> bool {
        matches!(self, Self::Running)
    }
}

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

/// Compute rates from two consecutive snapshots.
fn compute_rates(prev: &MetricsSnapshot, curr: &MetricsSnapshot) -> MetricsRates {
    let dt = curr.timestamp.duration_since(prev.timestamp).as_secs_f64();
    if dt > 0.0 {
        MetricsRates {
            requests_per_sec: (curr.requests_total.saturating_sub(prev.requests_total)) as f64
                / dt,
            cuda_per_sec: (curr.cuda_commands.saturating_sub(prev.cuda_commands)) as f64 / dt,
            vulkan_per_sec: (curr.vulkan_commands.saturating_sub(prev.vulkan_commands)) as f64
                / dt,
            errors_per_sec: (curr.errors_total.saturating_sub(prev.errors_total)) as f64 / dt,
        }
    } else {
        MetricsRates::default()
    }
}

// ============================================================================
// Server role state
// ============================================================================

/// Configuration for the server, edited in the Server panel.
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

/// Server role state — GPUs being served, metrics, connected clients.
pub struct ServerRoleState {
    /// How the server is running
    pub origin: ServiceOrigin,
    pub status: ServiceStatus,

    /// When the server was last started (for startup grace period)
    pub start_time: Option<std::time::Instant>,

    /// GPUs being served (from server's QueryGpus or direct gpu_infos())
    pub served_gpus: Vec<GpuInfo>,

    /// Server metrics
    pub metrics: Option<MetricsSnapshot>,
    pub metrics_history: VecDeque<MetricsSnapshot>,
    pub rates: MetricsRates,

    /// Server configuration (editable when stopped)
    pub config: LocalServerConfig,

    /// Connected client count (from metrics.connections_active)
    pub connected_clients: u32,

    // Control signals (UI -> fetcher)
    pub start_requested: bool,
    pub stop_requested: bool,
}

impl ServerRoleState {
    pub fn new() -> Self {
        Self {
            origin: ServiceOrigin::NotDetected,
            status: ServiceStatus::Unknown,
            start_time: None,
            served_gpus: Vec::new(),
            metrics: None,
            metrics_history: VecDeque::with_capacity(MAX_METRICS_HISTORY),
            rates: MetricsRates::default(),
            config: LocalServerConfig::default(),
            connected_clients: 0,
            start_requested: false,
            stop_requested: false,
        }
    }

    /// Returns true if we're within the startup grace period.
    pub fn in_startup_grace(&self) -> bool {
        if let Some(t) = self.start_time {
            t.elapsed().as_secs() < crate::state::STARTUP_GRACE_SECS
        } else {
            false
        }
    }

    /// Push a metrics snapshot and compute rates from previous.
    pub fn push_metrics(&mut self, snapshot: MetricsSnapshot) {
        if let Some(prev) = self.metrics_history.back() {
            self.rates = compute_rates(prev, &snapshot);
        }
        self.connected_clients = snapshot.connections_active;
        if self.metrics_history.len() >= MAX_METRICS_HISTORY {
            self.metrics_history.pop_front();
        }
        self.metrics = Some(snapshot.clone());
        self.metrics_history.push_back(snapshot);
    }

    /// Clear all monitoring data (on server stop).
    pub fn clear_monitoring(&mut self) {
        self.served_gpus.clear();
        self.metrics = None;
        self.metrics_history.clear();
        self.rates = MetricsRates::default();
        self.connected_clients = 0;
    }
}

// ============================================================================
// Client role state
// ============================================================================

/// Status of interpose library installation.
#[derive(Debug, Clone, Default)]
pub struct InterposeStatus {
    pub cuda: Option<bool>,
    pub vulkan: Option<bool>,
    pub nvenc: Option<bool>,
    pub nvdec: Option<bool>,
    pub nvml: Option<bool>,
}

/// A remote server as seen from the client daemon's perspective.
#[derive(Debug, Clone)]
pub struct ClientRemoteServer {
    pub address: String,
    pub server_id: Option<u16>,
    pub connected: bool,
    pub gpu_count: usize,
}

/// Client role state — GPU pool, connected servers, interpose status.
pub struct ClientRoleState {
    /// How the client daemon is running
    pub origin: ServiceOrigin,
    pub status: ServiceStatus,

    /// When the daemon was last started (for startup grace period)
    pub start_time: Option<std::time::Instant>,

    /// GPU pool from the daemon (local + remote, ordered)
    pub gpu_pool: Vec<GpuInfo>,

    /// Remote servers the daemon is connected to (derived from GPU pool)
    pub remote_servers: Vec<ClientRemoteServer>,

    /// Interpose library installation status
    pub interpose: InterposeStatus,

    /// Client metrics (if daemon supports QueryMetrics)
    pub metrics: Option<MetricsSnapshot>,
    pub metrics_history: VecDeque<MetricsSnapshot>,
    pub rates: MetricsRates,

    // Control signals
    pub start_requested: bool,
    pub stop_requested: bool,
}

/// Grace period (seconds) after starting a daemon before declaring it failed.
pub const STARTUP_GRACE_SECS: u64 = 10;

impl ClientRoleState {
    pub fn new() -> Self {
        Self {
            origin: ServiceOrigin::NotDetected,
            status: ServiceStatus::Unknown,
            start_time: None,
            gpu_pool: Vec::new(),
            remote_servers: Vec::new(),
            interpose: InterposeStatus::default(),
            metrics: None,
            metrics_history: VecDeque::with_capacity(MAX_METRICS_HISTORY),
            rates: MetricsRates::default(),
            start_requested: false,
            stop_requested: false,
        }
    }

    /// Returns true if we're within the startup grace period.
    pub fn in_startup_grace(&self) -> bool {
        if let Some(t) = self.start_time {
            t.elapsed().as_secs() < STARTUP_GRACE_SECS
        } else {
            false
        }
    }

    /// Push a metrics snapshot and compute rates.
    pub fn push_metrics(&mut self, snapshot: MetricsSnapshot) {
        if let Some(prev) = self.metrics_history.back() {
            self.rates = compute_rates(prev, &snapshot);
        }
        if self.metrics_history.len() >= MAX_METRICS_HISTORY {
            self.metrics_history.pop_front();
        }
        self.metrics = Some(snapshot.clone());
        self.metrics_history.push_back(snapshot);
    }

    /// Clear all monitoring data (on daemon stop).
    pub fn clear_monitoring(&mut self) {
        self.gpu_pool.clear();
        self.remote_servers.clear();
        self.metrics = None;
        self.metrics_history.clear();
        self.rates = MetricsRates::default();
    }
}

// ============================================================================
// Config editor state (shared)
// ============================================================================

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

// ============================================================================
// Top-level UI state
// ============================================================================

/// Active tab in the UI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UiTab {
    Dashboard,
    Server,
    Client,
    ConfigEditor,
}

/// State shared between the eframe UI thread and the background data-fetching thread.
pub struct UiState {
    pub active_tab: UiTab,
    pub config_path: String,
    pub poll_interval_secs: u64,
    pub error_log: VecDeque<String>,
    /// Signal the fetcher to stop
    pub should_stop: bool,

    // Role states
    pub server: ServerRoleState,
    pub client: ClientRoleState,

    // Config editor (shared, applies to both roles)
    pub config_editor: Option<ConfigEditorState>,
}

impl UiState {
    pub fn new(config_path: String, poll_interval_secs: u64) -> Self {
        // Try to load config for editor
        let config_editor = RgpuConfig::load(&config_path)
            .ok()
            .map(ConfigEditorState::from_config);

        // Pre-populate server config from loaded config if available
        let mut server = ServerRoleState::new();
        if let Some(ref editor) = config_editor {
            server.config.server_id = editor.config.server.server_id;
            server.config.port = editor.config.server.port;
            server.config.bind = editor.config.server.bind.clone();
            server.config.transport = editor.config.server.transport.clone();
            server.config.cert_path = editor.config.server.cert_path.clone().unwrap_or_default();
            server.config.key_path = editor.config.server.key_path.clone().unwrap_or_default();
            server.config.max_clients = editor.config.server.max_clients;
            server.config.tokens = editor.config.security.tokens.clone();
        }

        Self {
            active_tab: UiTab::Dashboard,
            config_path,
            poll_interval_secs,
            error_log: VecDeque::with_capacity(MAX_ERROR_LOG),
            should_stop: false,
            server,
            client: ClientRoleState::new(),
            config_editor,
        }
    }

    pub fn push_error(&mut self, msg: String) {
        if self.error_log.len() >= MAX_ERROR_LOG {
            self.error_log.pop_front();
        }
        self.error_log.push_back(msg);
    }

    /// Total GPU count across both roles.
    pub fn total_gpus(&self) -> usize {
        self.server.served_gpus.len() + self.client.gpu_pool.len()
    }
}
