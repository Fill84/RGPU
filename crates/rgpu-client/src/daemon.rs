use std::sync::Arc;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::TcpStream;
use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};

use rgpu_core::config::{ClientConfig, ServerEndpoint, TransportMode};
use rgpu_protocol::cuda_commands::{CudaCommand, CudaResponse};
use rgpu_protocol::gpu_info::GpuInfo;
use rgpu_protocol::handle::NetworkHandle;
use rgpu_protocol::messages::{Message, RequestId, PROTOCOL_VERSION};
use rgpu_protocol::vulkan_commands::{VulkanCommand, VulkanResponse};
use rgpu_protocol::wire;
use rgpu_transport::auth;
use rgpu_transport::quic::QuicConnection;

use crate::pool_manager::{ConnectionStatus, GpuPoolManager};

/// Transport-specific connection variant.
enum TransportConn {
    Tcp {
        reader: OwnedReadHalf,
        writer: OwnedWriteHalf,
    },
    Quic(QuicConnection),
}

/// A persistent, authenticated connection to an RGPU server.
struct ServerConn {
    transport: TransportConn,
    _address: String,
    _token: String,
}

impl ServerConn {
    /// Send a message and wait for the response on this connection.
    async fn send_and_receive(
        &mut self,
        msg: &Message,
    ) -> Result<Message, Box<dyn std::error::Error + Send + Sync>> {
        match &mut self.transport {
            TransportConn::Tcp { reader, writer } => {
                let frame = wire::encode_message(msg, 0)?;
                writer.write_all(&frame).await?;
                read_message(reader).await
            }
            TransportConn::Quic(quic) => {
                Ok(quic.send_and_receive(msg).await?)
            }
        }
    }
}

/// The RGPU client daemon. Connects to servers, manages the GPU pool,
/// and listens for IPC connections from the Vulkan ICD and CUDA interposition library.
pub struct ClientDaemon {
    config: ClientConfig,
    pool_manager: Arc<GpuPoolManager>,
    /// Cached GPU info from connected servers
    cached_gpus: Arc<tokio::sync::RwLock<Vec<GpuInfo>>>,
    /// Persistent server connections (one per server, behind Mutex for exclusive access)
    server_conns: Arc<tokio::sync::RwLock<Vec<Arc<Mutex<Option<ServerConn>>>>>>,
    /// Server endpoints for reconnection
    endpoints: Arc<tokio::sync::RwLock<Vec<ServerEndpoint>>>,
}

impl ClientDaemon {
    pub fn new(config: ClientConfig) -> Self {
        let ordering = config.gpu_ordering.clone();
        Self {
            config,
            pool_manager: Arc::new(GpuPoolManager::new(ordering)),
            cached_gpus: Arc::new(tokio::sync::RwLock::new(Vec::new())),
            server_conns: Arc::new(tokio::sync::RwLock::new(Vec::new())),
            endpoints: Arc::new(tokio::sync::RwLock::new(Vec::new())),
        }
    }

    /// Start the client daemon: connect to servers and start IPC listener.
    pub async fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("RGPU client daemon starting");

        // Connect to all configured servers and discover GPUs
        for (server_index, server) in self.config.servers.iter().enumerate() {
            match self.connect_and_discover(server).await {
                Ok((gpus, conn, server_id)) => {
                    info!(
                        "connected to server {} (id={}, {} GPUs)",
                        server.address,
                        server_id,
                        gpus.len()
                    );

                    // Register server in pool manager
                    self.pool_manager
                        .add_server(server.clone(), server_id, gpus.clone())
                        .await;

                    self.cached_gpus.write().await.extend(gpus);

                    // Store the persistent connection
                    self.server_conns
                        .write()
                        .await
                        .push(Arc::new(Mutex::new(Some(conn))));
                    self.endpoints.write().await.push(server.clone());
                }
                Err(e) => {
                    error!("failed to connect to {}: {}", server.address, e);
                    // Store a None connection slot for potential reconnection
                    self.server_conns
                        .write()
                        .await
                        .push(Arc::new(Mutex::new(None)));
                    self.endpoints.write().await.push(server.clone());
                    // Register as disconnected in pool manager
                    self.pool_manager
                        .add_server(server.clone(), server_index as u16, Vec::new())
                        .await;
                    self.pool_manager
                        .set_server_status(
                            server_index,
                            ConnectionStatus::Disconnected(e.to_string()),
                        )
                        .await;
                }
            }
        }

        // Apply GPU ordering after all servers are connected
        self.pool_manager.apply_ordering().await;

        let total_gpus = self.cached_gpus.read().await.len();
        info!("GPU pool ready: {} GPU(s) total", total_gpus);

        // Spawn background reconnection task
        let reconnect_conns = self.server_conns.clone();
        let reconnect_endpoints = self.endpoints.clone();
        let reconnect_pool = self.pool_manager.clone();
        tokio::spawn(async move {
            reconnection_loop(reconnect_conns, reconnect_endpoints, reconnect_pool).await;
        });

        // Start IPC listener for local applications
        let ipc_path = rgpu_common::platform::default_ipc_path();
        let cached_gpus = self.cached_gpus.clone();
        let server_conns = self.server_conns.clone();
        let endpoints = self.endpoints.clone();
        let pool_manager = self.pool_manager.clone();

        info!("starting IPC listener on {}", ipc_path);

        let ipc_future = crate::ipc::start_ipc_listener(&ipc_path, move |msg| {
            handle_ipc_message(&cached_gpus, &server_conns, &endpoints, &pool_manager, msg)
        });

        tokio::select! {
            result = ipc_future => { result?; }
            _ = client_shutdown_signal() => {
                info!("client daemon shutting down");
            }
        }

        Ok(())
    }

    /// Connect to a server, perform handshake, and return discovered GPUs + persistent connection + server_id.
    async fn connect_and_discover(
        &self,
        endpoint: &ServerEndpoint,
    ) -> Result<(Vec<GpuInfo>, ServerConn, u16), Box<dyn std::error::Error + Send + Sync>> {
        info!("connecting to server: {} ({:?})", endpoint.address, endpoint.transport);

        match endpoint.transport {
            TransportMode::Quic => self.connect_and_discover_quic(endpoint).await,
            TransportMode::Tcp => self.connect_and_discover_tcp(endpoint).await,
        }
    }

    /// TCP connect + handshake.
    async fn connect_and_discover_tcp(
        &self,
        endpoint: &ServerEndpoint,
    ) -> Result<(Vec<GpuInfo>, ServerConn, u16), Box<dyn std::error::Error + Send + Sync>> {
        let stream = TcpStream::connect(&endpoint.address).await?;
        let (mut reader, mut writer) = stream.into_split();

        // Send Hello
        let hello = Message::Hello {
            protocol_version: PROTOCOL_VERSION,
            name: "RGPU Client".to_string(),
            challenge: None,
        };
        let frame = wire::encode_message(&hello, 0)?;
        writer.write_all(&frame).await?;

        // Read server Hello
        let server_hello = read_message(&mut reader).await?;

        let challenge = match &server_hello {
            Message::Hello { challenge, .. } => challenge.clone().unwrap_or_default(),
            _ => Vec::new(),
        };

        // Send auth
        let challenge_response = auth::compute_challenge_response(&endpoint.token, &challenge);
        let auth_msg = Message::Authenticate {
            token: endpoint.token.clone(),
            challenge_response,
        };
        let frame = wire::encode_message(&auth_msg, 0)?;
        writer.write_all(&frame).await?;

        // Read auth result
        let auth_result = read_message(&mut reader).await?;

        parse_auth_result(auth_result, endpoint, TransportConn::Tcp { reader, writer })
    }

    /// QUIC connect + handshake.
    async fn connect_and_discover_quic(
        &self,
        endpoint: &ServerEndpoint,
    ) -> Result<(Vec<GpuInfo>, ServerConn, u16), Box<dyn std::error::Error + Send + Sync>> {
        let quic_conn = rgpu_transport::quic::connect_quic_client(&endpoint.address).await?;

        // Perform handshake over QUIC
        let hello = Message::Hello {
            protocol_version: PROTOCOL_VERSION,
            name: "RGPU Client".to_string(),
            challenge: None,
        };
        let server_hello = quic_conn.send_and_receive(&hello).await?;

        let challenge = match &server_hello {
            Message::Hello { challenge, .. } => challenge.clone().unwrap_or_default(),
            _ => Vec::new(),
        };

        let challenge_response = auth::compute_challenge_response(&endpoint.token, &challenge);
        let auth_msg = Message::Authenticate {
            token: endpoint.token.clone(),
            challenge_response,
        };
        let auth_result = quic_conn.send_and_receive(&auth_msg).await?;

        parse_auth_result(auth_result, endpoint, TransportConn::Quic(quic_conn))
    }
}

/// Read a single framed message from a reader.
async fn read_message<R: tokio::io::AsyncRead + Unpin>(
    reader: &mut R,
) -> Result<Message, Box<dyn std::error::Error + Send + Sync>> {
    let mut header_buf = [0u8; wire::HEADER_SIZE];
    reader.read_exact(&mut header_buf).await?;
    let (flags, _, payload_len) = wire::decode_header(&header_buf)?;
    let mut payload = vec![0u8; payload_len as usize];
    reader.read_exact(&mut payload).await?;
    let msg = wire::decode_message(&payload, flags)?;
    Ok(msg)
}

/// Parse an AuthResult message into GPUs + ServerConn.
fn parse_auth_result(
    auth_result: Message,
    endpoint: &ServerEndpoint,
    transport: TransportConn,
) -> Result<(Vec<GpuInfo>, ServerConn, u16), Box<dyn std::error::Error + Send + Sync>> {
    match auth_result {
        Message::AuthResult {
            success: true,
            server_id,
            available_gpus,
            ..
        } => {
            let sid = server_id.unwrap_or(0);
            info!(
                "authenticated with server (id={}), {} GPU(s): {}",
                sid,
                available_gpus.len(),
                available_gpus
                    .iter()
                    .map(|g| g.device_name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            let conn = ServerConn {
                transport,
                _address: endpoint.address.clone(),
                _token: endpoint.token.clone(),
            };
            Ok((available_gpus, conn, sid))
        }
        Message::AuthResult {
            success: false,
            error_message,
            ..
        } => Err(format!(
            "authentication failed: {}",
            error_message.unwrap_or_default()
        )
        .into()),
        _ => Err("unexpected response during auth".into()),
    }
}

/// Establish a new authenticated connection to a server.
async fn reconnect(
    endpoint: &ServerEndpoint,
) -> Result<(ServerConn, u16), Box<dyn std::error::Error + Send + Sync>> {
    info!("reconnecting to server: {} ({:?})", endpoint.address, endpoint.transport);

    match endpoint.transport {
        TransportMode::Quic => reconnect_quic(endpoint).await,
        TransportMode::Tcp => reconnect_tcp(endpoint).await,
    }
}

/// TCP reconnect.
async fn reconnect_tcp(
    endpoint: &ServerEndpoint,
) -> Result<(ServerConn, u16), Box<dyn std::error::Error + Send + Sync>> {
    let stream = TcpStream::connect(&endpoint.address).await?;
    let (mut reader, mut writer) = stream.into_split();

    // Hello
    let hello = Message::Hello {
        protocol_version: PROTOCOL_VERSION,
        name: "RGPU Client".to_string(),
        challenge: None,
    };
    let frame = wire::encode_message(&hello, 0)?;
    writer.write_all(&frame).await?;

    let server_hello = read_message(&mut reader).await?;
    let challenge = match &server_hello {
        Message::Hello { challenge, .. } => challenge.clone().unwrap_or_default(),
        _ => Vec::new(),
    };

    // Auth
    let challenge_response = auth::compute_challenge_response(&endpoint.token, &challenge);
    let auth_msg = Message::Authenticate {
        token: endpoint.token.clone(),
        challenge_response,
    };
    let frame = wire::encode_message(&auth_msg, 0)?;
    writer.write_all(&frame).await?;

    let auth_result = read_message(&mut reader).await?;
    match auth_result {
        Message::AuthResult {
            success: true,
            server_id,
            ..
        } => {
            let sid = server_id.unwrap_or(0);
            info!("reconnected to server {} (id={})", endpoint.address, sid);
            Ok((
                ServerConn {
                    transport: TransportConn::Tcp { reader, writer },
                    _address: endpoint.address.clone(),
                    _token: endpoint.token.clone(),
                },
                sid,
            ))
        }
        _ => Err("reconnection auth failed".into()),
    }
}

/// QUIC reconnect.
async fn reconnect_quic(
    endpoint: &ServerEndpoint,
) -> Result<(ServerConn, u16), Box<dyn std::error::Error + Send + Sync>> {
    let quic_conn = rgpu_transport::quic::connect_quic_client(&endpoint.address).await?;

    // Hello
    let hello = Message::Hello {
        protocol_version: PROTOCOL_VERSION,
        name: "RGPU Client".to_string(),
        challenge: None,
    };
    let server_hello = quic_conn.send_and_receive(&hello).await?;
    let challenge = match &server_hello {
        Message::Hello { challenge, .. } => challenge.clone().unwrap_or_default(),
        _ => Vec::new(),
    };

    // Auth
    let challenge_response = auth::compute_challenge_response(&endpoint.token, &challenge);
    let auth_msg = Message::Authenticate {
        token: endpoint.token.clone(),
        challenge_response,
    };
    let auth_result = quic_conn.send_and_receive(&auth_msg).await?;

    match auth_result {
        Message::AuthResult {
            success: true,
            server_id,
            ..
        } => {
            let sid = server_id.unwrap_or(0);
            info!("reconnected to server {} via QUIC (id={})", endpoint.address, sid);
            Ok((
                ServerConn {
                    transport: TransportConn::Quic(quic_conn),
                    _address: endpoint.address.clone(),
                    _token: endpoint.token.clone(),
                },
                sid,
            ))
        }
        _ => Err("reconnection auth failed".into()),
    }
}

// ── Handle Extraction ────────────────────────────────────────────────

/// Extract the primary NetworkHandle from a CUDA command for routing.
/// Returns None for creation/global commands that don't target a specific server.
fn extract_cuda_routing_handle(cmd: &CudaCommand) -> Option<NetworkHandle> {
    match cmd {
        // Global / creation commands — no routing handle
        CudaCommand::Init { .. }
        | CudaCommand::DriverGetVersion
        | CudaCommand::DeviceGetCount
        | CudaCommand::DeviceGet { .. }
        | CudaCommand::DeviceGetByPCIBusId { .. }
        | CudaCommand::CtxGetCurrent
        | CudaCommand::CtxSynchronize
        | CudaCommand::CtxPopCurrent
        | CudaCommand::CtxGetDevice
        | CudaCommand::CtxSetCacheConfig { .. }
        | CudaCommand::CtxGetCacheConfig
        | CudaCommand::CtxSetLimit { .. }
        | CudaCommand::CtxGetLimit { .. }
        | CudaCommand::CtxGetStreamPriorityRange
        | CudaCommand::CtxGetFlags
        | CudaCommand::CtxSetFlags { .. }
        | CudaCommand::CtxResetPersistingL2Cache
        | CudaCommand::ModuleLoadData { .. }
        | CudaCommand::ModuleLoad { .. }
        | CudaCommand::ModuleLoadDataEx { .. }
        | CudaCommand::ModuleLoadFatBinary { .. }
        | CudaCommand::LinkCreate { .. }
        | CudaCommand::MemAlloc { .. }
        | CudaCommand::MemAllocHost { .. }
        | CudaCommand::MemHostAlloc { .. }
        | CudaCommand::MemAllocManaged { .. }
        | CudaCommand::MemAllocPitch { .. }
        | CudaCommand::MemHostRegister { .. }
        | CudaCommand::MemGetInfo
        | CudaCommand::StreamCreate { .. }
        | CudaCommand::StreamCreateWithPriority { .. }
        | CudaCommand::EventCreate { .. } => None,

        // Device management — route via device handle
        CudaCommand::DeviceGetName { device, .. }
        | CudaCommand::DeviceGetAttribute { device, .. }
        | CudaCommand::DeviceTotalMem { device, .. }
        | CudaCommand::DeviceComputeCapability { device, .. }
        | CudaCommand::DeviceGetUuid { device }
        | CudaCommand::DeviceGetPCIBusId { device }
        | CudaCommand::DeviceGetDefaultMemPool { device }
        | CudaCommand::DeviceGetMemPool { device }
        | CudaCommand::DeviceSetMemPool { device, .. }
        | CudaCommand::DeviceGetTexture1DLinearMaxWidth { device, .. }
        | CudaCommand::DeviceGetExecAffinitySupport { device, .. }
        | CudaCommand::DevicePrimaryCtxRetain { device }
        | CudaCommand::DevicePrimaryCtxRelease { device }
        | CudaCommand::DevicePrimaryCtxReset { device }
        | CudaCommand::DevicePrimaryCtxGetState { device }
        | CudaCommand::DevicePrimaryCtxSetFlags { device, .. }
        | CudaCommand::MemPoolCreate { device, .. } => Some(*device),

        // P2P — route via src device
        CudaCommand::DeviceGetP2PAttribute { src_device, .. } => Some(*src_device),
        CudaCommand::DeviceCanAccessPeer { device, .. } => Some(*device),

        // Context management — route via context handle
        CudaCommand::CtxCreate { device, .. } => Some(*device),
        CudaCommand::CtxDestroy { ctx, .. } => Some(*ctx),
        CudaCommand::CtxSetCurrent { ctx, .. } => Some(*ctx),
        CudaCommand::CtxPushCurrent { ctx } => Some(*ctx),
        CudaCommand::CtxGetApiVersion { ctx } => Some(*ctx),
        CudaCommand::CtxEnablePeerAccess { peer_ctx, .. } => Some(*peer_ctx),
        CudaCommand::CtxDisablePeerAccess { peer_ctx } => Some(*peer_ctx),

        // Module management — route via module handle
        CudaCommand::ModuleUnload { module, .. } => Some(*module),
        CudaCommand::ModuleGetFunction { module, .. } => Some(*module),
        CudaCommand::ModuleGetGlobal { module, .. } => Some(*module),

        // Linker — route via link handle
        CudaCommand::LinkAddData { link, .. } => Some(*link),
        CudaCommand::LinkAddFile { link, .. } => Some(*link),
        CudaCommand::LinkComplete { link } => Some(*link),
        CudaCommand::LinkDestroy { link } => Some(*link),

        // Memory management — route via memory handle
        CudaCommand::MemFree { dptr, .. } => Some(*dptr),
        CudaCommand::MemcpyHtoD { dst, .. } => Some(*dst),
        CudaCommand::MemcpyDtoH { src, .. } => Some(*src),
        CudaCommand::MemcpyDtoD { dst, .. } => Some(*dst),
        CudaCommand::MemcpyHtoDAsync { dst, .. } => Some(*dst),
        CudaCommand::MemcpyDtoHAsync { src, .. } => Some(*src),
        CudaCommand::MemcpyDtoDAsync { dst, .. } => Some(*dst),
        CudaCommand::MemsetD8 { dst, .. } => Some(*dst),
        CudaCommand::MemsetD16 { dst, .. } => Some(*dst),
        CudaCommand::MemsetD32 { dst, .. } => Some(*dst),
        CudaCommand::MemsetD8Async { dst, .. } => Some(*dst),
        CudaCommand::MemsetD16Async { dst, .. } => Some(*dst),
        CudaCommand::MemsetD32Async { dst, .. } => Some(*dst),
        CudaCommand::MemGetAddressRange { dptr } => Some(*dptr),
        CudaCommand::MemFreeHost { ptr } => Some(*ptr),
        CudaCommand::MemHostGetDevicePointer { host_ptr, .. } => Some(*host_ptr),
        CudaCommand::MemHostGetFlags { host_ptr } => Some(*host_ptr),
        CudaCommand::MemHostUnregister { ptr } => Some(*ptr),
        CudaCommand::MemPrefetchAsync { dptr, .. } => Some(*dptr),
        CudaCommand::MemAdvise { dptr, .. } => Some(*dptr),
        CudaCommand::MemRangeGetAttribute { dptr, .. } => Some(*dptr),
        CudaCommand::MemFreeAsync { dptr, .. } => Some(*dptr),

        // Execution — route via function handle
        CudaCommand::LaunchKernel { func, .. } => Some(*func),
        CudaCommand::LaunchCooperativeKernel { func, .. } => Some(*func),
        CudaCommand::FuncGetAttribute { func, .. } => Some(*func),
        CudaCommand::FuncSetAttribute { func, .. } => Some(*func),
        CudaCommand::FuncSetCacheConfig { func, .. } => Some(*func),
        CudaCommand::FuncSetSharedMemConfig { func, .. } => Some(*func),
        CudaCommand::FuncGetModule { func } => Some(*func),
        CudaCommand::FuncGetName { func } => Some(*func),
        CudaCommand::OccupancyMaxActiveBlocksPerMultiprocessor { func, .. } => Some(*func),
        CudaCommand::OccupancyMaxActiveBlocksPerMultiprocessorWithFlags { func, .. } => Some(*func),
        CudaCommand::OccupancyAvailableDynamicSMemPerBlock { func, .. } => Some(*func),

        // Stream management — route via stream handle
        CudaCommand::StreamDestroy { stream, .. } => Some(*stream),
        CudaCommand::StreamSynchronize { stream, .. } => Some(*stream),
        CudaCommand::StreamQuery { stream, .. } => Some(*stream),
        CudaCommand::StreamWaitEvent { stream, .. } => Some(*stream),
        CudaCommand::StreamGetPriority { stream } => Some(*stream),
        CudaCommand::StreamGetFlags { stream } => Some(*stream),
        CudaCommand::StreamGetCtx { stream } => Some(*stream),

        // Event management — route via event handle
        CudaCommand::EventDestroy { event, .. } => Some(*event),
        CudaCommand::EventRecord { event, .. } => Some(*event),
        CudaCommand::EventRecordWithFlags { event, .. } => Some(*event),
        CudaCommand::EventSynchronize { event, .. } => Some(*event),
        CudaCommand::EventQuery { event, .. } => Some(*event),
        CudaCommand::EventElapsedTime { start, .. } => Some(*start),

        // Pointer queries — route via memory handle
        CudaCommand::PointerGetAttribute { ptr, .. } => Some(*ptr),
        CudaCommand::PointerGetAttributes { ptr, .. } => Some(*ptr),
        CudaCommand::PointerSetAttribute { ptr, .. } => Some(*ptr),

        // Memory pools — route via pool handle
        CudaCommand::MemPoolDestroy { pool } => Some(*pool),
        CudaCommand::MemPoolTrimTo { pool, .. } => Some(*pool),
        CudaCommand::MemPoolSetAttribute { pool, .. } => Some(*pool),
        CudaCommand::MemPoolGetAttribute { pool, .. } => Some(*pool),
        CudaCommand::MemAllocAsync { stream, .. } => Some(*stream),
        CudaCommand::MemAllocFromPoolAsync { pool, .. } => Some(*pool),
    }
}

/// Extract the primary NetworkHandle from a Vulkan command for routing.
/// Returns None for creation/global commands that don't target a specific server.
fn extract_vulkan_routing_handle(cmd: &VulkanCommand) -> Option<NetworkHandle> {
    match cmd {
        // Global / instance creation — no routing handle
        VulkanCommand::CreateInstance { .. }
        | VulkanCommand::EnumerateInstanceExtensionProperties { .. }
        | VulkanCommand::EnumerateInstanceLayerProperties => None,

        // Instance-level commands
        VulkanCommand::DestroyInstance { instance, .. } => Some(*instance),
        VulkanCommand::EnumeratePhysicalDevices { instance, .. } => Some(*instance),

        // Physical device queries
        VulkanCommand::EnumerateDeviceExtensionProperties { physical_device, .. }
        | VulkanCommand::GetPhysicalDeviceProperties { physical_device, .. }
        | VulkanCommand::GetPhysicalDeviceProperties2 { physical_device, .. }
        | VulkanCommand::GetPhysicalDeviceFeatures { physical_device, .. }
        | VulkanCommand::GetPhysicalDeviceFeatures2 { physical_device, .. }
        | VulkanCommand::GetPhysicalDeviceMemoryProperties { physical_device, .. }
        | VulkanCommand::GetPhysicalDeviceMemoryProperties2 { physical_device, .. }
        | VulkanCommand::GetPhysicalDeviceQueueFamilyProperties { physical_device, .. }
        | VulkanCommand::GetPhysicalDeviceQueueFamilyProperties2 { physical_device, .. }
        | VulkanCommand::GetPhysicalDeviceFormatProperties { physical_device, .. } => {
            Some(*physical_device)
        }

        // Device creation routes via physical device
        VulkanCommand::CreateDevice { physical_device, .. } => Some(*physical_device),

        // Device-level commands
        VulkanCommand::DestroyDevice { device, .. }
        | VulkanCommand::DeviceWaitIdle { device, .. }
        | VulkanCommand::GetDeviceQueue { device, .. }
        | VulkanCommand::AllocateMemory { device, .. }
        | VulkanCommand::FreeMemory { device, .. }
        | VulkanCommand::MapMemory { device, .. }
        | VulkanCommand::UnmapMemory { device, .. }
        | VulkanCommand::FlushMappedMemoryRanges { device, .. }
        | VulkanCommand::InvalidateMappedMemoryRanges { device, .. }
        | VulkanCommand::CreateBuffer { device, .. }
        | VulkanCommand::DestroyBuffer { device, .. }
        | VulkanCommand::BindBufferMemory { device, .. }
        | VulkanCommand::GetBufferMemoryRequirements { device, .. }
        | VulkanCommand::CreateShaderModule { device, .. }
        | VulkanCommand::DestroyShaderModule { device, .. }
        | VulkanCommand::CreateDescriptorSetLayout { device, .. }
        | VulkanCommand::DestroyDescriptorSetLayout { device, .. }
        | VulkanCommand::CreatePipelineLayout { device, .. }
        | VulkanCommand::DestroyPipelineLayout { device, .. }
        | VulkanCommand::CreateComputePipelines { device, .. }
        | VulkanCommand::DestroyPipeline { device, .. }
        | VulkanCommand::CreateDescriptorPool { device, .. }
        | VulkanCommand::DestroyDescriptorPool { device, .. }
        | VulkanCommand::AllocateDescriptorSets { device, .. }
        | VulkanCommand::FreeDescriptorSets { device, .. }
        | VulkanCommand::UpdateDescriptorSets { device, .. }
        | VulkanCommand::CreateCommandPool { device, .. }
        | VulkanCommand::DestroyCommandPool { device, .. }
        | VulkanCommand::ResetCommandPool { device, .. }
        | VulkanCommand::AllocateCommandBuffers { device, .. }
        | VulkanCommand::FreeCommandBuffers { device, .. }
        | VulkanCommand::CreateFence { device, .. }
        | VulkanCommand::DestroyFence { device, .. }
        | VulkanCommand::WaitForFences { device, .. }
        | VulkanCommand::ResetFences { device, .. }
        | VulkanCommand::GetFenceStatus { device, .. }
        | VulkanCommand::CreateImage { device, .. }
        | VulkanCommand::DestroyImage { device, .. }
        | VulkanCommand::GetImageMemoryRequirements { device, .. }
        | VulkanCommand::BindImageMemory { device, .. }
        | VulkanCommand::CreateImageView { device, .. }
        | VulkanCommand::DestroyImageView { device, .. }
        | VulkanCommand::CreateRenderPass { device, .. }
        | VulkanCommand::DestroyRenderPass { device, .. }
        | VulkanCommand::CreateFramebuffer { device, .. }
        | VulkanCommand::DestroyFramebuffer { device, .. }
        | VulkanCommand::CreateGraphicsPipelines { device, .. }
        | VulkanCommand::CreateSemaphore { device, .. }
        | VulkanCommand::DestroySemaphore { device, .. } => Some(*device),

        // Queue commands
        VulkanCommand::QueueSubmit { queue, .. }
        | VulkanCommand::QueueWaitIdle { queue, .. } => Some(*queue),

        // Recorded commands route via command buffer
        VulkanCommand::SubmitRecordedCommands { command_buffer, .. } => Some(*command_buffer),
    }
}

// ── Server Resolution ────────────────────────────────────────────────

/// Resolve which server_conns index to use for a given routing handle.
async fn resolve_server_index(
    pool_manager: &GpuPoolManager,
    handle: Option<NetworkHandle>,
) -> usize {
    if let Some(h) = handle {
        if let Some(idx) = pool_manager.server_index_for_handle(&h).await {
            return idx;
        }
    }
    // Fallback: first connected server
    pool_manager.default_server_index().await.unwrap_or(0)
}

// ── IPC Message Handler ──────────────────────────────────────────────

/// Handle an IPC message from a local application (Vulkan ICD or CUDA interpose lib).
fn handle_ipc_message(
    cached_gpus: &Arc<tokio::sync::RwLock<Vec<GpuInfo>>>,
    server_conns: &Arc<tokio::sync::RwLock<Vec<Arc<Mutex<Option<ServerConn>>>>>>,
    endpoints: &Arc<tokio::sync::RwLock<Vec<ServerEndpoint>>>,
    pool_manager: &Arc<GpuPoolManager>,
    msg: Message,
) -> Option<Message> {
    match msg {
        Message::QueryGpus => {
            let gpus = cached_gpus.clone();
            let rt = tokio::runtime::Handle::try_current().ok()?;
            let gpu_list = std::thread::spawn(move || {
                rt.block_on(async { gpus.read().await.clone() })
            })
            .join()
            .ok()?;
            Some(Message::GpuList(gpu_list))
        }

        Message::CudaCommand {
            request_id,
            command,
        } => {
            let conns = server_conns.clone();
            let eps = endpoints.clone();
            let pm = pool_manager.clone();
            let rt = tokio::runtime::Handle::try_current().ok()?;

            let response = std::thread::spawn(move || {
                rt.block_on(async {
                    forward_cuda_command_pooled(&conns, &eps, &pm, request_id, command).await
                })
            })
            .join()
            .ok()?;

            Some(response)
        }

        Message::VulkanCommand {
            request_id,
            command,
        } => {
            let conns = server_conns.clone();
            let eps = endpoints.clone();
            let pm = pool_manager.clone();
            let rt = tokio::runtime::Handle::try_current().ok()?;

            let response = std::thread::spawn(move || {
                rt.block_on(async {
                    forward_vulkan_command_pooled(&conns, &eps, &pm, request_id, command).await
                })
            })
            .join()
            .ok()?;

            Some(response)
        }

        Message::CudaBatch(commands) => {
            // Forward the batch as a single message to the appropriate server.
            let conns = server_conns.clone();
            let eps = endpoints.clone();
            let pm = pool_manager.clone();
            let rt = tokio::runtime::Handle::try_current().ok()?;

            let response = std::thread::spawn(move || {
                rt.block_on(async {
                    // Determine server from first command in batch
                    let routing_handle = commands.first()
                        .and_then(|cmd| extract_cuda_routing_handle(cmd));
                    let server_idx = resolve_server_index(&pm, routing_handle).await;
                    let batch_msg = Message::CudaBatch(commands);
                    let request_id = RequestId(0);
                    forward_to_server(&conns, &eps, server_idx, request_id, &batch_msg, true).await
                })
            })
            .join()
            .ok()?;

            Some(response)
        }

        Message::Ping => Some(Message::Pong),

        _ => {
            warn!("unhandled IPC message");
            None
        }
    }
}

// ── CUDA Forwarding ──────────────────────────────────────────────────

/// Forward a CUDA command using the pooled persistent connection.
/// Routes to the correct server based on handle's server_id.
async fn forward_cuda_command_pooled(
    server_conns: &Arc<tokio::sync::RwLock<Vec<Arc<Mutex<Option<ServerConn>>>>>>,
    endpoints: &Arc<tokio::sync::RwLock<Vec<ServerEndpoint>>>,
    pool_manager: &Arc<GpuPoolManager>,
    request_id: RequestId,
    command: CudaCommand,
) -> Message {
    // Special handling for DeviceGetCount: return pool total
    if matches!(command, CudaCommand::DeviceGetCount) {
        let count = pool_manager.cuda_device_count().await;
        return Message::CudaResponse {
            request_id,
            response: CudaResponse::DeviceCount(count as i32),
        };
    }

    // Special handling for DeviceGet: map pool ordinal to server-local ordinal
    if let CudaCommand::DeviceGet { ordinal } = &command {
        if let Some((server_idx, server_local_ordinal)) = pool_manager
            .server_for_pool_ordinal(*ordinal as u32)
            .await
        {
            let remapped_cmd = CudaCommand::DeviceGet {
                ordinal: server_local_ordinal as i32,
            };
            return forward_cuda_to_server(
                server_conns,
                endpoints,
                server_idx,
                request_id,
                remapped_cmd,
            )
            .await;
        }
        // Fallback: forward as-is to default server
    }

    let msg = Message::CudaCommand {
        request_id,
        command: command.clone(),
    };

    // Determine target server from handle
    let routing_handle = extract_cuda_routing_handle(&command);
    let server_idx = resolve_server_index(pool_manager, routing_handle).await;

    forward_to_server(server_conns, endpoints, server_idx, request_id, &msg, true).await
}

/// Forward a CUDA command to a specific server by index.
async fn forward_cuda_to_server(
    server_conns: &Arc<tokio::sync::RwLock<Vec<Arc<Mutex<Option<ServerConn>>>>>>,
    endpoints: &Arc<tokio::sync::RwLock<Vec<ServerEndpoint>>>,
    server_idx: usize,
    request_id: RequestId,
    command: CudaCommand,
) -> Message {
    let msg = Message::CudaCommand {
        request_id,
        command,
    };
    forward_to_server(server_conns, endpoints, server_idx, request_id, &msg, true).await
}

// ── Vulkan Forwarding ────────────────────────────────────────────────

/// Forward a Vulkan command using the pooled persistent connection.
/// Routes to the correct server based on handle's server_id.
async fn forward_vulkan_command_pooled(
    server_conns: &Arc<tokio::sync::RwLock<Vec<Arc<Mutex<Option<ServerConn>>>>>>,
    endpoints: &Arc<tokio::sync::RwLock<Vec<ServerEndpoint>>>,
    pool_manager: &Arc<GpuPoolManager>,
    request_id: RequestId,
    command: VulkanCommand,
) -> Message {
    // Broadcast commands: CreateInstance goes to all servers, merge responses
    if matches!(command, VulkanCommand::CreateInstance { .. }) {
        return broadcast_vulkan_create_instance(server_conns, endpoints, pool_manager, request_id, &command).await;
    }

    // EnumeratePhysicalDevices: merge from all servers that share the instance
    if let VulkanCommand::EnumeratePhysicalDevices { instance } = &command {
        return broadcast_vulkan_enumerate_physical_devices(
            server_conns,
            endpoints,
            pool_manager,
            request_id,
            *instance,
        )
        .await;
    }

    let msg = Message::VulkanCommand {
        request_id,
        command: command.clone(),
    };

    // Determine target server from handle
    let routing_handle = extract_vulkan_routing_handle(&command);
    let server_idx = resolve_server_index(pool_manager, routing_handle).await;

    forward_to_server(server_conns, endpoints, server_idx, request_id, &msg, false).await
}

// ── Broadcast Vulkan Commands ────────────────────────────────────────

/// Send CreateInstance to all connected servers. Return the first successful handle.
/// In multi-server mode, each server creates its own VkInstance.
/// The client tracks which instance handle belongs to which server via the NetworkHandle.
async fn broadcast_vulkan_create_instance(
    server_conns: &Arc<tokio::sync::RwLock<Vec<Arc<Mutex<Option<ServerConn>>>>>>,
    endpoints: &Arc<tokio::sync::RwLock<Vec<ServerEndpoint>>>,
    pool_manager: &Arc<GpuPoolManager>,
    request_id: RequestId,
    command: &VulkanCommand,
) -> Message {
    let server_indices = pool_manager.all_connected_server_indices().await;

    if server_indices.is_empty() {
        return Message::VulkanResponse {
            request_id,
            response: VulkanResponse::Error {
                code: -3,
                message: "no servers connected".to_string(),
            },
        };
    }

    // Send CreateInstance to all servers, collect handles
    let mut first_handle = None;
    for &idx in &server_indices {
        let msg = Message::VulkanCommand {
            request_id,
            command: command.clone(),
        };
        let resp = forward_to_server(server_conns, endpoints, idx, request_id, &msg, false).await;

        if let Message::VulkanResponse {
            response: VulkanResponse::InstanceCreated { handle },
            ..
        } = &resp
        {
            if first_handle.is_none() {
                first_handle = Some(*handle);
            }
            debug!("Instance created on server {} -> {:?}", idx, handle);
        }
    }

    match first_handle {
        Some(handle) => Message::VulkanResponse {
            request_id,
            response: VulkanResponse::InstanceCreated { handle },
        },
        None => Message::VulkanResponse {
            request_id,
            response: VulkanResponse::Error {
                code: -3,
                message: "failed to create instance on any server".to_string(),
            },
        },
    }
}

/// Enumerate physical devices from all servers, merge into single response.
async fn broadcast_vulkan_enumerate_physical_devices(
    server_conns: &Arc<tokio::sync::RwLock<Vec<Arc<Mutex<Option<ServerConn>>>>>>,
    endpoints: &Arc<tokio::sync::RwLock<Vec<ServerEndpoint>>>,
    pool_manager: &Arc<GpuPoolManager>,
    request_id: RequestId,
    instance: NetworkHandle,
) -> Message {
    let server_indices = pool_manager.all_connected_server_indices().await;
    let mut all_handles = Vec::new();

    for &idx in &server_indices {
        let msg = Message::VulkanCommand {
            request_id,
            command: VulkanCommand::EnumeratePhysicalDevices { instance },
        };
        let resp = forward_to_server(server_conns, endpoints, idx, request_id, &msg, false).await;

        if let Message::VulkanResponse {
            response: VulkanResponse::PhysicalDevices { handles },
            ..
        } = resp
        {
            debug!(
                "Server {} reported {} physical device(s)",
                idx,
                handles.len()
            );
            all_handles.extend(handles);
        }
    }

    Message::VulkanResponse {
        request_id,
        response: VulkanResponse::PhysicalDevices {
            handles: all_handles,
        },
    }
}

// ── Generic Forwarding with Reconnect ────────────────────────────────

/// Forward a message to a specific server, with reconnection on failure.
async fn forward_to_server(
    server_conns: &Arc<tokio::sync::RwLock<Vec<Arc<Mutex<Option<ServerConn>>>>>>,
    endpoints: &Arc<tokio::sync::RwLock<Vec<ServerEndpoint>>>,
    server_idx: usize,
    request_id: RequestId,
    msg: &Message,
    is_cuda: bool,
) -> Message {
    let conns = server_conns.read().await;
    let eps = endpoints.read().await;

    if server_idx >= conns.len() || server_idx >= eps.len() {
        return make_error_response(request_id, is_cuda, "server index out of range");
    }

    let conn_slot = conns[server_idx].clone();
    let endpoint = eps[server_idx].clone();
    drop(conns);
    drop(eps);

    let mut conn_guard = conn_slot.lock().await;

    // Try existing connection
    if let Some(ref mut conn) = *conn_guard {
        match conn.send_and_receive(msg).await {
            Ok(response) => {
                debug!("forwarded command to server {} via pooled connection", server_idx);
                return response;
            }
            Err(e) => {
                warn!(
                    "pooled connection to server {} failed: {} - reconnecting",
                    server_idx, e
                );
                *conn_guard = None;
            }
        }
    }

    // Connection is None or failed - try to reconnect
    match reconnect(&endpoint).await {
        Ok((mut new_conn, _sid)) => {
            match new_conn.send_and_receive(msg).await {
                Ok(response) => {
                    *conn_guard = Some(new_conn);
                    return response;
                }
                Err(e) => {
                    error!("reconnected to server {} but command failed: {}", server_idx, e);
                    *conn_guard = None;
                }
            }
        }
        Err(e) => {
            error!("reconnection to server {} failed: {}", server_idx, e);
        }
    }

    make_error_response(request_id, is_cuda, "failed to communicate with server")
}

/// Create an appropriate error response message.
fn make_error_response(request_id: RequestId, is_cuda: bool, msg: &str) -> Message {
    if is_cuda {
        Message::CudaResponse {
            request_id,
            response: CudaResponse::Error {
                code: 100,
                message: msg.to_string(),
            },
        }
    } else {
        Message::VulkanResponse {
            request_id,
            response: VulkanResponse::Error {
                code: -3,
                message: msg.to_string(),
            },
        }
    }
}

// ── Reconnection Loop ────────────────────────────────────────────────

/// Background task that periodically checks for disconnected servers and reconnects.
async fn reconnection_loop(
    server_conns: Arc<tokio::sync::RwLock<Vec<Arc<Mutex<Option<ServerConn>>>>>>,
    endpoints: Arc<tokio::sync::RwLock<Vec<ServerEndpoint>>>,
    pool_manager: Arc<GpuPoolManager>,
) {
    let mut backoff_secs = vec![1u64; 0]; // Will be initialized on first check

    loop {
        tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;

        let conns = server_conns.read().await;
        let eps = endpoints.read().await;
        let server_count = conns.len();

        // Initialize backoff array if needed
        if backoff_secs.len() < server_count {
            backoff_secs.resize(server_count, 1);
        }

        for i in 0..server_count {
            let conn_slot = conns[i].clone();
            let endpoint = eps[i].clone();

            // Check if connection is alive
            let needs_reconnect = {
                let guard = conn_slot.lock().await;
                guard.is_none()
            };

            if !needs_reconnect {
                // Heartbeat: send Ping to verify connection is still alive
                let mut guard = conn_slot.lock().await;
                if let Some(ref mut conn) = *guard {
                    let ping_result = tokio::time::timeout(
                        tokio::time::Duration::from_secs(10),
                        conn.send_and_receive(&Message::Ping),
                    )
                    .await;

                    match ping_result {
                        Ok(Ok(Message::Pong)) => {
                            // Server is alive
                        }
                        _ => {
                            warn!("server {} failed heartbeat, marking disconnected", i);
                            *guard = None;
                            drop(guard);
                            pool_manager
                                .set_server_status(
                                    i,
                                    ConnectionStatus::Disconnected("heartbeat failed".into()),
                                )
                                .await;
                        }
                    }
                }
                continue;
            }

            drop(conns);
            drop(eps);

            debug!("attempting reconnection to server {} (backoff={}s)", i, backoff_secs[i]);

            match reconnect(&endpoint).await {
                Ok((new_conn, sid)) => {
                    let conn_slot = {
                        let conns = server_conns.read().await;
                        conns[i].clone()
                    };
                    let mut guard = conn_slot.lock().await;
                    *guard = Some(new_conn);

                    pool_manager.set_server_status(i, ConnectionStatus::Connected).await;
                    pool_manager.add_server_mapping(sid, i).await;

                    info!("reconnected to server {} (id={})", i, sid);
                    backoff_secs[i] = 1; // Reset backoff
                }
                Err(e) => {
                    debug!("reconnection to server {} failed: {}", i, e);
                    pool_manager
                        .set_server_status(
                            i,
                            ConnectionStatus::Disconnected(e.to_string()),
                        )
                        .await;
                    // Exponential backoff: double up to 60s max
                    backoff_secs[i] = (backoff_secs[i] * 2).min(60);
                }
            }

            // Re-acquire locks for next iteration
            break; // Break inner loop, outer loop will re-enter
        }
    }
}

/// Wait for a shutdown signal (Ctrl+C or SIGTERM on Unix) for the client daemon.
async fn client_shutdown_signal() {
    let ctrl_c = tokio::signal::ctrl_c();

    #[cfg(unix)]
    {
        let mut sigterm = tokio::signal::unix::signal(
            tokio::signal::unix::SignalKind::terminate(),
        )
        .expect("failed to install SIGTERM handler");

        tokio::select! {
            _ = ctrl_c => { info!("received Ctrl+C, shutting down client daemon"); }
            _ = sigterm.recv() => { info!("received SIGTERM, shutting down client daemon"); }
        }
    }

    #[cfg(not(unix))]
    {
        ctrl_c.await.expect("failed to listen for Ctrl+C");
        info!("received Ctrl+C, shutting down client daemon");
    }
}
