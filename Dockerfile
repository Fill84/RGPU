# =============================================================================
# RGPU Docker Integration Test — Multi-stage Build
# =============================================================================
# Stage 1: builder   — compiles Rust workspace + C test programs
# Stage 2: server    — runs rgpu server with real GPU
# Stage 3: test-runner — runs client daemon + test programs (no GPU)
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.6.0-devel-ubuntu24.04 AS builder

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    g++ \
    pkg-config \
    cmake \
    libvulkan-dev \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy workspace
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/

# Build interpose libraries
RUN cargo build --release \
    -p rgpu-cuda-interpose \
    -p rgpu-vk-icd \
    -p rgpu-nvml-interpose \
    -p rgpu-nvenc-interpose \
    -p rgpu-nvdec-interpose

# Build CLI without UI feature (avoids windowing/eframe deps on headless Linux)
RUN cargo build --release -p rgpu-cli --no-default-features

# Compile C test programs
COPY tests/docker/ /build/tests/docker/

# test_cuda — uses dlopen, no link-time dependency on CUDA
RUN gcc -O2 -o /build/test_cuda /build/tests/docker/test_cuda.c -ldl

# test_vulkan — links against Vulkan loader
RUN gcc -O2 -o /build/test_vulkan /build/tests/docker/test_vulkan.c -lvulkan

# test_nvml — uses dlopen
RUN gcc -O2 -o /build/test_nvml /build/tests/docker/test_nvml.c -ldl

# test_nvenc — uses dlopen
RUN gcc -O2 -o /build/test_nvenc /build/tests/docker/test_nvenc.c -ldl

# test_nvdec — uses dlopen
RUN gcc -O2 -o /build/test_nvdec /build/tests/docker/test_nvdec.c -ldl


# ---------------------------------------------------------------------------
# Stage 2: Server (needs real GPU + CUDA/Vulkan runtime)
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.6.0-runtime-ubuntu24.04 AS server

RUN apt-get update && apt-get install -y --no-install-recommends \
    libvulkan1 \
    iproute2 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy the rgpu binary
COPY --from=builder /build/target/release/rgpu /usr/local/bin/rgpu

EXPOSE 9876

# Entrypoint: generate config from env vars, then start server
# RGPU_TOKEN env var must be set
ENTRYPOINT ["sh", "-c", "\
    mkdir -p /etc/rgpu && \
    printf '[server]\\nbind = \"0.0.0.0\"\\nport = 9876\\n\\n[security]\\n[[security.tokens]]\\ntoken = \"%s\"\\nname = \"docker-test\"\\n' \"$RGPU_TOKEN\" > /etc/rgpu/rgpu.toml && \
    exec rgpu server --config /etc/rgpu/rgpu.toml \
"]


# ---------------------------------------------------------------------------
# Stage 3: Test Runner (no GPU — connects to server over network)
# ---------------------------------------------------------------------------
FROM ubuntu:24.04 AS test-runner

RUN apt-get update && apt-get install -y --no-install-recommends \
    libvulkan1 \
    ca-certificates \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy rgpu binary (for client daemon)
COPY --from=builder /build/target/release/rgpu /usr/local/bin/rgpu

# Copy interpose libraries
RUN mkdir -p /usr/lib/rgpu
COPY --from=builder /build/target/release/librgpu_cuda_interpose.so /usr/lib/rgpu/
COPY --from=builder /build/target/release/librgpu_vk_icd.so /usr/lib/rgpu/
COPY --from=builder /build/target/release/librgpu_nvml_interpose.so /usr/lib/rgpu/
COPY --from=builder /build/target/release/librgpu_nvenc_interpose.so /usr/lib/rgpu/
COPY --from=builder /build/target/release/librgpu_nvdec_interpose.so /usr/lib/rgpu/

# Create symlinks with standard NVIDIA library names so apps that dlopen
# by system name (e.g. FFmpeg loading libcuda.so.1) find our interpose libs
RUN ln -sf /usr/lib/rgpu/librgpu_cuda_interpose.so /usr/lib/rgpu/libcuda.so.1 && \
    ln -sf /usr/lib/rgpu/librgpu_cuda_interpose.so /usr/lib/rgpu/libcuda.so && \
    ln -sf /usr/lib/rgpu/librgpu_nvenc_interpose.so /usr/lib/rgpu/libnvidia-encode.so.1 && \
    ln -sf /usr/lib/rgpu/librgpu_nvenc_interpose.so /usr/lib/rgpu/libnvidia-encode.so && \
    ln -sf /usr/lib/rgpu/librgpu_nvdec_interpose.so /usr/lib/rgpu/libnvcuvid.so.1 && \
    ln -sf /usr/lib/rgpu/librgpu_nvdec_interpose.so /usr/lib/rgpu/libnvcuvid.so && \
    ln -sf /usr/lib/rgpu/librgpu_nvml_interpose.so /usr/lib/rgpu/libnvidia-ml.so.1 && \
    ln -sf /usr/lib/rgpu/librgpu_nvml_interpose.so /usr/lib/rgpu/libnvidia-ml.so

# Make interpose libraries discoverable via ldconfig
RUN echo "/usr/lib/rgpu" > /etc/ld.so.conf.d/rgpu.conf && ldconfig

# Copy Vulkan ICD manifest
RUN mkdir -p /usr/share/vulkan/icd.d
COPY tests/docker/rgpu_icd.json /usr/share/vulkan/icd.d/rgpu_icd.json

# Copy test binaries
COPY --from=builder /build/test_cuda /usr/local/bin/
COPY --from=builder /build/test_vulkan /usr/local/bin/
COPY --from=builder /build/test_nvml /usr/local/bin/
COPY --from=builder /build/test_nvenc /usr/local/bin/
COPY --from=builder /build/test_nvdec /usr/local/bin/

# Copy test scripts
COPY tests/docker/run_tests.sh /usr/local/bin/run_tests.sh
COPY tests/docker/test_ffmpeg_hwaccel.sh /usr/local/bin/test_ffmpeg_hwaccel.sh
RUN chmod +x /usr/local/bin/run_tests.sh /usr/local/bin/test_ffmpeg_hwaccel.sh

# RGPU_SERVER and RGPU_TOKEN must be set via environment
ENTRYPOINT ["/usr/local/bin/run_tests.sh"]
