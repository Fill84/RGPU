# Build Linux .so interpose libraries for Docker container integration.
#
# Usage:
#   docker build -f packaging/linux/Dockerfile.builder -t rgpu-builder .
#   docker create --name rgpu-extract rgpu-builder
#   docker cp rgpu-extract:/output/. ./packaging/linux/container-libs/
#   docker rm rgpu-extract
#
# Or use: packaging/linux/build-container-libs.sh

FROM rust:1.83-bookworm AS builder
WORKDIR /src
COPY . .
RUN cargo build --release \
    -p rgpu-cuda-interpose \
    -p rgpu-nvml-interpose \
    -p rgpu-nvenc-interpose \
    -p rgpu-nvdec-interpose \
    -p rgpu-vk-icd

# Rename to standard NVIDIA library names and create symlinks
RUN mkdir /output && \
    cp target/release/librgpu_cuda_interpose.so /output/libcuda.so.1 && \
    cp target/release/librgpu_nvml_interpose.so /output/libnvidia-ml.so.1 && \
    cp target/release/librgpu_nvenc_interpose.so /output/libnvidia-encode.so.1 && \
    cp target/release/librgpu_nvdec_interpose.so /output/libnvcuvid.so.1 && \
    cp target/release/librgpu_vk_icd.so /output/librgpu_vk_icd.so && \
    cd /output && \
    ln -s libcuda.so.1 libcuda.so && \
    ln -s libnvidia-ml.so.1 libnvidia-ml.so && \
    ln -s libnvidia-encode.so.1 libnvidia-encode.so && \
    ln -s libnvcuvid.so.1 libnvcuvid.so

CMD ["ls", "-la", "/output/"]
