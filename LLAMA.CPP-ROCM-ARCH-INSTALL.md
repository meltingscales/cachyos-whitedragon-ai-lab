# llama.cpp ROCm Installation Guide for CachyOS

Complete guide for building and installing llama.cpp with ROCm (AMD GPU) support on CachyOS.

## Prerequisites

First, update your system and install required dependencies:

```bash
# Update system
sudo pacman -Syu

# Install base development tools
sudo pacman -S base-devel cmake git

# Install ROCm packages (CachyOS has ROCm in repos)
sudo pacman -S rocm-hip-sdk rocm-opencl-sdk rocblas hipblas

# Optional but recommended: additional ROCm libraries
sudo pacman -S rocm-smi-lib hsa-rocr
```

## Clone llama.cpp

```bash
# Clone the repository
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
```

## Build with CMake

### Option 1: Standard Build with ROCm

```bash
# Create build directory
mkdir build
cd build

# Configure with CMake (enable ROCm/HIP support)
cmake .. \
  -DGGML_HIPBLAS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DAMDGPU_TARGETS="gfx1030;gfx1100;gfx1101;gfx1102" \
  -DCMAKE_INSTALL_PREFIX=/usr/local

# Build (use all CPU cores)
cmake --build . --config Release -j$(nproc)
```

**Note:** Adjust `AMDGPU_TARGETS` to match your GPU architecture:
- **RX 6000 series (RDNA 2):** `gfx1030`, `gfx1031`, `gfx1032`
- **RX 7000 series (RDNA 3):** `gfx1100`, `gfx1101`, `gfx1102`
- **Radeon VII / MI50:** `gfx906`
- **MI100:** `gfx908`

To check your GPU architecture:
```bash
rocminfo | grep gfx
```

### Option 2: Build with Additional Optimizations

```bash
cmake .. \
  -DGGML_HIPBLAS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON \
  -DLLAMA_CURL=ON \
  -DAMDGPU_TARGETS="gfx1030;gfx1100;gfx1101" \
  -DCMAKE_INSTALL_PREFIX=/usr/local

cmake --build . --config Release -j$(nproc)
```

## Installation

### Method 1: System-wide Installation (Recommended)

```bash
# Install to /usr/local (requires sudo)
sudo cmake --install . --prefix /usr/local

# IMPORTANT: Configure library path for CachyOS
# Add /usr/local/lib to the system library path
echo "/usr/local/lib" | sudo tee /etc/ld.so.conf.d/local.conf

# Update library cache
sudo ldconfig

# Verify library is found
ldconfig -p | grep libllama

# Verify installation
which llama-cli
llama-cli --version
```

This installs:
- Binaries to `/usr/local/bin/` (automatically in PATH)
- Libraries to `/usr/local/lib/`
- Headers to `/usr/local/include/`

**Note:** CachyOS may not have `/usr/local/lib` in the default library search path, so the `ld.so.conf.d` configuration step is essential.

### Method 2: User Installation (No sudo required)

```bash
# Install to ~/.local
cmake --install . --prefix ~/.local

# Add to PATH if not already there
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Verify
which llama-cli
```

### Method 3: Manual Binary Installation

If you prefer manual control:

```bash
# From the build directory, copy binaries
sudo cp bin/llama-cli /usr/local/bin/
sudo cp bin/llama-server /usr/local/bin/
sudo cp bin/llama-quantize /usr/local/bin/
sudo cp bin/llama-embedding /usr/local/bin/
sudo cp bin/llama-bench /usr/local/bin/

# Make executable (usually already set)
sudo chmod +x /usr/local/bin/llama-*

# Copy shared library (it's in bin/ directory on CachyOS build)
sudo cp bin/libllama.so /usr/local/lib/

# IMPORTANT: Configure library path for CachyOS
echo "/usr/local/lib" | sudo tee /etc/ld.so.conf.d/local.conf

# Update library cache
sudo ldconfig

# Verify library is found
ldconfig -p | grep libllama
```

## Verify ROCm is Working

```bash
# Test that ROCm is detected
llama-cli --version

# Run a benchmark to confirm GPU usage
llama-bench -m /path/to/model.gguf

# Monitor GPU during inference
watch -n 1 rocm-smi
```

## Environment Variables (Optional)

Add these to `~/.bashrc` for better ROCm performance:

```bash
# ROCm environment variables
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # Adjust for your GPU
export GPU_MAX_HEAP_SIZE=100
export GPU_MAX_ALLOC_PERCENT=100
export HSA_ENABLE_SDMA=0  # Disable SDMA if issues occur
```

## Downloading Models

```bash
# Example: Download a model using huggingface-cli
pip install huggingface-hub --break-system-packages

# Download a GGUF model
huggingface-cli download TheBloke/Llama-2-7B-GGUF llama-2-7b.Q4_K_M.gguf --local-dir ./models
```

## Running Inference

```bash
# Basic CPU inference
llama-cli -m models/llama-2-7b.Q4_K_M.gguf -p "Hello, how are you?"

# GPU-accelerated inference (offload all layers to GPU)
llama-cli -m models/llama-2-7b.Q4_K_M.gguf -ngl 99 -p "Hello, how are you?"

# Start server mode
llama-server -m models/llama-2-7b.Q4_K_M.gguf -ngl 99 --host 0.0.0.0 --port 8080
```

## Troubleshooting

### libllama.so Not Found Error

If you get `error while loading shared libraries: libllama.so: cannot open shared object file`:

```bash
# Check if library exists
ls -la /usr/local/lib/libllama.so

# Check if /usr/local/lib is in library path
cat /etc/ld.so.conf.d/*.conf | grep /usr/local/lib

# If not found, add it (CachyOS specific fix)
echo "/usr/local/lib" | sudo tee /etc/ld.so.conf.d/local.conf

# Rebuild library cache
sudo ldconfig

# Verify library is now found
ldconfig -p | grep libllama

# Test
llama-cli --version
```

### ROCm Not Detected

```bash
# Check ROCm installation
rocminfo

# Verify user is in render/video groups
groups
# If not in these groups:
sudo usermod -aG render,video $USER
# Then log out and back in
```

### Build Errors

```bash
# Clean build and retry
cd build
rm -rf *
cmake .. -DGGML_HIPBLAS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

### Performance Issues

```bash
# Monitor GPU utilization
rocm-smi -d 0

# Check which device is being used
HIP_VISIBLE_DEVICES=0 llama-cli -m model.gguf -ngl 99 -p "test"
```

## Updating llama.cpp

```bash
cd llama.cpp
git pull
cd build
cmake --build . --config Release -j$(nproc)
sudo cmake --install .
```

## Uninstallation

```bash
# If installed with CMake
sudo rm /usr/local/bin/llama-*
sudo rm /usr/local/lib/libllama.*
sudo rm -rf /usr/local/include/llama*

# Or use CMake uninstall (if available)
cd llama.cpp/build
sudo cmake --build . --target uninstall
```

## Additional Resources

- **llama.cpp GitHub:** https://github.com/ggerganov/llama.cpp
- **ROCm Documentation:** https://rocm.docs.amd.com/
- **GGUF Models:** https://huggingface.co/models?library=gguf

---

**Author's Note:** This guide is specifically tailored for CachyOS with ROCm. Adjust `AMDGPU_TARGETS` based on your specific AMD GPU model for optimal performance.
