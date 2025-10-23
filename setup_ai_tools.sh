#!/bin/bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

check_port() {
    local port=$1
    local service_name=$2
    
    if command -v nc >/dev/null 2>&1; then
        if nc -z localhost "$port" 2>/dev/null; then
            warn "$service_name port $port is already in use"
            return 0
        fi
    else
        if netstat -tuln 2>/dev/null | grep -q ":$port "; then
            warn "$service_name port $port is already in use"
            return 0
        fi
    fi
    return 1
}

check_service_running() {
    local service_name=$1
    if systemctl is-active --quiet "$service_name" 2>/dev/null; then
        return 0
    fi
    return 1
}

check_user_service_running() {
    local service_name=$1
    if systemctl --user is-active --quiet "$service_name" 2>/dev/null; then
        return 0
    fi
    return 1
}

check_if_already_installed() {
    log "Checking for existing AI tool installations..."
    
    local already_installed=0
    
    if [ -d "$HOME/ai-tools/ComfyUI" ]; then
        info "✓ ComfyUI directory found"
        already_installed=1
    fi
    
    if command -v ollama >/dev/null 2>&1; then
        info "✓ Ollama command found"
        already_installed=1
    fi
    
    if check_service_running "ollama"; then
        info "✓ Ollama service running"
        already_installed=1
    fi
    
    if [ -d "$HOME/ai-tools/openwebui-env" ] && [ -f "$HOME/ai-tools/openwebui-env/bin/open-webui" ]; then
        info "✓ OpenWebUI installation found"
        already_installed=1
    fi
    
    if [ $already_installed -eq 1 ]; then
        warn "Some AI tools appear to already be installed"
        read -p "Do you want to continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log "Exiting as requested"
            exit 0
        fi
    fi
}

check_requirements() {
    log "Checking system requirements..."

    # Ensure PATH includes both possible uv locations
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

    if ! command -v python &> /dev/null; then
        error "Python is not installed. Please run ./setup_dependencies.sh first"
        exit 1
    fi

    if ! command -v pip &> /dev/null; then
        error "pip not found. Please run ./setup_dependencies.sh first"
        exit 1
    fi

    if ! command -v git &> /dev/null; then
        error "Git is not installed. Please run ./setup_dependencies.sh first"
        exit 1
    fi

    if ! command -v curl &> /dev/null; then
        error "curl is not installed. Please run ./setup_dependencies.sh first"
        exit 1
    fi

    if ! command -v node &> /dev/null; then
        error "Node.js is not installed. Please run ./setup_dependencies.sh first"
        exit 1
    fi

    if ! command -v uv &> /dev/null; then
        error "uv is not installed. Please run ./setup_dependencies.sh first"
        exit 1
    fi

    log "System requirements check completed"
}

install_comfyui() {
    log "Installing ComfyUI..."
    
    if check_port 8188 "ComfyUI"; then
        warn "ComfyUI may already be running on port 8188"
    fi
    
    local install_dir="$HOME/ai-tools"
    mkdir -p "$install_dir"
    cd "$install_dir"
    
    if [ -d "ComfyUI" ]; then
        warn "ComfyUI directory already exists, updating..."
        cd ComfyUI
        git pull
        cd ..
    else
        git clone https://github.com/comfyanonymous/ComfyUI.git
    fi
    
    cd ComfyUI
    
    if [ ! -d "venv" ]; then
        info "Creating Python environment for ComfyUI using uv..."
        uv venv venv
    fi
    
    source venv/bin/activate
    
    # Check for GPU (NVIDIA or AMD)
    local has_gpu=0
    local gpu_type="none"
    
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        has_gpu=1
        gpu_type="nvidia"
        info "NVIDIA GPU detected, installing CUDA-enabled PyTorch"
        uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    elif lspci | grep -i amd | grep -i vga >/dev/null 2>&1; then
        has_gpu=1
        gpu_type="amd"
        info "AMD GPU detected, installing ROCm-enabled PyTorch"
        # Install ROCm PyTorch (for AMD GPUs)
        uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
    else
        info "No compatible GPU detected, installing CPU-only PyTorch"
        uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # Install ComfyUI requirements
    if [ -f "requirements.txt" ]; then
        uv pip install -r requirements.txt
    else
        # Install common ComfyUI dependencies if requirements.txt doesn't exist
        uv pip install pillow "numpy<2.0" opencv-python psutil scipy tqdm
    fi
    
    # Fix NumPy compatibility issues with PyTorch
    info "Ensuring NumPy compatibility..."
    uv pip install "numpy<2.0"
    
    deactivate
    
    # Create launcher script with GPU/CPU detection
    if [ $has_gpu -eq 1 ]; then
        info "Creating GPU-enabled launcher script for $gpu_type GPU"
        cat > "$install_dir/launch_comfyui.sh" << 'EOF'
#!/bin/bash
cd "$HOME/ai-tools/ComfyUI"
source venv/bin/activate
python main.py "$@"
EOF
    else
        info "Creating CPU-only launcher script"
        cat > "$install_dir/launch_comfyui.sh" << 'EOF'
#!/bin/bash
cd "$HOME/ai-tools/ComfyUI"
source venv/bin/activate
python main.py --cpu "$@"
EOF
    fi
    
    chmod +x "$install_dir/launch_comfyui.sh"
    
    log "ComfyUI installed successfully"
    info "Launch ComfyUI with: $install_dir/launch_comfyui.sh"
}

pull_ollama_models(){
    log "Pulling Ollama models..."
    
    # Check if ollama is running
    if ! check_service_running "ollama"; then
        warn "Ollama service is not running. Starting it first..."
        sudo systemctl start ollama
        sleep 5
        
        if ! check_service_running "ollama"; then
            error "Cannot start Ollama service. Skipping model downloads."
            return 1
        fi
    fi
    
    # Wait for Ollama API to be ready
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s http://localhost:11434/api/version >/dev/null 2>&1; then
            info "Ollama API is ready"
            break
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            error "Ollama API not responding after ${max_attempts} attempts"
            return 1
        fi
        
        info "Waiting for Ollama API... (attempt $attempt/$max_attempts)"
        sleep 2
        ((attempt++))
    done
    
    # Array of models to download
    local models=(
        "llama3.2:3b"
        "yuiseki/devstral-small-2507:24b"
        "hf.co/bartowski/Qwen2.5-Coder-14B-Instruct-abliterated-GGUF:Q4_K_S"
        "hf.co/mlabonne/gemma-3-27b-it-abliterated-GGUF:Q4_K_M"
    )
    
    info "Downloading ${#models[@]} models. This may take a while..."
    
    for model in "${models[@]}"; do
        log "Pulling model: $model"
        
        # Check if model already exists
        if ollama list | grep -q "^${model%:*}"; then
            info "Model ${model%:*} already exists, skipping..."
            continue
        fi
        
        # Pull the model with progress
        if ollama pull "$model"; then
            info "✓ Successfully downloaded: $model"
        else
            warn "✗ Failed to download: $model"
            info "You can manually download it later with: ollama pull $model"
        fi
    done
    
    log "Model downloads completed"
    info "View downloaded models with: ollama list"
}

install_ollama() {
    log "Installing Ollama..."
    
    if check_port 11434 "Ollama"; then
        warn "Ollama may already be running on port 11434"
        if check_service_running "ollama"; then
            info "Ollama service is already running"
            return 0
        fi
    fi
    
    if command -v ollama >/dev/null 2>&1; then
        info "Ollama is already installed, checking service..."
    else
        info "Downloading and installing Ollama..."
        curl -fsSL https://ollama.ai/install.sh | sh
    fi
    
    # Start Ollama service
    sudo systemctl enable ollama
    sudo systemctl start ollama
    
    # Wait for service to be ready
    sleep 5
    
    # Check if Ollama is running
    if systemctl is-active --quiet ollama; then
        log "Ollama installed and started successfully"
        info "Ollama is available at http://localhost:11434"
        info "Download models with: ollama pull <model_name>"
        info "Example: ollama pull llama2"
    else
        error "Ollama service failed to start"
        return 1
    fi
}

install_openwebui() {
    log "Installing Open WebUI..."
    
    if check_port 8080 "OpenWebUI"; then
        warn "OpenWebUI may already be running on port 8080"
    fi
    
    local install_dir="$HOME/ai-tools"
    mkdir -p "$install_dir"
    cd "$install_dir"
    
    # Remove existing installation if it exists and is problematic
    if [ -d "open-webui" ]; then
        warn "Removing existing Open WebUI directory..."
        rm -rf "open-webui"
    fi
    
    # Install OpenWebUI using uv with Python 3.11+
    if [ ! -d "openwebui-env" ]; then
        info "Creating Python 3.11+ environment for OpenWebUI using uv..."
        uv venv openwebui-env --python 3.11
    fi
    
    # Activate the environment and install OpenWebUI
    source openwebui-env/bin/activate
    uv pip install open-webui
    deactivate
    
    # Create launcher script
    cat > "$install_dir/launch_openwebui.sh" << 'EOF'
#!/bin/bash
cd "$HOME/ai-tools"
source openwebui-env/bin/activate
export OLLAMA_BASE_URL=http://localhost:11434
open-webui serve --host 0.0.0.0 --port 8080
EOF
    
    chmod +x "$install_dir/launch_openwebui.sh"
    
    log "Open WebUI installed successfully"
    info "Launch Open WebUI with: $install_dir/launch_openwebui.sh"
    info "Access Open WebUI at http://localhost:8080"
}

create_desktop_shortcuts() {
    log "Creating desktop shortcuts..."
    
    local desktop_dir="$HOME/Desktop"
    mkdir -p "$desktop_dir"
    
    # ComfyUI shortcut
    cat > "$desktop_dir/ComfyUI.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=ComfyUI
Comment=ComfyUI Launcher
Exec=$HOME/ai-tools/launch_comfyui.sh
Icon=applications-graphics
Terminal=true
Categories=Graphics;
EOF
    
    # Open WebUI shortcut
    cat > "$desktop_dir/OpenWebUI.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Open WebUI
Comment=Open WebUI for Ollama
Exec=$HOME/ai-tools/launch_openwebui.sh
Icon=applications-internet
Terminal=true
Categories=Network;
EOF
    
    chmod +x "$desktop_dir"/*.desktop
    
    log "Desktop shortcuts created"
}

create_systemd_services() {
    log "Creating systemd user services..."
    
    local service_dir="$HOME/.config/systemd/user"
    mkdir -p "$service_dir"
    
    # Open WebUI service
    cat > "$service_dir/openwebui.service" << EOF
[Unit]
Description=Open WebUI
After=ollama.service
Wants=ollama.service

[Service]
Type=simple
WorkingDirectory=$HOME/ai-tools
ExecStart=$HOME/ai-tools/openwebui-env/bin/open-webui serve --host 0.0.0.0 --port 8080
Environment=OLLAMA_BASE_URL=http://localhost:11434
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
EOF
    
    # Reload systemd and enable services
    systemctl --user daemon-reload
    systemctl --user enable openwebui.service
    
    log "Systemd services created"
    info "Start Open WebUI service with: systemctl --user start openwebui"
    info "Enable auto-start with: systemctl --user enable openwebui"
}

print_usage_info() {
    log "Installation completed successfully!"
    echo ""
    info "=== Usage Information ==="
    info "ComfyUI:"
    info "  - Launch: $HOME/ai-tools/launch_comfyui.sh"
    info "  - Access: http://localhost:8188"
    echo ""
    info "Ollama:"
    info "  - Service: sudo systemctl status ollama"
    info "  - List models: ollama list"
    info "  - Download more models: ollama pull <model_name>"
    info "  - API: http://localhost:11434"
    info "  - Pre-installed models: llama3.2:3b, devstral-small, qwen2.5-coder, gemma-3"
    echo ""
    info "Open WebUI:"
    info "  - Launch: $HOME/ai-tools/launch_openwebui.sh"
    info "  - Service: systemctl --user start openwebui"
    info "  - Access: http://localhost:8080"
    echo ""
    info "All tools are installed in: $HOME/ai-tools/"
    echo ""
}

main() {
    log "Starting AI Tools Setup Script for CachyOS/Arch Linux"
    info "NOTE: Run ./setup_dependencies.sh first if you haven't already"

    check_if_already_installed
    check_requirements
    install_comfyui
    install_ollama
    pull_ollama_models
    install_openwebui
    create_desktop_shortcuts
    create_systemd_services
    print_usage_info

    log "Setup completed successfully!"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi