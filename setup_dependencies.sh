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

check_if_already_run() {
    log "Checking for existing installations..."

    local already_installed=0

    if package_installed "base-devel" && command -v python >/dev/null 2>&1 && command -v node >/dev/null 2>&1 && command -v uv >/dev/null 2>&1; then
        warn "Core dependencies appear to already be installed"
        info "✓ base-devel found"
        info "✓ Python found ($(python --version))"
        info "✓ Node.js found ($(node --version))"
        info "✓ uv found ($(uv --version))"
        already_installed=1
    fi

    if [ $already_installed -eq 1 ]; then
        read -p "Do you want to continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log "Exiting as requested"
            exit 0
        fi
    fi
}

package_installed() {
    pacman -Q "$1" &> /dev/null
}

update_system() {
    log "Updating system packages..."
    sudo pacman -Syu --noconfirm
    log "System packages updated"
}

install_basic_dependencies() {
    log "Installing basic system dependencies..."

    local packages=(
        base-devel
        ca-certificates
        gnupg
        curl
        wget
        git
        unzip
        zip
        openbsd-netcat
    )

    local to_install=()
    for pkg in "${packages[@]}"; do
        if ! package_installed "$pkg"; then
            to_install+=("$pkg")
        else
            info "$pkg already installed"
        fi
    done

    if [ ${#to_install[@]} -gt 0 ]; then
        info "Installing: ${to_install[*]}"
        sudo pacman -S --noconfirm --needed "${to_install[@]}"
        log "Basic dependencies installed"
    else
        info "All basic dependencies already installed"
    fi
}

install_python_stack() {
    log "Installing Python development stack..."

    # Install system Python and development tools
    sudo pacman -S --noconfirm --needed \
        python \
        python-pip \
        python-setuptools \
        python-wheel

    log "Python stack installed"
}

install_uv() {
    log "Installing uv (modern Python package manager)..."
    
    # Check if uv is already available
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    if command -v uv >/dev/null 2>&1; then
        info "uv is already installed"
        info "uv version: $(uv --version)"
        return 0
    fi
    
    # Create installation directory
    mkdir -p "$HOME/.local/bin"
    
    # Install uv using the official installer
    info "Downloading and installing uv..."
    export UV_INSTALL_DIR="$HOME/.local/bin"
    
    # Install with explicit directory
    if curl -LsSf https://astral.sh/uv/install.sh | sh; then
        info "uv installer completed"
    else
        error "uv installer failed"
        return 1
    fi
    
    # Handle snap environment - find and move uv if needed
    if [ ! -f "$HOME/.local/bin/uv" ]; then
        info "Looking for uv in alternative locations..."
        
        # Check common snap locations
        for snap_dir in "$HOME"/snap/*/.*; do
            if [ -f "$snap_dir/bin/uv" ]; then
                info "Found uv in snap directory: $snap_dir/bin/uv"
                cp "$snap_dir/bin/uv" "$HOME/.local/bin/uv"
                [ -f "$snap_dir/bin/uvx" ] && cp "$snap_dir/bin/uvx" "$HOME/.local/bin/uvx"
                chmod +x "$HOME/.local/bin/uv" "$HOME/.local/bin/uvx" 2>/dev/null
                break
            fi
        done
        
        # Check cargo location
        if [ ! -f "$HOME/.local/bin/uv" ] && [ -f "$HOME/.cargo/bin/uv" ]; then
            info "Found uv in cargo directory"
            cp "$HOME/.cargo/bin/uv" "$HOME/.local/bin/uv"
            [ -f "$HOME/.cargo/bin/uvx" ] && cp "$HOME/.cargo/bin/uvx" "$HOME/.local/bin/uvx"
            chmod +x "$HOME/.local/bin/uv" "$HOME/.local/bin/uvx" 2>/dev/null
        fi
    fi
    
    # Verify installation
    export PATH="$HOME/.local/bin:$PATH"
    if command -v uv >/dev/null 2>&1; then
        log "uv installed successfully"
        info "uv version: $(uv --version)"
    else
        error "uv installation failed - binary not accessible"
        return 1
    fi
}

install_nodejs() {
    log "Installing Node.js and npm..."

    # Install Node.js and npm from official Arch repos
    sudo pacman -S --noconfirm --needed nodejs npm

    # Update npm to latest
    sudo npm install -g npm@latest

    log "Node.js and npm installed"
    info "Node.js version: $(node --version)"
    info "npm version: $(npm --version)"
}

install_multimedia_libs() {
    log "Installing multimedia and graphics libraries..."

    sudo pacman -S --noconfirm --needed \
        ffmpeg \
        gstreamer \
        gst-plugins-base \
        libjpeg-turbo \
        libpng \
        libtiff \
        libwebp \
        opencv \
        mesa \
        glib2

    log "Multimedia libraries installed"
}

install_ai_ml_dependencies() {
    log "Installing AI/ML system dependencies..."

    sudo pacman -S --noconfirm --needed \
        blas \
        lapack \
        gcc-fortran \
        hdf5 \
        openssl \
        xz \
        bzip2 \
        readline \
        sqlite \
        llvm \
        ncurses \
        tk

    log "AI/ML dependencies installed"
}

install_rocm_drivers() {
    log "Installing ROCm drivers for AMD GPU..."

    # Check if AMD GPU is present
    if ! lspci | grep -i amd | grep -i vga >/dev/null 2>&1; then
        info "No AMD GPU detected, skipping ROCm installation"
        return 0
    fi

    # Check if ROCm is already installed
    if [ -f /opt/rocm/bin/rocm-smi ]; then
        info "ROCm appears to already be installed"
        /opt/rocm/bin/rocm-smi --version 2>/dev/null || true
        return 0
    fi

    info "AMD GPU detected, installing ROCm drivers for CachyOS/Arch Linux..."

    # Install ROCm from CachyOS/Arch repositories
    info "Installing ROCm packages (this will take several minutes)..."
    sudo pacman -S --noconfirm --needed \
        rocm-core \
        rocm-hip-sdk \
        rocm-opencl-sdk \
        rocm-smi-lib \
        rocminfo \
        hip-runtime-amd \
        rocblas \
        hipblas \
        rocrand \
        hiprand

    # Add user to render and video groups
    info "Adding user to render and video groups..."
    sudo usermod -a -G render,video "$USER"

    # Add ROCm to PATH
    if [ -f "$HOME/.bashrc" ]; then
        if ! grep -q "/opt/rocm/bin" "$HOME/.bashrc"; then
            echo 'export PATH="/opt/rocm/bin:$PATH"' >> "$HOME/.bashrc"
            info "Added /opt/rocm/bin to PATH in .bashrc"
        fi
    fi

    if [ -f "$HOME/.zshrc" ]; then
        if ! grep -q "/opt/rocm/bin" "$HOME/.zshrc"; then
            echo 'export PATH="/opt/rocm/bin:$PATH"' >> "$HOME/.zshrc"
            info "Added /opt/rocm/bin to PATH in .zshrc"
        fi
    fi

    log "ROCm installation completed"
    warn "You MUST reboot for ROCm drivers and group changes to take effect"
    info "After reboot, verify with: /opt/rocm/bin/rocm-smi"
}

install_optional_tools() {
    log "Installing optional development tools..."

    sudo pacman -S --noconfirm --needed \
        htop \
        tree \
        jq \
        vim \
        nano \
        tmux \
        screen \
        rsync \
        openssh

    log "Optional tools installed"
}

configure_environment() {
    log "Configuring environment..."
    
    # Add local bin to PATH if not already there (priority location for uv)
    for rc_file in "$HOME/.bashrc" "$HOME/.zshrc"; do
        if [ -f "$rc_file" ]; then
            if ! grep -q "$HOME/.local/bin" "$rc_file"; then
                echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$rc_file"
                info "Added ~/.local/bin to PATH in $(basename $rc_file)"
            fi

            # Add cargo bin as fallback for uv
            if ! grep -q "$HOME/.cargo/bin" "$rc_file"; then
                echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> "$rc_file"
                info "Added ~/.cargo/bin to PATH in $(basename $rc_file)"
            fi
        fi
    done
    
    # Create ai-tools directory
    mkdir -p "$HOME/ai-tools"
    
    log "Environment configured"
}

verify_installation() {
    log "Verifying installation..."

    info "System Information:"
    info "  OS: $(grep PRETTY_NAME /etc/os-release | cut -d'"' -f2)"
    info "  Kernel: $(uname -r)"
    info "  Python: $(python --version)"
    info "  pip: $(python -m pip --version | cut -d' ' -f2)"
    info "  uv: $(uv --version 2>/dev/null || echo 'Not available')"
    info "  Node.js: $(node --version)"
    info "  npm: $(npm --version)"
    info "  Git: $(git --version)"
    info "  FFmpeg: $(ffmpeg -version | head -1)"
    if [ -f /opt/rocm/bin/rocm-smi ]; then
        info "  ROCm: $(/opt/rocm/bin/rocm-smi --version 2>/dev/null | head -1 || echo 'Installed but needs reboot')"
    fi

    log "Installation verification completed"
}

main() {
    log "Starting Dependencies Setup for CachyOS/Arch Linux"

    check_if_already_run
    update_system
    install_basic_dependencies
    install_python_stack
    install_uv
    install_nodejs
    install_multimedia_libs
    install_ai_ml_dependencies
    install_rocm_drivers
    install_optional_tools
    configure_environment
    verify_installation

    log "Dependencies setup completed successfully!"
    info "Please run 'source ~/.bashrc' (or ~/.zshrc) or restart your terminal to apply PATH changes"

    # Check if reboot is recommended
    if lspci | grep -i amd | grep -i vga >/dev/null 2>&1 && command -v rocm-smi >/dev/null 2>&1; then
        warn "ROCm drivers were installed. Please reboot to ensure AMD GPU is fully accessible."
    fi
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi