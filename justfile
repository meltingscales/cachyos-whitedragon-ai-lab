# CachyOS WhiteDragon AI Lab - Service Management
# Install just: sudo pacman -S just

# Show available commands
default:
    @just --list

# Start all AI services
start-all:
    @echo "Starting all AI services..."
    @just start-ollama
    @just start-openwebui
    @just start-comfyui
    @just start-sdwebui

# Stop all AI services
stop-all:
    @echo "Stopping all AI services..."
    @just stop-ollama
    @just stop-openwebui
    @just stop-comfyui
    @just stop-sdwebui

# Show status of all services
status:
    @echo "=== Ollama Service Status ==="
    @sudo systemctl status ollama --no-pager || echo "Ollama service not running"
    @echo ""
    @echo "=== OpenWebUI Service Status ==="
    @systemctl --user status openwebui --no-pager || echo "OpenWebUI service not running"
    @echo ""
    @echo "=== Port Status ==="
    @echo "ComfyUI (8188):"
    @ss -tlnp | grep :8188 || echo "  Not running"
    @echo "Ollama (11434):"
    @ss -tlnp | grep :11434 || echo "  Not running"
    @echo "OpenWebUI (8080):"
    @ss -tlnp | grep :8080 || echo "  Not running"
    @echo "SD WebUI (7860):"
    @ss -tlnp | grep :7860 || echo "  Not running"

# ComfyUI commands
start-comfyui:
    @echo "Starting ComfyUI..."
    @if [ -f ~/ai-tools/launch_comfyui.sh ]; then \
        ~/ai-tools/launch_comfyui.sh & \
        echo "ComfyUI started at http://localhost:8188"; \
    else \
        echo "Error: ~/ai-tools/launch_comfyui.sh not found. Run ./setup_ai_tools.sh first"; \
    fi

stop-comfyui:
    @echo "Stopping ComfyUI..."
    @pkill -f "python.*main.py" || echo "ComfyUI not running"

# Stable Diffusion WebUI (Automatic1111) commands
start-sdwebui:
    @echo "Starting Stable Diffusion WebUI..."
    @if [ -f ~/stable-diffusion-webui/webui.sh ]; then \
        cd ~/stable-diffusion-webui && ./webui.sh --listen & \
        echo "SD WebUI starting at http://localhost:7860"; \
    elif [ -f ~/ai-tools/stable-diffusion-webui/webui.sh ]; then \
        cd ~/ai-tools/stable-diffusion-webui && ./webui.sh --listen & \
        echo "SD WebUI starting at http://localhost:7860"; \
    else \
        echo "Error: Stable Diffusion WebUI not found in ~/stable-diffusion-webui or ~/ai-tools/stable-diffusion-webui"; \
        echo "Install with: git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git ~/stable-diffusion-webui"; \
    fi

stop-sdwebui:
    @echo "Stopping Stable Diffusion WebUI..."
    @pkill -f "webui.py\|launch.py" || echo "SD WebUI not running"

# Ollama commands
start-ollama:
    @echo "Starting Ollama service..."
    @sudo systemctl start ollama
    @echo "Ollama started at http://localhost:11434"

stop-ollama:
    @echo "Stopping Ollama service..."
    @sudo systemctl stop ollama

restart-ollama:
    @sudo systemctl restart ollama
    @echo "Ollama restarted"

enable-ollama:
    @sudo systemctl enable ollama
    @echo "Ollama enabled to start on boot"

disable-ollama:
    @sudo systemctl disable ollama
    @echo "Ollama disabled from starting on boot"

# Ollama model management
ollama-list:
    @ollama list

ollama-pull model:
    @ollama pull {{model}}

ollama-logs:
    @journalctl -u ollama -f

# OpenWebUI commands
start-openwebui:
    @echo "Starting OpenWebUI..."
    @if [ -f ~/ai-tools/launch_openwebui.sh ]; then \
        ~/ai-tools/launch_openwebui.sh & \
        echo "OpenWebUI started at http://localhost:8080"; \
    else \
        echo "Error: ~/ai-tools/launch_openwebui.sh not found. Run ./setup_ai_tools.sh first"; \
    fi

start-openwebui-service:
    @systemctl --user start openwebui
    @echo "OpenWebUI service started at http://localhost:8080"

stop-openwebui:
    @echo "Stopping OpenWebUI..."
    @systemctl --user stop openwebui || pkill -f "open-webui" || echo "OpenWebUI not running"

restart-openwebui:
    @systemctl --user restart openwebui || echo "OpenWebUI service not configured. Use 'just start-openwebui' instead"

enable-openwebui:
    @systemctl --user enable openwebui
    @echo "OpenWebUI enabled to start on boot"

disable-openwebui:
    @systemctl --user disable openwebui
    @echo "OpenWebUI disabled from starting on boot"

openwebui-logs:
    @journalctl --user -u openwebui -f

# Open services in browser
open-comfyui:
    @xdg-open http://localhost:8188 2>/dev/null || echo "Open http://localhost:8188 in your browser"

open-ollama:
    @xdg-open http://localhost:11434 2>/dev/null || echo "Open http://localhost:11434 in your browser"

open-openwebui:
    @xdg-open http://localhost:8080 2>/dev/null || echo "Open http://localhost:8080 in your browser"

open-sdwebui:
    @xdg-open http://localhost:7860 2>/dev/null || echo "Open http://localhost:7860 in your browser"

# Setup commands
setup-deps:
    @./setup_dependencies.sh

setup-ai:
    @./setup_ai_tools.sh

setup: setup-deps setup-ai
    @echo "Setup complete! Reload your shell with: source ~/.bashrc"

# Test scripts
test:
    @./test_scripts.sh

# Health check
health:
    @echo "=== AI Lab Health Check ==="
    @echo ""
    @echo "ComfyUI:"
    @curl -s http://localhost:8188 >/dev/null && echo "  ✓ Running" || echo "  ✗ Not running"
    @echo "Ollama:"
    @curl -s http://localhost:11434/api/version >/dev/null && echo "  ✓ Running" || echo "  ✗ Not running"
    @echo "OpenWebUI:"
    @curl -s http://localhost:8080 >/dev/null && echo "  ✓ Running" || echo "  ✗ Not running"
    @echo "SD WebUI:"
    @curl -s http://localhost:7860 >/dev/null && echo "  ✓ Running" || echo "  ✗ Not running"
