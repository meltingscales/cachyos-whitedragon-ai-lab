# CachyOS WhiteDragon AI Lab

## note

Converted from Debian 12 setup to work with CachyOS (Arch Linux) with optimized AMD GPU support.

## specs

CPU Model: Intel(R) Xeon(R) E5-2699 v3 @ 3.60GHz (2x); CPU Cores: 72; RAM Total: 251GB; GPU Model: AMD Radeon RX 7900 XTX; OS: CachyOS x86_64; Kernel: Linux 6.17.4-4-cachyos

## hosts

| Service | URL | Description |
|---------|-----|-------------|
| ComfyUI | http://localhost:8188 | Stable Diffusion workflow interface |
| Ollama | http://localhost:11434 | Local LLM API server |
| Open WebUI | http://localhost:8080 | Chat interface for Ollama models |
| Stable Diffusion WebUI | http://localhost:7860 | AUTOMATIC1111 WebUI |
| Text Generation WebUI | http://localhost:5000 | Oobabooga text generation interface |
| Jupyter Lab | http://localhost:8888 | Interactive Python notebooks |

## quick start

Install `just` command runner:
```bash
sudo pacman -S just
```

Start all services:
```bash
just start-all
```

Check service status:
```bash
just status
```

Common commands:
```bash
just start-comfyui      # Start ComfyUI
just start-ollama       # Start Ollama service
just start-openwebui    # Start Open WebUI
just stop-all           # Stop all services
just health             # Check if services are responding
just ollama-list        # List installed Ollama models
```

See all available commands:
```bash
just
```
