# Adding Models to llama.cpp

This guide shows you how to add new models to the `models.json` configuration file.

## Quick Start

To add a new model, you need to:
1. Find the GGUF file on HuggingFace
2. Add an entry to `models.json`
3. Run `just download <model-name>` to download it
4. Run `just install <model-name>` to create the systemd service

## Understanding HuggingFace Model Links

HuggingFace models can be referenced in several ways:

### Short Links (hf.co)
```
hf.co/hungng/Llama-3.2-uncensored-erotica:F16
```
This breaks down to:
- `hungng` = username/organization
- `Llama-3.2-uncensored-erotica` = repository name
- `:F16` = tag (usually a quantization type)

### Full URLs
```
https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF
```

## Step-by-Step: Adding a Model

### 1. Find the GGUF Repository

Most models have a separate GGUF repository. For example:
- Original model: `hungng/Llama-3.2-uncensored-erotica`
- GGUF version: Usually ends with `-GGUF` (check user's repos)

Common GGUF quantizers:
- **bartowski** - Most popular, high-quality quants
- **mradermacher** - Alternative quantizations
- **TheBloke** - Older models (less active now)

**Example search:**
1. Go to https://huggingface.co/models
2. Search: `Llama-3.2-uncensored-erotica GGUF`
3. Look for repos ending in `-GGUF`

### 2. Browse Available Quantizations

Visit the GGUF repo and check the "Files and versions" tab.

Common quantization types (from smallest to largest):
- `Q2_K` - ~2.5 GB - Lowest quality, fastest
- `Q3_K_M` - ~3.5 GB - Low quality
- `Q4_K_M` - **~4.5 GB - Recommended balance** ⭐
- `Q4_K_S` - ~4.3 GB - Slightly smaller Q4
- `Q5_K_M` - ~5.5 GB - Good quality
- `Q6_K` - ~6.5 GB - High quality
- `Q8_0` - ~8 GB - Very high quality
- `F16` - Full size - Original quality

**Recommendation:** Start with `Q4_K_M` for best balance of size/quality.

### 3. Get the Exact Filename

Click on the file in HuggingFace to see its exact name. For example:
```
Llama-3.2-uncensored-erotica-Q4_K_M.gguf
```

**Important:** Copy the exact filename (case-sensitive, including hyphens/underscores).

### 4. Choose a Port

Pick an unused port for the model server:
- Existing models use: 8001, 8002, 8003, 8004
- Choose the next available (e.g., 8005, 8006, etc.)

### 5. Add to models.json

Edit `models.json` and add a new entry:

```json
{
  "name": "llama32-uncensored",
  "display_name": "Llama 3.2 Uncensored",
  "port": 8005,
  "service_name": "llama-server-llama32-uncensored",
  "model_file": "llama32-uncensored.gguf",
  "context_size": 4096,
  "download": {
    "repo": "bartowski/Llama-3.2-uncensored-erotica-GGUF",
    "file": "Llama-3.2-uncensored-erotica-Q4_K_M.gguf"
  }
}
```

**Field descriptions:**
- `name` - Short identifier (use in commands, no dots allowed!)
- `display_name` - Human-readable name
- `port` - Port number (must be unique)
- `service_name` - Systemd service name (prefix with `llama-server-`)
- `model_file` - Local filename (can be different from download name)
- `context_size` - Context window size (4096, 8192, 16384, etc.)
- `download.repo` - HuggingFace repo (`username/repo-name`)
- `download.file` - Exact GGUF filename from the repo

### 6. Context Size Guidelines

Common context sizes:
- **4096** - Most 3B-7B models
- **8192** - Most 13B-34B models
- **16384** - Larger models with extended context
- **32768+** - Specialized long-context models

Check the model card on HuggingFace for the official context size.

### 7. Download and Install

```bash
# Verify your config is valid
just list

# Download the model
just download llama32-uncensored

# Install the systemd service
just install llama32-uncensored

# Start the service
just start llama32-uncensored

# Check it's running
just status-model llama32-uncensored

# Test the endpoint
curl http://localhost:8005/health
```

## Complete Example

Let's add `hf.co/bartowski/Llama-3.2-3B-uncensored-GGUF`:

### 1. Research the Model
- Visit: https://huggingface.co/bartowski/Llama-3.2-3B-uncensored-GGUF
- Check available files
- Find: `Llama-3.2-3B-uncensored-Q4_K_M.gguf` (2.02 GB)

### 2. Add to models.json

```json
{
  "models": [
    ... existing models ...,
    {
      "name": "llama32-3b-uncensored",
      "display_name": "Llama 3.2 3B Uncensored",
      "port": 8005,
      "service_name": "llama-server-llama32-3b-uncensored",
      "model_file": "llama32-3b-uncensored.gguf",
      "context_size": 4096,
      "download": {
        "repo": "bartowski/Llama-3.2-3B-uncensored-GGUF",
        "file": "Llama-3.2-3B-uncensored-Q4_K_M.gguf"
      }
    }
  ]
}
```

### 3. Use it

```bash
just download llama32-3b-uncensored
just install llama32-3b-uncensored
just start llama32-3b-uncensored
```

## Finding GGUF Repos

### Method 1: Direct Search
Search on HuggingFace: `<model-name> GGUF`

Example: `Llama-3.2-uncensored GGUF`

### Method 2: Check Popular Quantizers

Check these users' profiles:
- https://huggingface.co/bartowski
- https://huggingface.co/mradermacher
- https://huggingface.co/TheBloke (older models)

### Method 3: Model Card Links

Many model cards have a "Quantized versions" section linking to GGUF repos.

## Troubleshooting

### Download Fails (404 Error)
- Double-check the repo name and filename (case-sensitive!)
- Verify the file exists in the HuggingFace repo
- Some repos require authentication (rare for GGUF)

### Service Won't Start
- Check logs: `just logs <model-name>`
- Verify model file exists: `just verify`
- Ensure port isn't already in use: `ss -tlnp | grep <port>`

### Out of Memory
- Try a smaller quantization (Q4_K_S instead of Q4_K_M)
- Check available VRAM: `rocm-smi` or `nvidia-smi`
- Reduce `n-gpu-layers` in the service file (advanced)

## Tips

1. **Start with Q4_K_M** - Best quality/size balance
2. **Use unique ports** - Each model needs its own port
3. **Check context size** - Larger context = more VRAM needed
4. **Name carefully** - Use lowercase, hyphens, no dots in `name` field
5. **Test first** - Download one model, test it works, then add more

## Advanced: Custom Service Options

To customize service parameters, edit the generated service file after installation:

```bash
# Edit the service
nano ~/.config/systemd/user/llama-server-<model-name>.service

# Reload systemd
systemctl --user daemon-reload

# Restart the service
just restart <model-name>
```

Common customizations:
- `--n-gpu-layers` - Number of layers on GPU (default: 99 = all)
- `-c` / `--ctx-size` - Context window size
- `--threads` - CPU threads to use
- `--batch-size` - Batch size for processing

## Integration with Open WebUI

Once your llama-server is running on a port (e.g., 8005):

1. Open Open WebUI: http://localhost:8080
2. Go to Settings → Connections
3. Add OpenAI-compatible endpoint: `http://localhost:8005`
4. The model will appear in the model selection dropdown

Each llama-server instance exposes an OpenAI-compatible API at its port.
