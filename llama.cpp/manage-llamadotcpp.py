#!/usr/bin/env python3
"""
Llama.cpp Model and Service Management Tool
Manages model downloads and systemd services for multiple llama-server instances.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Optional


class Config:
    """Configuration management"""
    def __init__(self, config_file: Path = Path("models.json")):
        self.config_file = config_file
        self.models_dir = Path(os.getenv("MODELS_DIR", Path.home() / "models"))
        self.working_dir = Path(__file__).parent.absolute()
        self.systemd_user_dir = Path.home() / ".config/systemd/user"
        self.user = os.getenv("USER")

        with open(config_file) as f:
            self.data = json.load(f)

    @property
    def models(self) -> List[Dict]:
        return self.data["models"]

    def get_model(self, name: str) -> Optional[Dict]:
        """Get model by name"""
        for model in self.models:
            if model["name"] == name:
                return model
        return None


class ModelDownloader:
    """Handle model downloads from HuggingFace"""

    def __init__(self, config: Config):
        self.config = config
        # Check for venv hf command first, then system huggingface-cli
        venv_hf = self.config.working_dir / ".venv/bin/hf"
        if venv_hf.exists():
            self.hf_cli = str(venv_hf)
        elif self._has_command("huggingface-cli"):
            self.hf_cli = "huggingface-cli"
        elif self._has_command("hf"):
            self.hf_cli = "hf"
        else:
            self.hf_cli = None

    def download(self, model: Dict) -> bool:
        """Download a single model"""
        self.config.models_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.config.models_dir / model["model_file"]

        # Skip if already downloaded
        if model_path.exists():
            file_size = model_path.stat().st_size / (1024**3)  # Size in GB
            print(f"✓ {model['display_name']} already downloaded ({file_size:.1f} GB)")
            return True

        print(f"Downloading {model['display_name']}...")
        repo = model["download"]["repo"]
        file = model["download"]["file"]

        # Try huggingface-cli first
        if self.hf_cli:
            return self._download_with_hf_cli(repo, file, model_path)
        else:
            print("huggingface-cli not found. Downloading with wget...")
            return self._download_with_wget(repo, file, model_path)

    def _download_with_hf_cli(self, repo: str, file: str, dest: Path) -> bool:
        """Download using huggingface-cli or hf command"""
        try:
            # Check if using 'hf' command (newer) or 'huggingface-cli' (older)
            is_hf_command = "hf" in self.hf_cli and "huggingface-cli" not in self.hf_cli

            if is_hf_command:
                # 'hf' command syntax (no --local-dir-use-symlinks option)
                subprocess.run([
                    self.hf_cli, "download", repo, file,
                    "--local-dir", str(self.config.models_dir)
                ], check=True)
            else:
                # 'huggingface-cli' command syntax
                subprocess.run([
                    self.hf_cli, "download", repo, file,
                    "--local-dir", str(self.config.models_dir),
                    "--local-dir-use-symlinks", "False"
                ], check=True)

            # Rename if needed
            downloaded = self.config.models_dir / file
            if downloaded != dest and downloaded.exists():
                downloaded.rename(dest)

            print(f"✓ Downloaded to {dest}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Download failed: {e}")
            return False

    def _download_with_wget(self, repo: str, file: str, dest: Path) -> bool:
        """Download using wget"""
        url = f"https://huggingface.co/{repo}/resolve/main/{file}"
        try:
            subprocess.run(["wget", "-O", str(dest), url], check=True)
            print(f"✓ Downloaded to {dest}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Download failed: {e}")
            return False

    @staticmethod
    def _has_command(cmd: str) -> bool:
        """Check if command exists"""
        return subprocess.run(
            ["command", "-v", cmd],
            shell=True,
            capture_output=True
        ).returncode == 0


class ServiceManager:
    """Manage systemd services"""

    def __init__(self, config: Config):
        self.config = config
        self.template_path = self.config.working_dir / "systemd/service.template"

    def install(self, model: Dict) -> bool:
        """Install systemd service for a model"""
        print(f"Installing {model['display_name']} service...")

        # Read template
        with open(self.template_path) as f:
            template = f.read()

        # Fill template
        service_content = template.format(
            display_name=model["display_name"],
            model_name=model["name"],
            user=self.config.user,
            working_dir=self.config.working_dir,
            models_dir=self.config.models_dir,
            model_file=model["model_file"],
            port=model["port"],
            context_size=model["context_size"]
        )

        # Write service file
        self.config.systemd_user_dir.mkdir(parents=True, exist_ok=True)
        service_file = self.config.systemd_user_dir / f"{model['service_name']}.service"

        with open(service_file, "w") as f:
            f.write(service_content)

        # Reload systemd
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        print(f"✓ Service installed: {model['service_name']}")
        return True

    def _systemctl(self, action: str, service_name: str) -> bool:
        """Run systemctl command"""
        try:
            subprocess.run(
                ["systemctl", "--user", action, f"{service_name}.service"],
                check=True,
                capture_output=True
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def enable(self, model: Dict) -> bool:
        """Enable service"""
        if self._systemctl("enable", model["service_name"]):
            print(f"✓ Enabled {model['display_name']}")
            return True
        return False

    def disable(self, model: Dict) -> bool:
        """Disable service"""
        if self._systemctl("disable", model["service_name"]):
            print(f"✓ Disabled {model['display_name']}")
            return True
        return False

    def start(self, model: Dict) -> bool:
        """Start service"""
        if self._systemctl("start", model["service_name"]):
            print(f"✓ Started {model['display_name']} on port {model['port']}")
            return True
        return False

    def stop(self, model: Dict) -> bool:
        """Stop service"""
        if self._systemctl("stop", model["service_name"]):
            print(f"✓ Stopped {model['display_name']}")
            return True
        return False

    def restart(self, model: Dict) -> bool:
        """Restart service"""
        if self._systemctl("restart", model["service_name"]):
            print(f"✓ Restarted {model['display_name']}")
            return True
        return False

    def status(self, model: Dict):
        """Show service status"""
        subprocess.run([
            "systemctl", "--user", "status",
            f"{model['service_name']}.service",
            "--no-pager"
        ])

    def logs(self, model: Dict, follow: bool = True):
        """Show service logs"""
        cmd = ["journalctl", "--user", "-u", f"{model['service_name']}.service"]
        if follow:
            cmd.append("-f")
        subprocess.run(cmd)

    def uninstall(self, model: Dict) -> bool:
        """Uninstall service"""
        self.stop(model)
        self.disable(model)

        service_file = self.config.systemd_user_dir / f"{model['service_name']}.service"
        if service_file.exists():
            service_file.unlink()

        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        print(f"✓ Uninstalled {model['display_name']}")
        return True

    def is_running(self, model: Dict) -> bool:
        """Check if service is running"""
        result = subprocess.run(
            ["systemctl", "--user", "is-active", f"{model['service_name']}.service"],
            capture_output=True,
            text=True
        )
        return result.stdout.strip() == "active"


class ChatSession:
    """Interactive chat with a running llama-server using Textual TUI"""

    def __init__(self, model: Dict, service_manager: ServiceManager):
        self.model = model
        self.service_manager = service_manager
        self.port = model['port']

        # Initialize OpenAI client pointing to local llama-server
        try:
            from openai import OpenAI
            self.client = OpenAI(
                base_url=f"http://localhost:{self.port}/v1",
                api_key="not-needed"  # llama-server doesn't require an API key
            )
        except ImportError:
            print("Error: OpenAI library not installed")
            print("Install it with: uv pip install -r requirements.txt")
            sys.exit(1)

    def get_model_context_size(self):
        """Query the model's context size from the server"""
        try:
            import requests
            response = requests.get(f"http://localhost:{self.port}/v1/models", timeout=5)
            if response.status_code == 200:
                data = response.json()
                # Try to extract context size from model info
                if 'data' in data and len(data['data']) > 0:
                    model_info = data['data'][0]
                    # llama-server might expose this in different ways
                    if 'context_length' in model_info:
                        return model_info['context_length']
                    elif 'max_tokens' in model_info:
                        return model_info['max_tokens']
        except Exception as e:
            print(f"Warning: Could not query model context size: {e}")

        # Fallback to configured value from models.json
        return self.model.get('context_size', 8192)

    def start(self):
        """Start interactive chat session"""
        # Check if service is running
        if not self.service_manager.is_running(self.model):
            print(f"Error: {self.model['display_name']} service is not running")
            print(f"Start it with: just start {self.model['name']}")
            sys.exit(1)

        # Get model context size
        self.model_context_size = self.get_model_context_size()
        print(f"Model context size: {self.model_context_size} tokens")

        # Start Textual TUI
        try:
            from textual.app import App, ComposeResult
            from textual.containers import Container, VerticalScroll, Horizontal
            from textual.widgets import Header, Footer, TextArea, RichLog, Static, Input, Button, DirectoryTree
            from textual.binding import Binding
            from textual.screen import ModalScreen
            from textual import work, on
            from pathlib import Path
            import os
        except ImportError:
            print("Error: Textual library not installed")
            print("Install it with: uv pip install -r requirements.txt")
            sys.exit(1)

        chat_session = self

        class FilePickerScreen(ModalScreen[str]):
            """Modal screen for file selection"""

            BINDINGS = [
                Binding("escape", "cancel", "Cancel"),
                Binding("u", "go_up", "Up Directory"),
            ]

            def __init__(self):
                super().__init__()
                self.current_path = Path.cwd()

            def compose(self) -> ComposeResult:
                from textual.widgets import Footer
                yield Container(
                    Static("File Browser - Arrow keys to navigate | Enter: Select | U: Up dir | Esc: Cancel", id="picker-header"),
                    Static(f"Current: {self.current_path}", id="current-dir"),
                    DirectoryTree(str(self.current_path), id="file-tree"),
                    id="file-picker-dialog"
                )

            @on(DirectoryTree.FileSelected)
            def on_file_selected(self, event: DirectoryTree.FileSelected) -> None:
                """Handle file selection"""
                self.dismiss(str(event.path))

            @on(DirectoryTree.DirectorySelected)
            def on_directory_selected(self, event: DirectoryTree.DirectorySelected) -> None:
                """Handle directory selection - update current path display"""
                self.current_path = event.path
                current_dir = self.query_one("#current-dir", Static)
                current_dir.update(f"Current: {self.current_path}")

            def action_go_up(self) -> None:
                """Go up one directory"""
                parent = self.current_path.parent
                if parent != self.current_path:  # Don't go above root
                    self.current_path = parent
                    # Recreate the tree with new path
                    file_tree = self.query_one("#file-tree", DirectoryTree)
                    file_tree.path = str(self.current_path)
                    file_tree.reload()
                    # Update display
                    current_dir = self.query_one("#current-dir", Static)
                    current_dir.update(f"Current: {self.current_path}")

            def action_cancel(self) -> None:
                """Cancel file selection"""
                self.dismiss(None)

        class ChatApp(App):
            """Textual chat application"""

            ENABLE_COMMAND_PALETTE = True

            def __init__(self, model_context_size: int):
                super().__init__()
                self.model_context_size = model_context_size
                self.history = []
                self.is_generating = False
                self.current_file_chunks = []
                self.current_chunk_index = 0
                self.current_file_path = ""
                # Reserve 20% of context for output, 10% for system/overhead
                self.usable_context = int(model_context_size * 0.7)
                # Message queue for handling multiple messages
                self.message_queue = []
                self.is_processing_queue = False

                # Initialize tokenizer for accurate token counting
                self.tokenizer = None
                try:
                    import tiktoken
                    # Use cl100k_base (GPT-4/ChatGPT tokenizer) as close approximation for Llama
                    self.tokenizer = tiktoken.get_encoding("cl100k_base")
                except Exception:
                    pass  # Fall back to estimation

            def estimate_tokens(self, text: str) -> int:
                """Accurate token counting using tiktoken"""
                if self.tokenizer:
                    try:
                        return len(self.tokenizer.encode(text))
                    except Exception:
                        pass

                # Fallback: conservative estimation for technical content
                # Technical logs (IPs, MACs, hex) tokenize worse than English
                return int(len(text) / 2.5)

            def get_history_tokens(self) -> int:
                """Estimate total tokens in conversation history"""
                total = 0
                for msg in self.history:
                    total += self.estimate_tokens(msg.get("content", ""))
                return total

            def should_summarize(self) -> bool:
                """Check if we should summarize to free up context"""
                history_tokens = self.get_history_tokens()
                # Summarize if we're using >80% of usable context
                return history_tokens > (self.usable_context * 0.8)

            def auto_summarize_history(self, keep_recent: int = 4) -> None:
                """Automatically summarize old conversation to free context"""
                if len(self.history) < 8:  # Need enough messages to summarize
                    return

                chat_log = self.query_one("#chat-log", RichLog)
                stats_bar = self.query_one("#stats-bar", Static)

                try:
                    # Keep only the most recent N messages
                    # For chunked files, we can be more aggressive (keep less)
                    recent_messages = self.history[-keep_recent:]
                    old_messages = self.history[:-keep_recent]

                    if not old_messages:
                        return

                    # Build conversation text for summarization
                    conversation_text = ""
                    for msg in old_messages:
                        role = msg["role"].capitalize()
                        content = msg["content"]
                        conversation_text += f"{role}: {content}\n\n"

                    # Ask LLM to summarize
                    self.call_from_thread(chat_log.write, "[yellow]⚙️  Context getting full, auto-summarizing older messages...[/]\n")

                    summary_request = f"Please provide a concise summary of this conversation history in 2-3 paragraphs. Focus on key points, decisions, and context:\n\n{conversation_text}"

                    # Create temporary history for summarization
                    temp_history = [{"role": "user", "content": summary_request}]

                    stream = chat_session.client.chat.completions.create(
                        model="local-model",
                        messages=temp_history,
                        temperature=0.5,
                        max_tokens=500,  # Keep summary concise
                        stream=True
                    )

                    summary = ""
                    for chunk in stream:
                        if chunk.choices[0].delta.content is not None:
                            summary += chunk.choices[0].delta.content

                    # Replace old messages with summary
                    summary_msg = {
                        "role": "system",
                        "content": f"[Previous conversation summary]: {summary.strip()}"
                    }

                    self.history = [summary_msg] + recent_messages

                    # Show what happened
                    old_tokens = sum(self.estimate_tokens(msg.get("content", "")) for msg in old_messages)
                    new_tokens = self.estimate_tokens(summary_msg["content"])
                    saved_tokens = old_tokens - new_tokens

                    self.call_from_thread(chat_log.write, f"[green]✓ Summarized {len(old_messages)} old messages, freed ~{saved_tokens} tokens[/]\n")

                except Exception as e:
                    self.call_from_thread(chat_log.write, f"[yellow]⚠ Auto-summarization failed: {e}[/]\n")
                    # Continue anyway - better to try sending than to fail completely

            CSS = """
            Screen {
                background: $surface;
            }

            #stats-container {
                height: auto;
                dock: top;
            }

            #stats-bar {
                height: 2;
                background: $boost;
                color: $text;
                padding: 0 1;
            }

            #gpu-stats {
                height: 5;
                background: $panel;
                padding: 1;
            }

            #chat-log {
                height: 1fr;
                border: solid $primary;
                background: $panel;
            }

            #file-container {
                height: 4;
                border: solid $warning;
            }

            #file-path {
                width: 1fr;
            }

            #file-info {
                height: 1;
                padding: 0 1;
            }

            #input-container {
                height: 8;
                border: solid $accent;
            }

            TextArea {
                height: 100%;
            }

            #file-picker-dialog {
                width: 80;
                height: 28;
                background: $panel;
                border: thick $primary;
            }

            #picker-header {
                width: 100%;
                background: $primary;
                color: $text;
                text-style: bold;
                padding: 1;
                text-align: center;
            }

            #current-dir {
                width: 100%;
                background: $boost;
                color: $text;
                padding: 0 1;
            }

            #file-tree {
                width: 100%;
                height: 1fr;
            }
            """

            BINDINGS = [
                Binding("ctrl+s", "send", "Send", show=True),
                Binding("ctrl+f", "attach_file", "Attach File", show=True),
                Binding("ctrl+d", "dump_log", "Save Log", show=True),
                Binding("ctrl+l", "clear", "Clear", show=True),
                Binding("ctrl+q", "quit", "Quit", show=True),
            ]


            def compose(self) -> ComposeResult:
                """Create child widgets"""
                yield Header()
                with Container(id="stats-container"):
                    yield Static("", id="stats-bar")
                    yield Static("GPU Stats: Loading...", id="gpu-stats")
                yield RichLog(id="chat-log", highlight=True, markup=True, wrap=True, auto_scroll=True)
                with Container(id="file-container"):
                    with Horizontal():
                        yield Static("File: ", shrink=True)
                        yield Input(placeholder="Enter file path (or Ctrl+F)", id="file-path")
                        yield Button("Clear", id="clear-file", variant="warning")
                    yield Static("", id="file-info")
                with Container(id="input-container"):
                    yield TextArea(id="input-area", language="markdown")
                yield Footer()

            def on_mount(self) -> None:
                """Set up the app on mount"""
                self.title = f"Chat: {chat_session.model['display_name']}"
                chat_log = self.query_one("#chat-log", RichLog)
                chat_log.write(f"[bold green]Connected to:[/] http://localhost:{chat_session.port}/v1\n")

                tokenizer_status = "[green]tiktoken[/]" if self.tokenizer else "[yellow]estimation[/]"
                chat_log.write(f"[dim]Context: {self.model_context_size} tokens | Token counting: {tokenizer_status} | Auto-summarization: Enabled[/]\n")
                chat_log.write(f"[dim]💡 Tip: Hold Shift and drag to select/copy text from the chat log[/]\n")

                # Start system stats update timer (every 5 seconds)
                self.set_interval(5.0, self.update_system_stats)
                self.update_system_stats()  # Initial update

                # Start context usage update timer (every 2 seconds)
                self.set_interval(2.0, self.update_context_display)
                self.update_context_display()  # Initial update

            def update_context_display(self) -> None:
                """Update context usage percentage display"""
                stats_bar = self.query_one("#stats-bar", Static)

                # Only update if not currently streaming
                if not self.is_generating:
                    history_tokens = self.get_history_tokens()
                    context_percent = (history_tokens / self.usable_context) * 100

                    # Color code based on usage
                    if context_percent >= 80:
                        color = "red"
                        icon = "🔴"
                    elif context_percent >= 60:
                        color = "yellow"
                        icon = "🟡"
                    else:
                        color = "green"
                        icon = "🟢"

                    context_text = f"{icon} Context: [{color}]{context_percent:.0f}%[/] ({history_tokens}/{self.usable_context} tokens)"

                    # Add queue info if messages are queued
                    if self.message_queue:
                        queue_count = len(self.message_queue)
                        context_text += f" | [yellow]📋 {queue_count} queued[/]"

                    stats_bar.update(context_text)

            def update_system_stats(self) -> None:
                """Update GPU, CPU, and RAM statistics"""
                import glob

                stats_widget = self.query_one("#gpu-stats", Static)

                # Collect all stats
                stats_lines = []

                # GPU stats
                try:
                    gpu_stats = self._read_amd_sysfs()
                    stats_lines.append(gpu_stats)
                except Exception:
                    stats_lines.append("[dim]GPU: N/A[/]")

                # CPU and RAM stats
                try:
                    cpu_ram_stats = self._read_cpu_ram_stats()
                    stats_lines.append(cpu_ram_stats)
                except Exception:
                    stats_lines.append("[dim]CPU/RAM: N/A[/]")

                stats_widget.update("\n".join(stats_lines))

            def _read_amd_sysfs(self) -> str:
                """Read AMD GPU stats from sysfs"""
                import glob
                from pathlib import Path

                gpu_info = []

                # Find all AMD GPU cards
                card_dirs = sorted(glob.glob("/sys/class/drm/card[0-9]*/device"))

                for card_dir in card_dirs[:2]:  # Limit to 2 GPUs
                    card_path = Path(card_dir)

                    # Check if this is an AMD GPU (has gpu_busy_percent)
                    busy_file = card_path / "gpu_busy_percent"
                    if not busy_file.exists():
                        continue

                    try:
                        # Read GPU usage
                        with open(busy_file, 'r') as f:
                            busy_percent = f.read().strip()

                        # Extract card number from path
                        card_num = card_path.parent.name.replace("card", "")

                        # Try to read VRAM info
                        mem_info = ""
                        mem_used_file = card_path / "mem_info_vram_used"
                        mem_total_file = card_path / "mem_info_vram_total"

                        if mem_used_file.exists() and mem_total_file.exists():
                            with open(mem_used_file, 'r') as f:
                                used_bytes = int(f.read().strip())
                            with open(mem_total_file, 'r') as f:
                                total_bytes = int(f.read().strip())

                            used_gb = used_bytes / (1024**3)
                            total_gb = total_bytes / (1024**3)
                            mem_info = f" VRAM:{used_gb:.1f}/{total_gb:.0f}GB"

                        gpu_info.append(f"GPU{card_num}:{busy_percent}%{mem_info}")

                    except Exception:
                        continue

                if gpu_info:
                    return "[green]●[/] " + " | ".join(gpu_info)

                raise Exception("No AMD GPU found")

            def _read_cpu_ram_stats(self) -> str:
                """Read CPU and RAM stats"""
                import os
                import psutil

                stats = []

                try:
                    # CPU info
                    cpu_count = os.cpu_count()
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    stats.append(f"[cyan]●[/] CPU:{cpu_percent}% ({cpu_count} cores)")

                    # RAM info
                    mem = psutil.virtual_memory()
                    used_gb = mem.used / (1024**3)
                    total_gb = mem.total / (1024**3)
                    mem_percent = mem.percent
                    stats.append(f"RAM:{mem_percent}% ({used_gb:.1f}/{total_gb:.1f}GB)")

                except ImportError:
                    # Fallback if psutil not available
                    cpu_count = os.cpu_count()
                    stats.append(f"[cyan]●[/] CPU: {cpu_count} cores")

                    # Read RAM from /proc/meminfo
                    try:
                        with open('/proc/meminfo', 'r') as f:
                            lines = f.readlines()
                            mem_total = 0
                            mem_available = 0
                            for line in lines:
                                if line.startswith('MemTotal:'):
                                    mem_total = int(line.split()[1]) / (1024**2)  # Convert to GB
                                elif line.startswith('MemAvailable:'):
                                    mem_available = int(line.split()[1]) / (1024**2)
                            mem_used = mem_total - mem_available
                            mem_percent = (mem_used / mem_total * 100) if mem_total > 0 else 0
                            stats.append(f"RAM:{mem_percent:.0f}% ({mem_used:.1f}/{mem_total:.1f}GB)")
                    except:
                        stats.append("RAM: N/A")

                return " | ".join(stats)

            def action_attach_file(self) -> None:
                """Open file picker (Ctrl+F)"""
                self.open_file_picker()

            @work(exclusive=False)
            async def open_file_picker(self) -> None:
                """Open file picker modal"""
                file_path = await self.push_screen_wait(FilePickerScreen())
                if file_path:
                    file_input = self.query_one("#file-path", Input)
                    file_input.value = file_path
                    self.load_and_chunk_file(file_path)

            def on_input_changed(self, event: Input.Changed) -> None:
                """Handle input changes"""
                if event.input.id == "file-path" and event.value.strip():
                    self.load_and_chunk_file(event.value.strip())

            def load_and_chunk_file(self, file_path: str) -> None:
                """Load file and split into chunks dynamically based on context"""
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    file_size = len(content)
                    file_tokens = self.estimate_tokens(content)
                    self.current_file_path = file_path

                    # Calculate how much context we have available
                    history_tokens = self.get_history_tokens()
                    available_tokens = self.usable_context - history_tokens

                    # Reserve tokens for framing messages and responses (~500 tokens overhead per chunk)
                    # Use conservative chunking: max 1000 tokens per chunk
                    # This ensures we have room for multiple chunks + responses in context
                    safe_chunk_tokens = min(1000, available_tokens // 5)  # Use 1/5 of available, max 1000 tokens

                    # Estimate characters for this token count (tiktoken gives accurate count)
                    if self.tokenizer:
                        # With accurate tokenizer, we can be more precise
                        chunk_size_chars = safe_chunk_tokens * 3  # Conservative: 3 chars/token average
                    else:
                        chunk_size_chars = int(safe_chunk_tokens * 2.5)  # Fallback estimation

                    file_info = self.query_one("#file-info", Static)

                    # Get filename for display
                    from pathlib import Path
                    filename = Path(file_path).name

                    # Check if file fits in context at all
                    if file_tokens > available_tokens:
                        # Need to chunk
                        self.current_file_chunks = []
                        for i in range(0, file_size, chunk_size_chars):
                            self.current_file_chunks.append(content[i:i + chunk_size_chars])

                        num_chunks = len(self.current_file_chunks)
                        chunk_size_kb = chunk_size_chars / 1024

                        # Warn if we'll exceed context even with chunking
                        total_messages_needed = num_chunks * 2  # user + assistant per chunk
                        if total_messages_needed > 20:
                            file_info.update(f"[bold cyan]📄 {filename}[/] | [yellow]{file_size / 1024:.1f}KB | {num_chunks} chunks | Will auto-summarize to manage context[/]")
                        else:
                            file_info.update(f"[bold cyan]📄 {filename}[/] | [yellow]{file_size / 1024:.1f}KB | {num_chunks} chunks (~{chunk_size_kb:.0f}KB each) | Context: {history_tokens}/{self.usable_context} tokens[/]")
                    else:
                        # File fits in one chunk
                        self.current_file_chunks = [content]
                        file_info.update(f"[bold cyan]📄 {filename}[/] | Size: {file_size / 1024:.1f}KB | Fits in context ({file_tokens} tokens)")

                    self.current_chunk_index = 0

                except Exception as e:
                    file_info = self.query_one("#file-info", Static)
                    file_info.update(f"[red]Error loading file: {e}[/]")

            def on_button_pressed(self, event: Button.Pressed) -> None:
                """Handle button presses"""
                if event.button.id == "clear-file":
                    file_input = self.query_one("#file-path", Input)
                    file_input.value = ""
                    self.current_file_chunks = []
                    self.current_chunk_index = 0
                    self.current_file_path = ""

                    file_info = self.query_one("#file-info", Static)
                    file_info.update("")

            def action_send(self) -> None:
                """Send message (Ctrl+S)"""
                input_area = self.query_one("#input-area", TextArea)
                user_message = input_area.text.strip()

                if not user_message:
                    return

                # Clear input
                input_area.clear()

                chat_log = self.query_one("#chat-log", RichLog)

                # Check for attached file chunks
                if self.current_file_chunks:
                    total_chunks = len(self.current_file_chunks)

                    if total_chunks > 1:
                        # Multi-chunk file
                        if self.is_generating:
                            # Queue the message
                            queue_position = len(self.message_queue) + 1
                            self.message_queue.append({
                                "type": "multi_chunk",
                                "message": user_message,
                                "total_chunks": total_chunks
                            })
                            chat_log.write(f"[yellow]📋 Message queued (position {queue_position})[/]\n")
                        else:
                            # Send immediately
                            self.send_multi_chunk_file(user_message, total_chunks)
                    else:
                        # Single chunk file
                        message = f"{user_message}\n\n--- File: {self.current_file_path} ---\n{self.current_file_chunks[0]}\n--- End of file ---"

                        if self.is_generating:
                            # Queue the message
                            queue_position = len(self.message_queue) + 1
                            self.message_queue.append({
                                "type": "single",
                                "message": message
                            })
                            chat_log.write(f"[yellow]📋 Message queued (position {queue_position}): {user_message[:50]}...[/]\n")
                        else:
                            # Send immediately
                            chat_log.write(f"\n[bold cyan]You:[/] {message}\n")
                            self.history.append({"role": "user", "content": message})
                            self.send_message()

                        # Clear file after queuing/sending
                        self.current_file_chunks = []
                        self.current_file_path = ""
                        file_info = self.query_one("#file-info", Static)
                        file_info.update("")
                        file_input = self.query_one("#file-path", Input)
                        file_input.value = ""
                else:
                    # No file attached
                    if self.is_generating:
                        # Queue the message
                        queue_position = len(self.message_queue) + 1
                        self.message_queue.append({
                            "type": "single",
                            "message": user_message
                        })
                        chat_log.write(f"[yellow]📋 Message queued (position {queue_position}): {user_message[:50]}...[/]\n")
                    else:
                        # Send immediately
                        chat_log.write(f"\n[bold cyan]You:[/] {user_message}\n")
                        self.history.append({"role": "user", "content": user_message})
                        self.send_message()

            def dump_to_log(self, error_context: str = None) -> str:
                """Dump conversation history to a log file"""
                from datetime import datetime
                from pathlib import Path

                # Create logs directory
                logs_dir = Path.cwd() / "chat_logs"
                logs_dir.mkdir(exist_ok=True)

                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"chat_{chat_session.model['name']}_{timestamp}.log"
                log_path = logs_dir / filename

                # Write conversation to file
                with open(log_path, 'w', encoding='utf-8') as f:
                    f.write(f"Chat Log - {chat_session.model['display_name']}\n")
                    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Model Context Size: {self.model_context_size} tokens\n")
                    f.write(f"Context Usage: {self.get_history_tokens()}/{self.usable_context} tokens\n")
                    f.write("=" * 80 + "\n\n")

                    if error_context:
                        f.write(f"ERROR CONTEXT:\n{error_context}\n")
                        f.write("=" * 80 + "\n\n")

                    # Write conversation history
                    for i, msg in enumerate(self.history, 1):
                        role = msg.get("role", "unknown").upper()
                        content = msg.get("content", "")
                        f.write(f"[Message {i}] {role}:\n")
                        f.write(f"{content}\n")
                        f.write("-" * 80 + "\n\n")

                    # Write metadata
                    f.write("=" * 80 + "\n")
                    f.write(f"Total messages: {len(self.history)}\n")
                    f.write(f"Log saved at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

                return str(log_path)

            def action_dump_log(self) -> None:
                """Save chat log to file (Ctrl+D)"""
                try:
                    log_path = self.dump_to_log()
                    chat_log = self.query_one("#chat-log", RichLog)
                    chat_log.write(f"[green]✓ Chat log saved to: {log_path}[/]\n")
                except Exception as e:
                    chat_log = self.query_one("#chat-log", RichLog)
                    chat_log.write(f"[red]✗ Failed to save log: {e}[/]\n")

            def action_clear(self) -> None:
                """Clear chat history (Ctrl+L)"""
                if not self.is_generating:
                    self.history = []
                    self.message_queue = []  # Also clear queue
                    chat_log = self.query_one("#chat-log", RichLog)
                    chat_log.clear()
                    chat_log.write(f"[bold green]Connected to:[/] http://localhost:{chat_session.port}/v1\n")
                    chat_log.write("[yellow]Chat history cleared[/]\n")
                    # Clear stats
                    stats_bar = self.query_one("#stats-bar", Static)
                    stats_bar.update("")

            def process_message_queue(self) -> None:
                """Process next message in queue if not currently generating"""
                if self.is_processing_queue or self.is_generating or not self.message_queue:
                    return

                self.is_processing_queue = True

                # Get next message from queue
                queued_item = self.message_queue.pop(0)
                message_type = queued_item["type"]

                chat_log = self.query_one("#chat-log", RichLog)

                if message_type == "single":
                    # Single message or single-chunk file
                    chat_log.write(f"\n[bold cyan]You:[/] {queued_item['message']}\n")
                    self.history.append({"role": "user", "content": queued_item["message"]})
                    self.send_message()

                elif message_type == "multi_chunk":
                    # Multi-chunk file
                    self.send_multi_chunk_file(queued_item["message"], queued_item["total_chunks"])

                self.is_processing_queue = False

            @work(exclusive=True, thread=True)
            def send_multi_chunk_file(self, user_message: str, total_chunks: int) -> None:
                """Send multi-chunk file with framing messages"""
                import time

                self.is_generating = True
                chat_log = self.query_one("#chat-log", RichLog)
                stats_bar = self.query_one("#stats-bar", Static)

                try:
                    # PROACTIVE check at start: do we have enough space for chunked file?
                    # Estimate: each chunk ~1500 tokens, need space for at least 3-4 chunks
                    current_tokens = self.get_history_tokens()
                    if current_tokens > (self.usable_context * 0.3):  # Very aggressive: 30% for chunk mode
                        self.call_from_thread(chat_log.write, f"[yellow]⚙️  Pre-chunking summarization to ensure space...[/]\n")
                        self.auto_summarize_history(keep_recent=1)

                    # 1. Send user's question FIRST so AI knows what to focus on during chunks
                    initial_prompt = f"Question: {user_message}\n\nI'm going to send you a file to help answer this. Please acknowledge and wait for all chunks."
                    self.call_from_thread(chat_log.write, f"\n[bold cyan]You:[/] {initial_prompt}\n")
                    self.history.append({"role": "user", "content": initial_prompt})
                    self._send_and_get_acknowledgment("Initial prompt")

                    # 2. Send framing message about chunks
                    framing_msg = f"Now sending file '{self.current_file_path}' in {total_chunks} chunks. Just acknowledge each chunk with 'OK'."
                    self.call_from_thread(chat_log.write, f"\n[bold cyan]You:[/] {framing_msg}\n")
                    self.history.append({"role": "user", "content": framing_msg})
                    self._send_and_get_acknowledgment("Framing message")

                    # 3. Send each chunk and wait for response
                    chunk_start_time = time.time()
                    chunk_times = []  # Track time for each chunk

                    for i, chunk in enumerate(self.current_file_chunks, 1):
                        chunk_iteration_start = time.time()
                        chunk_msg = f"--- Chunk {i}/{total_chunks} ---\n{chunk}\n--- End of chunk {i} ---"

                        # Calculate and display stats BEFORE sending
                        elapsed_total = time.time() - chunk_start_time
                        progress_pct = int(((i - 1) / total_chunks) * 100)  # Use i-1 for "about to send chunk i"

                        # Get context usage for display
                        context_tokens = self.get_history_tokens()

                        # Show estimated stats before sending
                        if i > 1 and chunk_times:
                            avg_time_per_chunk = sum(chunk_times) / len(chunk_times)
                            chunks_remaining = total_chunks - i + 1  # +1 because we haven't sent this one yet
                            eta_seconds = avg_time_per_chunk * chunks_remaining

                            # Format elapsed time
                            if elapsed_total < 60:
                                elapsed_str = f"{int(elapsed_total)}s"
                            else:
                                elapsed_min = int(elapsed_total / 60)
                                elapsed_sec = int(elapsed_total % 60)
                                elapsed_str = f"{elapsed_min}m {elapsed_sec}s"

                            # Format ETA
                            if eta_seconds < 60:
                                eta_str = f"{int(eta_seconds)}s"
                            else:
                                eta_minutes = int(eta_seconds / 60)
                                eta_secs = int(eta_seconds % 60)
                                eta_str = f"{eta_minutes}m {eta_secs}s"

                            stats_text = f"📤 Sending {i}/{total_chunks} ({progress_pct}%) | Elapsed: {elapsed_str} | ETA: {eta_str} | Context: {context_tokens}/{self.usable_context}"
                        else:
                            # First chunk - no ETA yet
                            stats_text = f"📤 Sending {i}/{total_chunks} (0%) | Context: {context_tokens}/{self.usable_context}"

                        self.call_from_thread(stats_bar.update, stats_text)

                        # PROACTIVE check: will this chunk fit?
                        chunk_tokens = self.estimate_tokens(chunk_msg)
                        current_tokens = self.get_history_tokens()
                        # Reserve 500 tokens for response space
                        total_if_sent = current_tokens + chunk_tokens + 500

                        # If adding this chunk would exceed 40% of usable context, summarize FIRST
                        # (Very aggressive: 40% to account for token estimation errors)
                        if total_if_sent > (self.usable_context * 0.4):
                            self.call_from_thread(chat_log.write, f"[yellow]⚙️  Next chunk would exceed context, summarizing first...[/]\n")
                            # Be very aggressive for chunked files - only keep last message
                            self.auto_summarize_history(keep_recent=1)

                        self.call_from_thread(chat_log.write, f"\n[bold cyan]You:[/] [Sending chunk {i}/{total_chunks}...]\n")

                        self.history.append({"role": "user", "content": chunk_msg})
                        self._send_and_get_acknowledgment(f"Chunk {i}/{total_chunks}")

                        # Track time for this chunk
                        chunk_iteration_end = time.time()
                        chunk_times.append(chunk_iteration_end - chunk_iteration_start)

                    # After all chunks sent - show completion
                    elapsed_total = time.time() - chunk_start_time
                    if elapsed_total < 60:
                        elapsed_str = f"{int(elapsed_total)}s"
                    else:
                        elapsed_min = int(elapsed_total / 60)
                        elapsed_sec = int(elapsed_total % 60)
                        elapsed_str = f"{elapsed_min}m {elapsed_sec}s"

                    self.call_from_thread(stats_bar.update,
                        f"✓ All {total_chunks} chunks sent in {elapsed_str}")

                    # 4. Send final message restating the question for full response
                    final_msg = f"[END OF FILE] All {total_chunks} chunks sent. Now please answer my original question: {user_message}"
                    self.call_from_thread(chat_log.write, f"\n[bold cyan]You:[/] {final_msg}\n")
                    self.history.append({"role": "user", "content": final_msg})

                    # Get final streaming response
                    self._send_final_streaming_response()

                    # Clear file after sending all chunks
                    self.current_file_chunks = []
                    self.current_file_path = ""
                    def clear_file_ui():
                        file_info = self.query_one("#file-info", Static)
                        file_info.update("")
                        file_input = self.query_one("#file-path", Input)
                        file_input.value = ""
                    self.call_from_thread(clear_file_ui)

                except Exception as e:
                    error_msg = f"Error in send_multi_chunk_file: {e}"
                    self.call_from_thread(chat_log.write, f"\n[red]✗ {error_msg}[/]\n")

                    # Auto-dump log on error
                    try:
                        import traceback
                        error_details = f"{error_msg}\n\nTraceback:\n{traceback.format_exc()}"
                        log_path = self.dump_to_log(error_context=error_details)
                        self.call_from_thread(chat_log.write, f"[yellow]📝 Error log saved to: {log_path}[/]\n")
                    except Exception as log_error:
                        self.call_from_thread(chat_log.write, f"[red]⚠ Failed to save error log: {log_error}[/]\n")

                    # Clean up history on error
                    if self.history and self.history[-1]["role"] == "user":
                        self.history.pop()
                finally:
                    self.is_generating = False
                    # Process next message in queue if any
                    self.call_from_thread(self.process_message_queue)

            def _send_and_get_acknowledgment(self, label: str) -> None:
                """Send message to LLM and get brief acknowledgment"""
                import time

                chat_log = self.query_one("#chat-log", RichLog)
                stats_bar = self.query_one("#stats-bar", Static)

                try:
                    start_time = time.time()

                    # Make actual API call with current history
                    stream = chat_session.client.chat.completions.create(
                        model="local-model",
                        messages=self.history,
                        temperature=0.7,
                        max_tokens=10,  # Very short acknowledgment only
                        stream=True
                    )

                    full_response = ""
                    for chunk in stream:
                        if chunk.choices[0].delta.content is not None:
                            full_response += chunk.choices[0].delta.content

                    elapsed = time.time() - start_time

                    # Add brief acknowledgment to history
                    self.history.append({"role": "assistant", "content": full_response.strip()})
                    self.call_from_thread(chat_log.write, f"[dim][green]✓ {label} acknowledged ({elapsed:.1f}s)[/][/]\n")

                except Exception as e:
                    self.call_from_thread(chat_log.write, f"[red]✗ Error with {label}: {e}[/]\n")
                    raise

            def _send_final_streaming_response(self) -> None:
                """Send final message and stream full response"""
                import time

                chat_log = self.query_one("#chat-log", RichLog)
                stats_bar = self.query_one("#stats-bar", Static)

                try:
                    start_time = time.time()

                    stream = chat_session.client.chat.completions.create(
                        model="local-model",
                        messages=self.history,
                        temperature=0.7,
                        max_tokens=-1,
                        stream=True
                    )

                    full_response = ""
                    buffer = "[bold green]Assistant:[/] "
                    chunk_count = 0
                    char_count = 0

                    # Animation frames
                    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

                    for chunk in stream:
                        if chunk.choices[0].delta.content is not None:
                            content = chunk.choices[0].delta.content
                            full_response += content
                            buffer += content
                            chunk_count += 1
                            char_count += len(content)

                            # Update stats with animation
                            elapsed = time.time() - start_time
                            frame = frames[chunk_count % len(frames)]
                            chars_per_sec = char_count / elapsed if elapsed > 0 else 0
                            stats_text = f"{frame} Streaming... | {char_count} chars | {chunk_count} chunks | {chars_per_sec:.1f} chars/s | {elapsed:.1f}s"
                            self.call_from_thread(stats_bar.update, stats_text)

                            # Write when we encounter a newline
                            if '\n' in buffer:
                                self.call_from_thread(chat_log.write, buffer.rstrip('\n'))
                                buffer = ""

                    # Write any remaining buffered content
                    if buffer:
                        self.call_from_thread(chat_log.write, buffer.rstrip('\n'))

                    # Final stats
                    elapsed = time.time() - start_time
                    final_stats = f"✓ Complete | {char_count} chars | {chunk_count} chunks | {elapsed:.2f}s"
                    self.call_from_thread(stats_bar.update, final_stats)

                    # Add to history
                    self.history.append({"role": "assistant", "content": full_response})

                except Exception as e:
                    self.call_from_thread(chat_log.write, f"\n[red]Error: {e}[/]")

            @work(exclusive=True, thread=True)
            def send_message(self) -> None:
                """Send message and stream response"""
                import time

                self.is_generating = True
                chat_log = self.query_one("#chat-log", RichLog)
                stats_bar = self.query_one("#stats-bar", Static)

                try:
                    # PROACTIVE check: will the current message history fit?
                    # Check BEFORE sending, not after
                    current_tokens = self.get_history_tokens()
                    if current_tokens > (self.usable_context * 0.5):
                        self.auto_summarize_history()

                    start_time = time.time()

                    stream = chat_session.client.chat.completions.create(
                        model="local-model",
                        messages=self.history,
                        temperature=0.7,
                        max_tokens=-1,
                        stream=True
                    )

                    full_response = ""
                    buffer = "[bold green]Assistant:[/] "
                    chunk_count = 0
                    char_count = 0

                    # Animation frames
                    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

                    for chunk in stream:
                        if chunk.choices[0].delta.content is not None:
                            content = chunk.choices[0].delta.content
                            full_response += content
                            buffer += content
                            chunk_count += 1
                            char_count += len(content)

                            # Update stats with animation
                            elapsed = time.time() - start_time
                            frame = frames[chunk_count % len(frames)]
                            chars_per_sec = char_count / elapsed if elapsed > 0 else 0
                            stats_text = f"{frame} Streaming... | {char_count} chars | {chunk_count} chunks | {chars_per_sec:.1f} chars/s | {elapsed:.1f}s"
                            self.call_from_thread(stats_bar.update, stats_text)

                            # Write when we encounter a newline (strip it since write() adds one)
                            if '\n' in buffer:
                                self.call_from_thread(chat_log.write, buffer.rstrip('\n'))
                                buffer = ""

                    # Write any remaining buffered content
                    if buffer:
                        self.call_from_thread(chat_log.write, buffer.rstrip('\n'))

                    # Final stats
                    elapsed = time.time() - start_time
                    final_stats = f"✓ Complete | {char_count} chars | {chunk_count} chunks | {elapsed:.2f}s"
                    self.call_from_thread(stats_bar.update, final_stats)

                    # Add to history
                    self.history.append({"role": "assistant", "content": full_response})

                except KeyboardInterrupt:
                    self.call_from_thread(chat_log.write, "\n[yellow]Interrupted[/]")
                    self.call_from_thread(stats_bar.update, "✗ Interrupted")
                    # Remove user message on interrupt
                    if self.history and self.history[-1]["role"] == "user":
                        self.history.pop()
                except Exception as e:
                    error_msg = f"Error in send_message: {e}"
                    self.call_from_thread(chat_log.write, f"\n[red]✗ {error_msg}[/]")
                    self.call_from_thread(stats_bar.update, f"✗ Error: {e}")

                    # Auto-dump log on error
                    try:
                        import traceback
                        error_details = f"{error_msg}\n\nTraceback:\n{traceback.format_exc()}"
                        log_path = self.dump_to_log(error_context=error_details)
                        self.call_from_thread(chat_log.write, f"[yellow]📝 Error log saved to: {log_path}[/]\n")
                    except Exception as log_error:
                        self.call_from_thread(chat_log.write, f"[red]⚠ Failed to save error log: {log_error}[/]\n")

                    # Remove user message on error
                    if self.history and self.history[-1]["role"] == "user":
                        self.history.pop()
                finally:
                    self.is_generating = False
                    # Process next message in queue if any
                    self.call_from_thread(self.process_message_queue)

        # Run the app
        app = ChatApp(chat_session.model_context_size)
        app.run()


def main():
    """Main CLI entry point"""
    if len(sys.argv) < 2:
        print("Usage: manage.py <command> [model_name]")
        print("\nCommands:")
        print("  download [model]     Download model(s)")
        print("  verify               Verify model files")
        print("  install [model]      Install service(s)")
        print("  enable [model]       Enable service(s)")
        print("  disable [model]      Disable service(s)")
        print("  start [model]        Start service(s)")
        print("  stop [model]         Stop service(s)")
        print("  restart [model]      Restart service(s)")
        print("  status [model]       Show service status")
        print("  logs <model>         Show service logs")
        print("  chat <model>         Interactive chat with model")
        print("  uninstall [model]    Uninstall service(s)")
        print("  endpoints            Show all endpoints")
        print("  list                 List all models")
        sys.exit(1)

    config = Config()
    downloader = ModelDownloader(config)
    services = ServiceManager(config)

    command = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else None

    # Get target models
    if model_name:
        model = config.get_model(model_name)
        if not model:
            print(f"Error: Model '{model_name}' not found")
            sys.exit(1)
        models = [model]
    else:
        models = config.models

    # Execute command
    if command == "download":
        for model in models:
            downloader.download(model)

    elif command == "verify":
        print("=== Verifying Model Files ===")
        print(f"Models directory: {config.models_dir}\n")
        for model in config.models:
            path = config.models_dir / model["model_file"]
            if path.exists():
                print(f"✓ {model['model_file']}")
            else:
                print(f"✗ {model['model_file']} (missing)")

    elif command == "install":
        for model in models:
            services.install(model)

    elif command == "enable":
        for model in models:
            services.enable(model)

    elif command == "disable":
        for model in models:
            services.disable(model)

    elif command == "start":
        for model in models:
            services.start(model)

    elif command == "stop":
        for model in models:
            services.stop(model)

    elif command == "restart":
        for model in models:
            services.restart(model)

    elif command == "status":
        for model in models:
            print(f"\n--- {model['display_name']} (Port {model['port']}) ---")
            services.status(model)

    elif command == "logs":
        if not model_name:
            print("Error: logs command requires a model name")
            sys.exit(1)
        services.logs(models[0])

    elif command == "chat":
        if not model_name:
            print("Error: chat command requires a model name")
            sys.exit(1)
        chat = ChatSession(models[0], services)
        chat.start()

    elif command == "uninstall":
        for model in models:
            services.uninstall(model)

    elif command == "endpoints":
        print("=== Llama Server Endpoints ===")
        for model in config.models:
            print(f"{model['display_name']:25} http://localhost:{model['port']}")
        print(f"\nModel files location: {config.models_dir}")

    elif command == "list":
        print("=== Available Models ===")
        for model in config.models:
            print(f"  {model['name']:20} - {model['display_name']} (port {model['port']})")

    else:
        print(f"Error: Unknown command '{command}'")
        sys.exit(1)


if __name__ == "__main__":
    main()
