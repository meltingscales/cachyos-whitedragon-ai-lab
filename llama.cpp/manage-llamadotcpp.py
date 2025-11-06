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

    def start(self):
        """Start interactive chat session"""
        # Check if service is running
        if not self.service_manager.is_running(self.model):
            print(f"Error: {self.model['display_name']} service is not running")
            print(f"Start it with: just start {self.model['name']}")
            sys.exit(1)

        # Start Textual TUI
        try:
            from textual.app import App, ComposeResult
            from textual.containers import Container, VerticalScroll
            from textual.widgets import Header, Footer, TextArea, RichLog, Static
            from textual.binding import Binding
            from textual import work
        except ImportError:
            print("Error: Textual library not installed")
            print("Install it with: uv pip install -r requirements.txt")
            sys.exit(1)

        chat_session = self

        class ChatApp(App):
            """Textual chat application"""

            CSS = """
            Screen {
                background: $surface;
            }

            #chat-log {
                height: 1fr;
                border: solid $primary;
                background: $panel;
            }

            #input-container {
                height: 8;
                border: solid $accent;
            }

            TextArea {
                height: 100%;
            }

            .status-line {
                background: $accent;
                color: $text;
                text-style: bold;
                height: 1;
                content-align: center middle;
            }
            """

            BINDINGS = [
                Binding("ctrl+s", "send", "Send", show=True),
                Binding("ctrl+l", "clear", "Clear", show=True),
                Binding("ctrl+q", "quit", "Quit", show=True),
            ]

            def __init__(self):
                super().__init__()
                self.history = []
                self.is_generating = False

            def compose(self) -> ComposeResult:
                """Create child widgets"""
                yield Header()
                yield RichLog(id="chat-log", highlight=True, markup=True, wrap=True, auto_scroll=True)
                with Container(id="input-container"):
                    yield TextArea(id="input-area", language="markdown")
                yield Static("Ctrl+S: Send | Ctrl+L: Clear | Ctrl+Q: Quit", classes="status-line")
                yield Footer()

            def on_mount(self) -> None:
                """Set up the app on mount"""
                self.title = f"Chat: {chat_session.model['display_name']}"
                chat_log = self.query_one("#chat-log", RichLog)
                chat_log.write(f"[bold green]Connected to:[/] http://localhost:{chat_session.port}/v1\n")

            def action_send(self) -> None:
                """Send message (Ctrl+S)"""
                if self.is_generating:
                    return

                input_area = self.query_one("#input-area", TextArea)
                message = input_area.text.strip()

                if not message:
                    return

                # Clear input
                input_area.clear()

                # Display user message
                chat_log = self.query_one("#chat-log", RichLog)
                chat_log.write(f"\n[bold cyan]You:[/] {message}\n")

                # Add to history and send
                self.history.append({"role": "user", "content": message})
                self.send_message()

            def action_clear(self) -> None:
                """Clear chat history (Ctrl+L)"""
                if not self.is_generating:
                    self.history = []
                    chat_log = self.query_one("#chat-log", RichLog)
                    chat_log.clear()
                    chat_log.write(f"[bold green]Connected to:[/] http://localhost:{chat_session.port}/v1\n")
                    chat_log.write("[yellow]Chat history cleared[/]\n")

            @work(exclusive=True, thread=True)
            def send_message(self) -> None:
                """Send message and stream response"""
                self.is_generating = True
                chat_log = self.query_one("#chat-log", RichLog)

                try:
                    stream = chat_session.client.chat.completions.create(
                        model="local-model",
                        messages=self.history,
                        temperature=0.7,
                        max_tokens=-1,
                        stream=True
                    )

                    full_response = ""
                    buffer = "[bold green]Assistant:[/] "

                    for chunk in stream:
                        if chunk.choices[0].delta.content is not None:
                            content = chunk.choices[0].delta.content
                            full_response += content
                            buffer += content

                            # Write when we encounter a newline (strip it since write() adds one)
                            if '\n' in buffer:
                                self.call_from_thread(chat_log.write, buffer.rstrip('\n'))
                                buffer = ""

                    # Write any remaining buffered content
                    if buffer:
                        self.call_from_thread(chat_log.write, buffer.rstrip('\n'))

                    # Add to history
                    self.history.append({"role": "assistant", "content": full_response})

                except KeyboardInterrupt:
                    self.call_from_thread(chat_log.write, "\n[yellow]Interrupted[/]")
                    # Remove user message on interrupt
                    if self.history and self.history[-1]["role"] == "user":
                        self.history.pop()
                except Exception as e:
                    self.call_from_thread(chat_log.write, f"\n[red]Error: {e}[/]")
                    # Remove user message on error
                    if self.history and self.history[-1]["role"] == "user":
                        self.history.pop()
                finally:
                    self.is_generating = False

        # Run the app
        app = ChatApp()
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
