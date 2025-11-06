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
    """Interactive chat with a running llama-server using asciimatics TUI"""

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

        # Start asciimatics TUI
        try:
            from asciimatics.widgets import Frame, Layout, TextBox, Text, Button, Label
            from asciimatics.scene import Scene
            from asciimatics.screen import Screen
            from asciimatics.exceptions import ResizeScreenError, StopApplication
            import threading
            import queue
        except ImportError:
            print("Error: asciimatics library not installed")
            print("Install it with: uv pip install -r requirements.txt")
            sys.exit(1)

        chat_session = self

        class ChatFrame(Frame):
            def __init__(self, screen):
                super(ChatFrame, self).__init__(
                    screen,
                    screen.height,
                    screen.width,
                    has_border=True,
                    can_scroll=False,
                    title=f"Chat: {chat_session.model['display_name']}"
                )

                self.history = []
                self.is_generating = False
                self.response_queue = queue.Queue()

                # Set theme and customize palette
                self.set_theme("bright")

                # Customize palette for better visibility
                self.palette["title"] = (Screen.COLOUR_CYAN, Screen.A_BOLD, Screen.COLOUR_BLACK)

                # Create layout for chat display
                layout = Layout([100], fill_frame=True)
                self.add_layout(layout)

                # Chat display area (takes most of the screen)
                self._chat_display = TextBox(
                    height=screen.height - 12,
                    label="Chat History:",
                    name="chat_display",
                    as_string=True,
                    line_wrap=True,
                    readonly=True
                )
                self._chat_display.custom_colour = "field"
                layout.add_widget(self._chat_display)

                # Separator label
                separator = Label("─" * (screen.width - 4))
                separator.custom_colour = "label"
                layout.add_widget(separator)

                # Input area (multiline text box)
                self._input = TextBox(
                    height=5,
                    label="Your Message (Ctrl+S to send):",
                    name="input",
                    as_string=True,
                    line_wrap=True
                )
                self._input.custom_colour = "edit_text"
                layout.add_widget(self._input)

                # Buttons
                layout2 = Layout([1, 1, 1, 1])
                self.add_layout(layout2)
                layout2.add_widget(Button("Send (Ctrl+S)", self._on_send), 0)
                layout2.add_widget(Button("Clear (Ctrl+L)", self._on_clear), 1)
                layout2.add_widget(Button("Quit (Ctrl+Q)", self._on_quit), 2)

                # Status bar
                layout3 = Layout([100])
                self.add_layout(layout3)
                self._status = Label("Status: Ready | Ctrl+S: Send | Ctrl+L: Clear | Ctrl+Q: Quit", height=1)
                self._status.custom_colour = "selected_focus_field"
                layout3.add_widget(self._status)

                self.fix()

                # Initialize chat display
                initial_text = f"Connected to: http://localhost:{chat_session.port}/v1\n\n"
                self._chat_display.value = initial_text

            def _update_chat_display(self):
                """Update the chat display with current messages"""
                lines = []
                for msg in self.history:
                    role = msg["role"]
                    content = msg["content"]
                    if role == "user":
                        lines.append(f"You: {content}")
                    else:
                        lines.append(f"Assistant: {content}")
                    lines.append("")  # Empty line between messages

                self._chat_display.value = "\n".join(lines)

            def _on_send(self):
                """Send message"""
                if self.is_generating:
                    self._status.text = "Status: Generating... please wait"
                    return

                message = self._input.value.strip()
                if not message:
                    return

                # Clear input
                self._input.value = ""

                # Add to history
                self.history.append({"role": "user", "content": message})
                self._update_chat_display()

                # Send in background thread
                self.is_generating = True
                self._status.text = "Status: Generating response..."
                threading.Thread(target=self._send_message, daemon=True).start()

            def _send_message(self):
                """Send message to API and stream response"""
                try:
                    stream = chat_session.client.chat.completions.create(
                        model="local-model",
                        messages=self.history,
                        temperature=0.7,
                        max_tokens=-1,
                        stream=True
                    )

                    full_response = ""
                    for chunk in stream:
                        if chunk.choices[0].delta.content is not None:
                            content = chunk.choices[0].delta.content
                            full_response += content

                    # Add to history
                    self.history.append({"role": "assistant", "content": full_response})
                    self.response_queue.put(("success", None))

                except KeyboardInterrupt:
                    # Remove user message on interrupt
                    if self.history and self.history[-1]["role"] == "user":
                        self.history.pop()
                    self.response_queue.put(("interrupted", None))
                except Exception as e:
                    # Remove user message on error
                    if self.history and self.history[-1]["role"] == "user":
                        self.history.pop()
                    self.response_queue.put(("error", str(e)))

            def _on_clear(self):
                """Clear chat history"""
                if not self.is_generating:
                    self.history = []
                    self._chat_display.value = f"Connected to: http://localhost:{chat_session.port}/v1\n\n[History cleared]\n\n"
                    self._status.text = "Status: History cleared"

            def _on_quit(self):
                """Quit application"""
                raise StopApplication("User quit")

            def _update(self, frame_no):
                """Update UI - called every frame"""
                # Check for completed responses
                try:
                    while True:
                        status, error = self.response_queue.get_nowait()
                        self.is_generating = False

                        if status == "success":
                            self._update_chat_display()
                            self._status.text = "Status: Ready"
                        elif status == "interrupted":
                            self._update_chat_display()
                            self._status.text = "Status: Interrupted"
                        elif status == "error":
                            self._update_chat_display()
                            self._status.text = f"Status: Error - {error}"
                except queue.Empty:
                    pass

                super(ChatFrame, self)._update(frame_no)

            def process_event(self, event):
                """Handle keyboard events"""
                from asciimatics.event import KeyboardEvent

                if isinstance(event, KeyboardEvent):
                    if event.key_code == ord('s') - ord('a') + 1:  # Ctrl+S
                        self._on_send()
                        return None
                    elif event.key_code == ord('l') - ord('a') + 1:  # Ctrl+L
                        self._on_clear()
                        return None
                    elif event.key_code == ord('q') - ord('a') + 1:  # Ctrl+Q
                        self._on_quit()
                        return None

                return super(ChatFrame, self).process_event(event)

        def run_app(screen, last_scene):
            screen.play([Scene([ChatFrame(screen)], -1)], stop_on_resize=True, start_scene=last_scene)

        last_scene = None
        while True:
            try:
                Screen.wrapper(run_app, catch_interrupt=True, arguments=[last_scene])
                sys.exit(0)
            except ResizeScreenError as e:
                last_scene = e.scene


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
