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

    def download(self, model: Dict) -> bool:
        """Download a single model"""
        print(f"Downloading {model['display_name']}...")
        self.config.models_dir.mkdir(parents=True, exist_ok=True)

        model_path = self.config.models_dir / model["model_file"]
        repo = model["download"]["repo"]
        file = model["download"]["file"]

        # Try huggingface-cli first
        if self._has_command("huggingface-cli"):
            return self._download_with_hf_cli(repo, file, model_path)
        else:
            print("huggingface-cli not found. Downloading with wget...")
            return self._download_with_wget(repo, file, model_path)

    def _download_with_hf_cli(self, repo: str, file: str, dest: Path) -> bool:
        """Download using huggingface-cli"""
        try:
            subprocess.run([
                "huggingface-cli", "download", repo, file,
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
