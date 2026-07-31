"""
config_loader.py — Configuration management for CRF experiments
================================================================
Loads and manages YAML configuration files with OmegaConf for type safety.
"""

import os
import random
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import yaml
from omegaconf import OmegaConf, DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


def load_config(
    config_path: str = str(DEFAULT_CONFIG_PATH),
    overrides: Optional[Dict[str, Any]] = None,
) -> DictConfig:
    """
    Load configuration from YAML file with optional overrides.

    Args:
        config_path: Path to YAML configuration file
        overrides: Dictionary of parameter overrides

    Returns:
        OmegaConf DictConfig with merged configuration
    """
    # Load base config
    with open(config_path, "r") as f:
        config = OmegaConf.create(yaml.safe_load(f))

    # Apply overrides
    if overrides:
        config = OmegaConf.merge(config, overrides)

    return config


def save_config(config: DictConfig, save_path: str) -> None:
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        yaml.dump(OmegaConf.to_container(config), f, default_flow_style=False)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_device(config: DictConfig) -> torch.device:
    """
    Get the appropriate device based on configuration and availability.

    Args:
        config: Configuration object

    Returns:
        torch.device object
    """
    device_str = config.hardware.device

    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    if device_str == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, using CPU")
        device_str = "cpu"

    device = torch.device(device_str)

    # Set specific GPU if requested
    if device_str == "cuda" and config.hardware.gpu_ids:
        torch.cuda.set_device(config.hardware.gpu_ids[0])

    return device


def log_system_info(config: DictConfig) -> Dict[str, str]:
    """
    Log system information for reproducibility.

    Args:
        config: Configuration object

    Returns:
        Dictionary of system information
    """
    info = {
        "python_version": os.sys.version,
        "pytorch_version": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "device": str(get_device(config)),
        "gpu_count": (
            str(torch.cuda.device_count()) if torch.cuda.is_available() else "0"
        ),
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        ),
    }

    return info


def create_experiment_name(config: DictConfig) -> str:
    """
    Generate a descriptive experiment name from configuration.

    Args:
        config: Configuration object

    Returns:
        Experiment name string
    """
    if config.experiment.run_name:
        return config.experiment.run_name

    model_type = config.model.type
    dataset = config.data.dataset
    d_model = config.model.d_model
    n_cells = config.model.n_init_cells
    n_steps = config.model.n_crf_steps

    name = f"{model_type}_{dataset}_d{d_model}_N{n_cells}_S{n_steps}"

    if config.model.ablation != "full":
        name += f"_{config.model.ablation}"

    return name


class ConfigManager:
    """
    Manages experiment configuration with validation and defaults.
    """

    def __init__(self, config_path: str = str(DEFAULT_CONFIG_PATH)):
        self.config_path = config_path
        self.config = load_config(config_path)
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        # Validate model parameters
        assert self.config.model.d_model > 0, "d_model must be positive"
        assert self.config.model.n_init_cells > 0, "n_init_cells must be positive"
        assert (
            self.config.model.max_cells >= self.config.model.n_init_cells
        ), "max_cells must be >= n_init_cells"

        # Validate training parameters
        assert 0 < self.config.training.lr < 1, "Learning rate must be in (0, 1)"
        assert self.config.training.batch_size > 0, "Batch size must be positive"

        # Validate ablation name
        valid_ablations = [
            "full",
            "no_split_death",
            "no_merge",
            "no_routing",
            "no_energy",
            "no_messaging",
            "no_spatial",
        ]
        assert (
            self.config.model.ablation in valid_ablations
        ), f"Invalid ablation: {self.config.model.ablation}"

    def update(self, **kwargs) -> None:
        """Update configuration with new values."""
        self.config = OmegaConf.merge(self.config, kwargs)
        self._validate_config()

    def save(self, save_path: str) -> None:
        """Save current configuration to file."""
        save_config(self.config, save_path)

    def get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration as dictionary."""
        return OmegaConf.to_container(self.config.model)

    def get_training_config(self) -> Dict[str, Any]:
        """Get training-specific configuration as dictionary."""
        return OmegaConf.to_container(self.config.training)

    def setup_experiment(self) -> Dict[str, Any]:
        """
        Set up experiment environment: seeds, device, logging directories.

        Returns:
            Dictionary with experiment setup information
        """
        # Set seeds
        set_seed(self.config.experiment.seed, self.config.experiment.deterministic)

        # Get device
        device = get_device(self.config)

        # Create directories
        for dir_key in ["log_dir", "checkpoint_dir", "results_dir"]:
            dir_path = self.config.experiment[dir_key]
            os.makedirs(dir_path, exist_ok=True)

        # Generate experiment name
        exp_name = create_experiment_name(self.config)

        # Log system info
        system_info = log_system_info(self.config)

        return {
            "device": device,
            "experiment_name": exp_name,
            "system_info": system_info,
            "config": self.config,
        }


if __name__ == "__main__":
    # Test configuration loading
    config = load_config(str(DEFAULT_CONFIG_PATH))
    print("Default config loaded successfully")
    print(f"Model type: {config.model.type}")
    print(f"Dataset: {config.data.dataset}")

    # Test ConfigManager
    manager = ConfigManager(str(DEFAULT_CONFIG_PATH))
    setup = manager.setup_experiment()
    print(f"\nExperiment setup:")
    print(f"Device: {setup['device']}")
    print(f"Experiment name: {setup['experiment_name']}")
    print(f"System info: {setup['system_info']}")
