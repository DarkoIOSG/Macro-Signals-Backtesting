import yaml
from pathlib import Path


def load_config(config_path: str) -> dict:
    """
    Load YAML configuration file.
    
    Parameters:
    -----------
    config_path : str, path to YAML config file
    
    Returns:
    --------
    dict with configuration values
    """
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def load_data_config() -> dict:
    return load_config("config/data_config.yaml")


def load_model_config() -> dict:
    return load_config("config/model_config.yaml")


def load_feature_config() -> dict:
    return load_config("config/feature_config.yaml")