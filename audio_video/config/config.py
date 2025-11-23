"""
Configuration loader for the GLips AVSR project.
This module provides utilities to load and parse configuration files.
"""

import os
import yaml
from typing import Dict, Any, Optional


class Config:
    """
    Configuration class for loading and accessing configuration parameters.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the Config class with a path to the configuration file.
        
        Args:
            config_path (str): Path to the configuration file (.yaml)
        """
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """
        Load the configuration from the YAML file.
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        return config
    
    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key (str): Configuration key, can be nested using dot notation (e.g., 'model.name')
            default (Any, optional): Default value if key is not found
            
        Returns:
            Any: Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get the entire configuration dictionary.
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        return self.config


def load_config(config_path: str) -> Config:
    """
    Load a configuration file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        Config: Configuration object
    """
    return Config(config_path)


if __name__ == "__main__":
    base_path = "/home/aswath/Projects/capstone/multimodel_lipread/audio/configs"
    config = load_config(os.path.join(base_path, "audio_config.yaml"))
    print(config.get_all())