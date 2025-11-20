"""
Configuration loader and management for multimodal LLM.
Handles loading and validation of configuration files.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Config:
    """Configuration container with attribute-style access."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
        
        # Convert nested dictionaries to Config objects
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        return getattr(self, key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dictionary."""
        result = {}
        for key, value in self._config.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file or return default configuration.
    
    Args:
        config_path: Path to configuration file. If None, uses default config.
    
    Returns:
        Config object with loaded configuration
    """
    if config_path and os.path.exists(config_path):
        return _load_config_file(config_path)
    else:
        # Try to find config file in common locations
        possible_paths = [
            "configs/config.yaml",
            "configs/config.yml",
            "configs/config.json",
            "config.yaml",
            "config.yml", 
            "config.json"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found configuration file: {path}")
                return _load_config_file(path)
        
        logger.info("No configuration file found, using default configuration")
        return Config(_get_default_config())

def _load_config_file(config_path: str) -> Config:
    """Load configuration from a specific file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.endswith('.json'):
                config_dict = json.load(f)
            elif config_path.endswith(('.yaml', '.yml')):
                config_dict = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path}")
        
        logger.info(f"Configuration loaded from {config_path}")
        return Config(config_dict)
        
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        logger.info("Falling back to default configuration")
        return Config(_get_default_config())

def _get_default_config() -> Dict[str, Any]:
    """Get default configuration for multimodal LLM."""
    
    config = {
        "environment": "development",
        
        # Device configuration
        "device": {
            "auto_detect": True,
            "preferred": "auto",  # auto, cuda, cpu, mps
            "optimize_for_device": True
        },
        
        # Text model configuration
        "text": {
            "generation_model": "microsoft/DialoGPT-small",
            "classification_model": "microsoft/DialoGPT-small",
            "max_length": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "quantization": {
                "enabled": False,
                "bits": 8,
                "cpu_offload": True
            }
        },
        
        # Audio model configuration
        "audio": {
            "whisper_model": "base",  # tiny, base, small, medium, large
            "sample_rate": 16000,
            "language": None,  # Auto-detect if None
            "task": "transcribe",  # transcribe or translate
            "enable_classification": True
        },
        
        # Video model configuration
        "video": {
            "clip_model": "openai/clip-vit-base-patch32",
            "object_detection_model": "facebook/detr-resnet-50",
            "captioning_model": "Salesforce/blip-image-captioning-base",
            "max_frames": 16,
            "frame_sampling": "uniform",  # uniform, random, keyframe
            "enable_object_detection": True,
            "enable_captioning": True
        },
        
        # Multimodal configuration
        "multimodal": {
            "fusion_strategy": "concatenate",  # concatenate, attention, cross_modal
            "max_context_length": 512,
            "enable_cross_modal_similarity": True,
            "similarity_threshold": 0.7
        },
        
        # API configuration
        "api": {
            "host": "0.0.0.0",
            "port": 8000,
            "workers": 1,
            "max_request_size": "10MB",
            "cors_origins": ["*"],
            "enable_auth": False,
            "rate_limiting": {
                "enabled": False,
                "requests_per_minute": 60
            }
        },
        
        # Training configuration
        "training": {
            "batch_size": 8,
            "learning_rate": 1e-5,
            "epochs": 10,
            "warmup_steps": 100,
            "save_strategy": "steps",
            "save_steps": 500,
            "evaluation_strategy": "steps",
            "eval_steps": 500,
            "logging_steps": 100,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "dataloader_num_workers": 2,
            "mixed_precision": False,
            "deepspeed": {
                "enabled": False,
                "config_path": "configs/deepspeed.json"
            }
        },
        
        # Data configuration
        "data": {
            "cache_dir": "./data/cache",
            "dataset_dir": "./data/datasets",
            "preprocessing": {
                "max_text_length": 512,
                "max_audio_duration": 30,  # seconds
                "max_video_duration": 60,  # seconds
                "audio_preprocessing": {
                    "normalize": True,
                    "remove_silence": False,
                    "noise_reduction": False
                },
                "video_preprocessing": {
                    "resize": [224, 224],
                    "normalize": True,
                    "frame_extraction_fps": 1
                }
            }
        },
        
        # Optimization configuration
        "optimization": {
            "model_optimization": {
                "quantization": False,
                "pruning": False,
                "distillation": False
            },
            "inference_optimization": {
                "batch_inference": True,
                "caching": True,
                "async_processing": True
            },
            "memory_optimization": {
                "gradient_checkpointing": False,
                "cpu_offload": False,
                "model_sharding": False
            }
        },
        
        # Logging configuration
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": None,  # Set to path for file logging
            "max_file_size": "10MB",
            "backup_count": 5
        },
        
        # Monitoring configuration
        "monitoring": {
            "enable_metrics": False,
            "metrics_port": 9090,
            "health_check_interval": 30,
            "log_system_metrics": True
        },
        
        # Security configuration
        "security": {
            "enable_input_validation": True,
            "max_file_size": "100MB",
            "allowed_file_types": {
                "audio": ["wav", "mp3", "flac", "ogg"],
                "video": ["mp4", "avi", "mov", "mkv"],
                "text": ["txt", "json"]
            },
            "rate_limiting": {
                "enabled": False,
                "max_requests_per_minute": 60,
                "max_requests_per_hour": 1000
            }
        }
    }
    
    return config

def save_config(config: Config, config_path: str):
    """Save configuration to file."""
    try:
        # Ensure directory exists
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = config.to_dict()
        
        with open(config_path, 'w', encoding='utf-8') as f:
            if config_path.endswith('.json'):
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            elif config_path.endswith(('.yaml', '.yml')):
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
            else:
                raise ValueError(f"Unsupported config file format: {config_path}")
        
        logger.info(f"Configuration saved to {config_path}")
        
    except Exception as e:
        logger.error(f"Failed to save configuration to {config_path}: {e}")
        raise

def validate_config(config: Config) -> bool:
    """Validate configuration for required fields and valid values."""
    try:
        # Check required sections
        required_sections = ["device", "text", "audio", "video", "multimodal"]
        for section in required_sections:
            if not hasattr(config, section):
                logger.error(f"Missing required configuration section: {section}")
                return False
        
        # Validate device configuration
        if hasattr(config.device, 'preferred'):
            valid_devices = ["auto", "cuda", "cpu", "mps"]
            if config.device.preferred not in valid_devices:
                logger.error(f"Invalid device preference: {config.device.preferred}")
                return False
        
        # Validate model names (basic check)
        if hasattr(config.text, 'generation_model'):
            if not isinstance(config.text.generation_model, str):
                logger.error("Text generation model must be a string")
                return False
        
        # Validate numeric ranges
        if hasattr(config.text, 'temperature'):
            if not 0 < config.text.temperature <= 2:
                logger.error("Text temperature must be between 0 and 2")
                return False
        
        if hasattr(config.audio, 'sample_rate'):
            if config.audio.sample_rate not in [8000, 16000, 22050, 44100, 48000]:
                logger.warning(f"Unusual sample rate: {config.audio.sample_rate}")
        
        logger.info("Configuration validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configuration dictionaries, with override taking precedence."""
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged

def get_config_template() -> str:
    """Get a YAML template for configuration file."""
    template = """
# Multimodal LLM Configuration

environment: development  # development, production

device:
  auto_detect: true
  preferred: auto  # auto, cuda, cpu, mps
  optimize_for_device: true

text:
  generation_model: "microsoft/DialoGPT-small"
  classification_model: "microsoft/DialoGPT-small"
  max_length: 512
  temperature: 0.7
  top_p: 0.9

audio:
  whisper_model: "base"  # tiny, base, small, medium, large
  sample_rate: 16000
  language: null  # Auto-detect

video:
  clip_model: "openai/clip-vit-base-patch32"
  max_frames: 16
  frame_sampling: "uniform"

multimodal:
  fusion_strategy: "concatenate"
  max_context_length: 512

api:
  host: "0.0.0.0"
  port: 8000
  workers: 1

# Add more sections as needed...
"""
    return template.strip()