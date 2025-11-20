"""
Device detection and optimization utilities for multimodal LLM.
Automatically detects the best available device and configures optimizations.
"""

import torch
import psutil
import platform
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def get_optimal_device() -> str:
    """Detect and return the optimal device for model inference."""
    
    # Check for CUDA GPU
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory // 1024**3
        
        logger.info(f"CUDA GPU detected: {gpu_name} ({gpu_memory}GB)")
        logger.info(f"Number of GPUs: {gpu_count}")
        
        # Check if GPU has sufficient memory (at least 4GB for basic models)
        if gpu_memory >= 4:
            return "cuda"
        else:
            logger.warning(f"GPU memory ({gpu_memory}GB) may be insufficient for larger models")
            return "cuda"  # Still try to use GPU
    
    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logger.info("Apple MPS detected")
        return "mps"
    
    # Fall back to CPU
    cpu_count = psutil.cpu_count(logical=False)
    cpu_name = platform.processor()
    memory_gb = psutil.virtual_memory().total // 1024**3
    
    logger.info(f"Using CPU: {cpu_name}")
    logger.info(f"CPU cores: {cpu_count}, RAM: {memory_gb}GB")
    
    return "cpu"

def get_device_info() -> Dict[str, Any]:
    """Get comprehensive device information."""
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
    }
    
    # CPU information
    info["cpu"] = {
        "processor": platform.processor(),
        "cores_physical": psutil.cpu_count(logical=False),
        "cores_logical": psutil.cpu_count(logical=True),
        "frequency": psutil.cpu_freq().current if psutil.cpu_freq() else "Unknown",
        "architecture": platform.machine()
    }
    
    # Memory information
    memory = psutil.virtual_memory()
    info["memory"] = {
        "total_gb": memory.total // 1024**3,
        "available_gb": memory.available // 1024**3,
        "used_percent": memory.percent
    }
    
    # GPU information
    if torch.cuda.is_available():
        info["gpu"] = {
            "available": True,
            "count": torch.cuda.device_count(),
            "devices": []
        }
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            device_info = {
                "id": i,
                "name": props.name,
                "memory_total_gb": props.total_memory // 1024**3,
                "memory_allocated_gb": torch.cuda.memory_allocated(i) // 1024**3,
                "compute_capability": f"{props.major}.{props.minor}"
            }
            info["gpu"]["devices"].append(device_info)
    else:
        info["gpu"] = {"available": False}
    
    # MPS information (Apple Silicon)
    if hasattr(torch.backends, 'mps'):
        info["mps"] = {"available": torch.backends.mps.is_available()}
    
    return info

def configure_device_optimizations(device: str, config: Optional[Dict] = None) -> Dict[str, Any]:
    """Configure device-specific optimizations."""
    config = config or {}
    optimizations = {}
    
    if device == "cuda":
        # CUDA optimizations
        optimizations.update({
            "mixed_precision": config.get("mixed_precision", True),
            "cudnn_benchmark": config.get("cudnn_benchmark", True),
            "max_batch_size": config.get("max_batch_size", 8),
            "gradient_checkpointing": config.get("gradient_checkpointing", False),
            "dataloader_workers": config.get("dataloader_workers", 4)
        })
        
        # Enable optimizations
        if optimizations["cudnn_benchmark"]:
            torch.backends.cudnn.benchmark = True
            
        logger.info("CUDA optimizations configured")
        
    elif device == "cpu":
        # CPU optimizations
        cpu_cores = psutil.cpu_count(logical=False)
        
        optimizations.update({
            "num_threads": config.get("num_threads", min(cpu_cores, 4)),
            "max_batch_size": config.get("max_batch_size", 2),
            "use_mkldnn": config.get("use_mkldnn", True),
            "dataloader_workers": config.get("dataloader_workers", 2),
            "gradient_checkpointing": config.get("gradient_checkpointing", True)
        })
        
        # Set thread limits
        torch.set_num_threads(optimizations["num_threads"])
        torch.set_num_interop_threads(optimizations["num_threads"])
        
        logger.info(f"CPU optimizations configured: {optimizations['num_threads']} threads")
        
    elif device == "mps":
        # Apple MPS optimizations
        optimizations.update({
            "mixed_precision": config.get("mixed_precision", False),  # MPS may not support all precision types
            "max_batch_size": config.get("max_batch_size", 4),
            "dataloader_workers": config.get("dataloader_workers", 2)
        })
        
        logger.info("MPS optimizations configured")
    
    return optimizations

def get_memory_usage() -> Dict[str, Any]:
    """Get current memory usage information."""
    usage = {}
    
    # System memory
    memory = psutil.virtual_memory()
    usage["system"] = {
        "total_gb": memory.total // 1024**3,
        "used_gb": memory.used // 1024**3,
        "available_gb": memory.available // 1024**3,
        "percent": memory.percent
    }
    
    # GPU memory
    if torch.cuda.is_available():
        usage["gpu"] = {}
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) // 1024**3
            cached = torch.cuda.memory_reserved(i) // 1024**3
            total = torch.cuda.get_device_properties(i).total_memory // 1024**3
            
            usage["gpu"][f"device_{i}"] = {
                "allocated_gb": allocated,
                "cached_gb": cached,
                "total_gb": total,
                "percent": (allocated / total) * 100 if total > 0 else 0
            }
    
    return usage

def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("GPU memory cleared")

def estimate_model_memory(model_name: str, precision: str = "fp32") -> float:
    """Estimate memory requirements for a model (in GB)."""
    
    # Model size estimates (in millions of parameters)
    model_sizes = {
        "gpt2": 117,
        "distilgpt2": 82,
        "microsoft/DialoGPT-small": 117,
        "microsoft/DialoGPT-medium": 345,
        "microsoft/DialoGPT-large": 762,
        "openai/clip-vit-base-patch32": 151,
        "facebook/detr-resnet-50": 41,
        "Salesforce/blip-image-captioning-base": 129,
        "superb/hubert-base-superb-er": 95
    }
    
    # Get model size
    params = model_sizes.get(model_name, 100)  # Default to 100M params
    
    # Bytes per parameter based on precision
    bytes_per_param = {
        "fp32": 4,
        "fp16": 2,
        "int8": 1,
        "int4": 0.5
    }
    
    bytes_pp = bytes_per_param.get(precision, 4)
    
    # Estimate memory (parameters + activations + overhead)
    memory_gb = (params * 1e6 * bytes_pp) / 1024**3
    memory_gb *= 1.5  # Add 50% overhead for activations and buffers
    
    return memory_gb

def check_hardware_compatibility(target_device: str) -> Dict[str, Any]:
    """Check if hardware is compatible with target device."""
    compatibility = {
        "compatible": False,
        "warnings": [],
        "recommendations": []
    }
    
    if target_device == "cuda":
        if not torch.cuda.is_available():
            compatibility["warnings"].append("CUDA not available")
            compatibility["recommendations"].append("Install CUDA-enabled PyTorch")
        else:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory // 1024**3
            if gpu_memory < 4:
                compatibility["warnings"].append(f"Limited GPU memory: {gpu_memory}GB")
                compatibility["recommendations"].append("Consider using smaller models or CPU inference")
            else:
                compatibility["compatible"] = True
                
    elif target_device == "cpu":
        cpu_cores = psutil.cpu_count(logical=False)
        memory_gb = psutil.virtual_memory().total // 1024**3
        
        if memory_gb < 8:
            compatibility["warnings"].append(f"Limited RAM: {memory_gb}GB")
            compatibility["recommendations"].append("Consider using quantized models")
        if cpu_cores < 4:
            compatibility["warnings"].append(f"Limited CPU cores: {cpu_cores}")
            compatibility["recommendations"].append("Expect slower inference times")
        
        compatibility["compatible"] = True
        
    elif target_device == "mps":
        if not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available():
            compatibility["warnings"].append("MPS not available")
            compatibility["recommendations"].append("Use CPU device instead")
        else:
            compatibility["compatible"] = True
    
    return compatibility