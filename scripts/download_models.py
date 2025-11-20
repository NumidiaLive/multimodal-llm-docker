"""
Model downloader script for multimodal LLM.
Downloads and caches required models for text, audio, and video processing.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any
from huggingface_hub import snapshot_download, hf_hub_download
import whisper
from transformers import AutoTokenizer, AutoModel

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.device_utils import get_optimal_device, get_memory_usage
from src.utils.config_loader import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model configurations
MODELS_CONFIG = {
    "text": {
        "microsoft/DialoGPT-small": {
            "type": "generation",
            "size_mb": 100,
            "required_memory_gb": 1
        },
        "distilgpt2": {
            "type": "generation", 
            "size_mb": 80,
            "required_memory_gb": 0.5
        }
    },
    "audio": {
        "whisper": {
            "tiny": {"size_mb": 39, "required_memory_gb": 0.5},
            "base": {"size_mb": 142, "required_memory_gb": 1},
            "small": {"size_mb": 244, "required_memory_gb": 2},
            "medium": {"size_mb": 769, "required_memory_gb": 5},
            "large": {"size_mb": 1550, "required_memory_gb": 10}
        },
        "superb/hubert-base-superb-er": {
            "type": "classification",
            "size_mb": 95,
            "required_memory_gb": 1
        }
    },
    "video": {
        "openai/clip-vit-base-patch32": {
            "type": "multimodal",
            "size_mb": 151,
            "required_memory_gb": 2
        },
        "facebook/detr-resnet-50": {
            "type": "object_detection",
            "size_mb": 167,
            "required_memory_gb": 2
        },
        "Salesforce/blip-image-captioning-base": {
            "type": "captioning",
            "size_mb": 447,
            "required_memory_gb": 3
        }
    }
}

class ModelDownloader:
    """Downloads and manages model files."""
    
    def __init__(self, cache_dir: str = "./models", device: str = "auto"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = get_optimal_device() if device == "auto" else device
        self.memory_info = get_memory_usage()
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Available memory: {self.memory_info}")
    
    def download_all_models(self, config: Any = None):
        """Download all required models based on configuration."""
        config = config or load_config()
        
        logger.info("Starting model download process...")
        
        # Download text models
        self.download_text_models(config.text)
        
        # Download audio models
        self.download_audio_models(config.audio)
        
        # Download video models  
        self.download_video_models(config.video)
        
        logger.info("All models downloaded successfully!")
    
    def download_text_models(self, text_config: Any):
        """Download text processing models."""
        logger.info("Downloading text models...")
        
        # Generation model
        generation_model = getattr(text_config, 'generation_model', 'microsoft/DialoGPT-small')
        self._download_huggingface_model(generation_model, "text generation")
        
        # Classification model (if different)
        classification_model = getattr(text_config, 'classification_model', generation_model)
        if classification_model != generation_model:
            self._download_huggingface_model(classification_model, "text classification")
    
    def download_audio_models(self, audio_config: Any):
        """Download audio processing models."""
        logger.info("Downloading audio models...")
        
        # Whisper model
        whisper_model = getattr(audio_config, 'whisper_model', 'base')
        self._download_whisper_model(whisper_model)
        
        # Audio classification model (optional)
        try:
            self._download_huggingface_model(
                "superb/hubert-base-superb-er", 
                "audio classification"
            )
        except Exception as e:
            logger.warning(f"Could not download audio classification model: {e}")
    
    def download_video_models(self, video_config: Any):
        """Download video processing models."""
        logger.info("Downloading video models...")
        
        # CLIP model
        clip_model = getattr(video_config, 'clip_model', 'openai/clip-vit-base-patch32')
        self._download_huggingface_model(clip_model, "CLIP")
        
        # Object detection model (optional)
        if getattr(video_config, 'enable_object_detection', True):
            try:
                self._download_huggingface_model(
                    "facebook/detr-resnet-50",
                    "object detection"
                )
            except Exception as e:
                logger.warning(f"Could not download object detection model: {e}")
        
        # Image captioning model (optional)
        if getattr(video_config, 'enable_captioning', True):
            try:
                self._download_huggingface_model(
                    "Salesforce/blip-image-captioning-base",
                    "image captioning"
                )
            except Exception as e:
                logger.warning(f"Could not download image captioning model: {e}")
    
    def _download_huggingface_model(self, model_name: str, model_type: str):
        """Download a model from Hugging Face Hub."""
        try:
            logger.info(f"Downloading {model_type} model: {model_name}")
            
            # Check if model already exists
            model_path = self.cache_dir / model_name.replace("/", "_")
            if model_path.exists():
                logger.info(f"Model {model_name} already exists, skipping download")
                return
            
            # Download model
            snapshot_download(
                repo_id=model_name,
                cache_dir=str(self.cache_dir),
                resume_download=True,
                local_files_only=False
            )
            
            # Test loading to ensure it works
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=str(self.cache_dir)
                )
                model = AutoModel.from_pretrained(
                    model_name,
                    cache_dir=str(self.cache_dir),
                    torch_dtype="auto"
                )
                logger.info(f"Successfully verified {model_name}")
                del model, tokenizer  # Free memory
                
            except Exception as e:
                logger.warning(f"Could not verify model {model_name}: {e}")
            
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")
            raise
    
    def _download_whisper_model(self, model_size: str):
        """Download Whisper model."""
        try:
            logger.info(f"Downloading Whisper model: {model_size}")
            
            # Whisper downloads are handled by the whisper package
            model = whisper.load_model(model_size, download_root=str(self.cache_dir))
            logger.info(f"Successfully downloaded Whisper {model_size}")
            
            # Free memory
            del model
            
        except Exception as e:
            logger.error(f"Failed to download Whisper {model_size}: {e}")
            raise
    
    def check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space for model downloads."""
        import shutil
        
        total, used, free = shutil.disk_usage(self.cache_dir)
        
        return {
            "total_gb": total // 1024**3,
            "used_gb": used // 1024**3,
            "free_gb": free // 1024**3,
            "free_percent": (free / total) * 100
        }
    
    def estimate_download_size(self, config: Any = None) -> Dict[str, Any]:
        """Estimate total download size for all models."""
        config = config or load_config()
        
        total_size_mb = 0
        models_to_download = []
        
        # Text models
        generation_model = getattr(config.text, 'generation_model', 'microsoft/DialoGPT-small')
        if generation_model in MODELS_CONFIG["text"]:
            size = MODELS_CONFIG["text"][generation_model]["size_mb"]
            total_size_mb += size
            models_to_download.append({"name": generation_model, "size_mb": size})
        
        # Audio models
        whisper_model = getattr(config.audio, 'whisper_model', 'base')
        if whisper_model in MODELS_CONFIG["audio"]["whisper"]:
            size = MODELS_CONFIG["audio"]["whisper"][whisper_model]["size_mb"]
            total_size_mb += size
            models_to_download.append({"name": f"whisper-{whisper_model}", "size_mb": size})
        
        # Video models
        clip_model = getattr(config.video, 'clip_model', 'openai/clip-vit-base-patch32')
        if clip_model in MODELS_CONFIG["video"]:
            size = MODELS_CONFIG["video"][clip_model]["size_mb"]
            total_size_mb += size
            models_to_download.append({"name": clip_model, "size_mb": size})
        
        return {
            "total_size_mb": total_size_mb,
            "total_size_gb": total_size_mb / 1024,
            "models": models_to_download
        }
    
    def cleanup_old_models(self, keep_current: bool = True):
        """Clean up old or unused model files."""
        logger.info("Cleaning up model cache...")
        
        # This is a placeholder for more sophisticated cleanup
        # In a real implementation, you might:
        # - Remove old versions of models
        # - Clean up temporary files
        # - Remove unused model variants
        
        import shutil
        
        cache_size_before = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, dirnames, filenames in os.walk(self.cache_dir)
            for filename in filenames
        ) // 1024**2  # MB
        
        # Example cleanup: remove .git directories from huggingface cache
        for root, dirs, files in os.walk(self.cache_dir):
            if '.git' in dirs:
                git_path = os.path.join(root, '.git')
                try:
                    shutil.rmtree(git_path)
                    logger.info(f"Removed git directory: {git_path}")
                except Exception as e:
                    logger.warning(f"Could not remove {git_path}: {e}")
        
        cache_size_after = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, dirnames, filenames in os.walk(self.cache_dir)
            for filename in filenames
        ) // 1024**2  # MB
        
        saved_mb = cache_size_before - cache_size_after
        logger.info(f"Cache cleanup completed. Saved {saved_mb} MB")

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Download models for multimodal LLM")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--cache-dir", type=str, default="./models", help="Model cache directory")
    parser.add_argument("--device", type=str, default="auto", help="Target device (auto, cpu, cuda)")
    parser.add_argument("--models", nargs="+", help="Specific models to download")
    parser.add_argument("--check-space", action="store_true", help="Check disk space requirements")
    parser.add_argument("--cleanup", action="store_true", help="Clean up old model files")
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = ModelDownloader(cache_dir=args.cache_dir, device=args.device)
    
    if args.check_space:
        # Check disk space
        space_info = downloader.check_disk_space()
        print(f"Disk space: {space_info['free_gb']} GB free ({space_info['free_percent']:.1f}%)")
        
        # Estimate download size
        config = load_config(args.config)
        size_info = downloader.estimate_download_size(config)
        print(f"Estimated download size: {size_info['total_size_gb']:.1f} GB")
        
        if space_info['free_gb'] < size_info['total_size_gb'] * 2:  # 2x for safety
            logger.warning("Insufficient disk space for model downloads")
            return
    
    if args.cleanup:
        downloader.cleanup_old_models()
        return
    
    # Load configuration
    config = load_config(args.config)
    
    if args.models:
        # Download specific models
        logger.info(f"Downloading specific models: {args.models}")
        for model_name in args.models:
            try:
                if "whisper" in model_name.lower():
                    model_size = model_name.split("-")[-1] if "-" in model_name else "base"
                    downloader._download_whisper_model(model_size)
                else:
                    downloader._download_huggingface_model(model_name, "custom")
            except Exception as e:
                logger.error(f"Failed to download {model_name}: {e}")
    else:
        # Download all models
        downloader.download_all_models(config)
    
    logger.info("Model download process completed!")

if __name__ == "__main__":
    main()