"""
FastAPI main application for multimodal LLM inference.
Supports text, audio, and video processing endpoints.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Union
import torch
import asyncio
import logging
from contextlib import asynccontextmanager

from ..models.text_model import TextModel
from ..models.audio_model import AudioModel  
from ..models.video_model import VideoModel
from ..models.multimodal_model import MultimodalModel
from ..utils.device_utils import get_optimal_device
from ..utils.config_loader import load_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instances
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup models on app startup/shutdown."""
    # Startup
    logger.info("Loading multimodal models...")
    device = get_optimal_device()
    config = load_config()
    
    try:
        models["text"] = TextModel(device=device, config=config.text)
        models["audio"] = AudioModel(device=device, config=config.audio)
        models["video"] = VideoModel(device=device, config=config.video)
        models["multimodal"] = MultimodalModel(device=device, config=config.multimodal)
        
        logger.info(f"Models loaded successfully on device: {device}")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("Cleaning up models...")
    models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Initialize FastAPI app
app = FastAPI(
    title="Multimodal LLM API",
    description="API for text, audio, and video processing with LLMs",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for requests/responses
class TextGenerationRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

class TextGenerationResponse(BaseModel):
    generated_text: str
    tokens_generated: int
    inference_time: float

class AudioTranscriptionResponse(BaseModel):
    transcription: str
    confidence: float
    language: Optional[str]
    inference_time: float

class VideoAnalysisResponse(BaseModel):
    description: str
    objects: List[str]
    actions: List[str]
    inference_time: float

class MultimodalChatRequest(BaseModel):
    text_prompt: str
    context: Optional[str] = None

class MultimodalChatResponse(BaseModel):
    response: str
    modalities_used: List[str]
    inference_time: float

# Dependency to get models
def get_text_model() -> TextModel:
    if "text" not in models:
        raise HTTPException(status_code=503, detail="Text model not loaded")
    return models["text"]

def get_audio_model() -> AudioModel:
    if "audio" not in models:
        raise HTTPException(status_code=503, detail="Audio model not loaded")
    return models["audio"]

def get_video_model() -> VideoModel:
    if "video" not in models:
        raise HTTPException(status_code=503, detail="Video model not loaded")
    return models["video"]

def get_multimodal_model() -> MultimodalModel:
    if "multimodal" not in models:
        raise HTTPException(status_code=503, detail="Multimodal model not loaded")
    return models["multimodal"]

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": list(models.keys()),
        "device": str(get_optimal_device())
    }

# Text processing endpoints
@app.post("/text/generate", response_model=TextGenerationResponse)
async def generate_text(
    request: TextGenerationRequest,
    text_model: TextModel = Depends(get_text_model)
):
    """Generate text using the text model."""
    try:
        import time
        start_time = time.time()
        
        result = await text_model.generate(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        inference_time = time.time() - start_time
        
        return TextGenerationResponse(
            generated_text=result["text"],
            tokens_generated=result["tokens_generated"],
            inference_time=inference_time
        )
    except Exception as e:
        logger.error(f"Text generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/text/classify")
async def classify_text(
    text: str,
    labels: List[str],
    text_model: TextModel = Depends(get_text_model)
):
    """Classify text using the text model."""
    try:
        result = await text_model.classify(text=text, labels=labels)
        return {"predictions": result}
    except Exception as e:
        logger.error(f"Text classification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Audio processing endpoints
@app.post("/audio/transcribe", response_model=AudioTranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    audio_model: AudioModel = Depends(get_audio_model)
):
    """Transcribe audio file to text."""
    try:
        import time
        start_time = time.time()
        
        # Read audio file
        audio_data = await file.read()
        
        result = await audio_model.transcribe(audio_data)
        
        inference_time = time.time() - start_time
        
        return AudioTranscriptionResponse(
            transcription=result["text"],
            confidence=result["confidence"],
            language=result.get("language"),
            inference_time=inference_time
        )
    except Exception as e:
        logger.error(f"Audio transcription failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/audio/generate")
async def generate_audio(
    text: str,
    voice: Optional[str] = "default",
    audio_model: AudioModel = Depends(get_audio_model)
):
    """Generate audio from text."""
    try:
        result = await audio_model.text_to_speech(text=text, voice=voice)
        return {"audio_url": result["url"], "duration": result["duration"]}
    except Exception as e:
        logger.error(f"Audio generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Video processing endpoints
@app.post("/video/analyze", response_model=VideoAnalysisResponse)
async def analyze_video(
    file: UploadFile = File(...),
    video_model: VideoModel = Depends(get_video_model)
):
    """Analyze video content."""
    try:
        import time
        start_time = time.time()
        
        # Read video file
        video_data = await file.read()
        
        result = await video_model.analyze(video_data)
        
        inference_time = time.time() - start_time
        
        return VideoAnalysisResponse(
            description=result["description"],
            objects=result["objects"],
            actions=result["actions"],
            inference_time=inference_time
        )
    except Exception as e:
        logger.error(f"Video analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Multimodal endpoints
@app.post("/multimodal/chat", response_model=MultimodalChatResponse)
async def multimodal_chat(
    request: MultimodalChatRequest,
    text_file: Optional[UploadFile] = File(None),
    audio_file: Optional[UploadFile] = File(None),
    video_file: Optional[UploadFile] = File(None),
    multimodal_model: MultimodalModel = Depends(get_multimodal_model)
):
    """Chat with multimodal inputs."""
    try:
        import time
        start_time = time.time()
        
        # Process uploaded files
        inputs = {"text": request.text_prompt}
        modalities_used = ["text"]
        
        if text_file:
            text_content = await text_file.read()
            inputs["additional_text"] = text_content.decode("utf-8")
        
        if audio_file:
            audio_data = await audio_file.read()
            inputs["audio"] = audio_data
            modalities_used.append("audio")
        
        if video_file:
            video_data = await video_file.read()
            inputs["video"] = video_data
            modalities_used.append("video")
        
        result = await multimodal_model.chat(inputs, context=request.context)
        
        inference_time = time.time() - start_time
        
        return MultimodalChatResponse(
            response=result["response"],
            modalities_used=modalities_used,
            inference_time=inference_time
        )
    except Exception as e:
        logger.error(f"Multimodal chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)