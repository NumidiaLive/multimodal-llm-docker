# Multimodal LLM Docker Project

A containerized multimodal Large Language Model with text, audio, and video processing capabilities, optimized for deployment across Intel NUC and AMD gaming laptop environments.

## 🏗️ Architecture Overview

```
multimodal-llm/
├── docker/                    # Docker configurations
├── src/                      # Source code
│   ├── text/                # Text processing models
│   ├── audio/               # Audio processing models
│   ├── video/               # Video processing models
│   ├── multimodal/          # Fusion and multimodal models
│   ├── api/                 # FastAPI endpoints
│   └── utils/               # Shared utilities
├── data/                    # Training and test data
├── models/                  # Pretrained and fine-tuned models
├── configs/                 # Configuration files
└── scripts/                 # Training and deployment scripts
```

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- NVIDIA Docker (for GPU support)
- At least 16GB RAM (32GB recommended)

### Launch Development Environment
```bash
# Clone and setup
git clone <your-repo>
cd multimodal-llm

# Start all services
docker-compose up -d

# Access Jupyter Lab
# http://localhost:8888

# Access API endpoints
# http://localhost:8000/docs
```

## 🔧 Hardware Configurations

### Intel NUC6i7KYK (Skull Canyon)
- **CPU**: i7-6770HQ (4 cores, 8 threads)
- **Graphics**: Intel Iris Pro Graphics 580
- **RAM**: 32GB DDR4
- **Optimizations**: CPU-focused inference, efficient model quantization

### Gaming TUF Laptop
- **CPU**: AMD Ryzen 9 4900H (8 cores, 16 threads)  
- **Graphics**: Radeon Graphics
- **RAM**: 32GB
- **Optimizations**: Multi-core training, larger batch sizes

## 📦 Docker Services

| Service | Purpose | Port | Hardware Target |
|---------|---------|------|-----------------|
| `jupyter` | Development environment | 8888 | Both |
| `api` | FastAPI inference server | 8000 | Both |
| `training` | Model training service | - | High-spec laptop |
| `redis` | Caching and queue | 6379 | Both |
| `postgres` | Data storage | 5432 | Both |

## 🤖 Supported Models

### Text Processing
- **GPT-2/GPT-Neo**: Text generation and completion
- **BERT/RoBERTa**: Text classification and embedding
- **T5**: Text-to-text transfer tasks

### Audio Processing
- **Whisper**: Speech-to-text transcription
- **Wav2Vec2**: Audio feature extraction
- **MusicGen**: Audio generation

### Video Processing
- **CLIP**: Image-text understanding
- **VideoMAE**: Video understanding
- **I3D**: Action recognition

### Multimodal Fusion
- **BLIP-2**: Image-text reasoning
- **VideoChatGPT**: Video-text conversation
- **Custom fusion models**: Cross-modal attention

## 🛠️ Development Workflow

1. **Environment Setup**: Use Docker Compose for consistent development
2. **Model Development**: Jupyter notebooks for experimentation
3. **Training**: Distributed training across available hardware
4. **Inference**: RESTful API endpoints for each modality
5. **Deployment**: Container orchestration for production

## 📊 Performance Targets

| Task | Intel NUC | Gaming Laptop | Notes |
|------|-----------|---------------|-------|
| Text Generation | 10-15 tokens/sec | 25-35 tokens/sec | Small models |
| Speech-to-Text | Real-time | Real-time+ | Whisper base |
| Video Analysis | 2-5 FPS | 10-15 FPS | CLIP features |

## 🔗 API Endpoints

- `POST /text/generate` - Text generation
- `POST /text/classify` - Text classification  
- `POST /audio/transcribe` - Speech-to-text
- `POST /audio/generate` - Audio synthesis
- `POST /video/analyze` - Video understanding
- `POST /multimodal/chat` - Cross-modal conversation

## 📈 Monitoring & Scaling

- **Health Checks**: Container health monitoring
- **Resource Monitoring**: CPU, memory, GPU utilization
- **Load Balancing**: Multi-instance deployment
- **Auto-scaling**: Based on request queue depth

## 🔐 Security Features

- **API Authentication**: JWT token-based auth
- **Input Validation**: Sanitized multimodal inputs
- **Container Isolation**: Sandboxed execution environment
- **Resource Limits**: Prevent resource exhaustion

## 📚 Getting Started

1. **Setup Environment**: Follow Docker installation guide
2. **Choose Configuration**: Select hardware-optimized compose file
3. **Download Models**: Pre-download required model weights
4. **Start Services**: Launch with docker-compose
5. **Run Examples**: Try provided Jupyter notebooks

For detailed setup instructions, see the [Installation Guide](docs/installation.md).