# Gaming Laptop Setup Guide
## RTX 2060 (6GB VRAM) Configuration

This guide is specifically optimized for your gaming laptop with:
- **GPU**: NVIDIA GeForce RTX 2060 (6GB VRAM)
- **CUDA**: Version 12.7
- **Driver**: 565.90
- **CPU**: AMD Ryzen 9 4900H (8 cores, 16 threads)

## ✅ Installed Software

All required software has been installed using winget:

| Software | Version | Status |
|----------|---------|--------|
| **Git** | 2.52.0 | ✅ Installed |
| **Python** | 3.11.9 | ✅ Installed |
| **Docker Desktop** | 28.5.2 | ✅ Installed |
| **WSL2** | 2.6.1 | ✅ Installed |
| **Windows Terminal** | Latest | ✅ Pre-installed |
| **NVIDIA Drivers** | 565.90 | ✅ Pre-installed |

## 🚀 Quick Start

### 1. First-Time Setup

Run the automated setup script:

```powershell
.\setup-gaming.ps1
```

This script will:
- Verify all prerequisites
- Install Python dependencies
- Build Docker images optimized for RTX 2060
- Create required directories
- Prepare your environment

### 2. Start Docker Desktop

**IMPORTANT**: Before running Docker containers, you must:

1. Open **Docker Desktop** from the Start menu
2. Wait for it to fully start (green icon in system tray)
3. Enable GPU support:
   - Settings → General → "Use the WSL 2 based engine" ✓
   - Settings → Resources → WSL Integration → Enable integration

### 3. Launch Services

Start all services with GPU support:

```powershell
docker-compose -f docker-compose.gaming.yml up -d
```

To view logs:
```powershell
docker-compose -f docker-compose.gaming.yml logs -f
```

### 4. Access Services

| Service | URL | Token/Credentials |
|---------|-----|-------------------|
| **Jupyter Lab** | http://localhost:8888 | Token: `gaming-llm` |
| **API Docs** | http://localhost:8000/docs | N/A |
| **API Health** | http://localhost:8000/health | N/A |

## 🎮 RTX 2060 Optimizations

Your configuration has been optimized for 6GB VRAM:

### Model Selection
- **Text**: `distilgpt2` (smaller, faster, good quality)
- **Audio**: Whisper `base` (optimal speed/quality balance)
- **Video**: CLIP `vit-base-patch32` + DETR + BLIP

### Training Settings
- **Batch Size**: 4 (fits in 6GB VRAM)
- **Mixed Precision**: Enabled (FP16 for 2x speed)
- **Gradient Checkpointing**: Enabled (saves memory)
- **Max Video Frames**: 12 (reduced from 16)

### Expected Performance
- **Text Generation**: 30-40 tokens/sec
- **Speech-to-Text**: 3-4x realtime
- **Image Analysis**: 15-20 images/sec
- **Video Analysis**: 8-12 FPS

## 📊 Monitor GPU Usage

Real-time GPU monitoring:
```powershell
nvidia-smi -l 1
```

Check GPU memory usage:
```powershell
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

## 🔧 Common Commands

### Docker Management

```powershell
# Start services
docker-compose -f docker-compose.gaming.yml up -d

# Stop services
docker-compose -f docker-compose.gaming.yml down

# Restart a specific service
docker-compose -f docker-compose.gaming.yml restart api

# View logs
docker-compose -f docker-compose.gaming.yml logs -f jupyter

# Check status
docker-compose -f docker-compose.gaming.yml ps
```

### Python Environment

```powershell
# Install additional packages
python -m pip install package-name

# Check PyTorch CUDA
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# Test GPU in Python
python -c "import torch; x = torch.rand(5, 3).cuda(); print(x)"
```

### Development Workflow

```powershell
# 1. Start Jupyter for experimentation
docker-compose -f docker-compose.gaming.yml up -d jupyter

# 2. Open Jupyter and run quickstart
start http://localhost:8888

# 3. Test API endpoints
docker-compose -f docker-compose.gaming.yml up -d api
start http://localhost:8000/docs

# 4. Train models
docker-compose -f docker-compose.gaming.yml run --rm training python scripts/train.py
```

## 🎯 Next Steps

### 1. Verify Setup
```powershell
# Test Python with GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Test Docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### 2. Run Quickstart Notebook
1. Open Jupyter: http://localhost:8888 (token: `gaming-llm`)
2. Navigate to `notebooks/quickstart.ipynb`
3. Run all cells to verify setup

### 3. Download Models
```powershell
# Download required models
docker-compose -f docker-compose.gaming.yml run --rm training python scripts/download_models.py
```

### 4. Test API
```powershell
# Health check
curl http://localhost:8000/health

# Test text generation
curl -X POST http://localhost:8000/text/generate `
  -H "Content-Type: application/json" `
  -d '{"prompt": "Hello, how are you?", "max_length": 50}'
```

## ⚠️ Important Notes

### Memory Management
- **6GB VRAM is limited**: Don't run all services simultaneously if memory constrained
- Use `nvidia-smi` to monitor GPU memory
- Close Jupyter notebooks when not in use
- Consider running models sequentially rather than all at once

### Docker Desktop Settings
- Allocate at least 8GB RAM to Docker Desktop
- Allocate at least 4 CPU cores
- Enable WSL2 integration
- GPU support requires WSL2 backend

### Windows-Specific
- Use **PowerShell** (not CMD) for best compatibility
- Run setup scripts from the project root directory
- Docker volumes may be slower on Windows; use WSL2 for better performance

## 🐛 Troubleshooting

### Docker GPU Not Working
```powershell
# Verify NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# If fails, restart Docker Desktop and ensure WSL2 backend is enabled
```

### Out of Memory Errors
```powershell
# Reduce batch size in configs/config.gaming.yaml
# Change: batch_size: 4 → batch_size: 2

# Or use CPU for specific tasks
# Edit docker-compose.gaming.yml and remove GPU reservation
```

### Python Import Errors
```powershell
# Rebuild containers
docker-compose -f docker-compose.gaming.yml build --no-cache

# Or install in host environment
python -m pip install -r requirements.jupyter.txt
```

### WSL2 Issues
```powershell
# Update WSL2
wsl --update

# Set WSL2 as default
wsl --set-default-version 2

# Restart WSL
wsl --shutdown
```

## 📚 Additional Resources

- **Project README**: [README.md](README.md)
- **Deployment Guide**: [DEPLOYMENT.md](DEPLOYMENT.md)
- **Docker Compose**: [docker-compose.gaming.yml](docker-compose.gaming.yml)
- **Configuration**: [configs/config.gaming.yaml](configs/config.gaming.yaml)

## 🔐 Security Notes

- Default tokens are for development only
- Change tokens/passwords before any public deployment
- Keep Docker Desktop and NVIDIA drivers updated
- Use VPN when accessing services remotely

---

**Your gaming laptop is now ready for multimodal LLM development!** 🎉

For questions or issues, check the troubleshooting section or open an issue in the repository.
