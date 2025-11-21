# Quick Reference - Gaming Laptop

## Hardware
- **GPU**: NVIDIA RTX 2060 (6GB VRAM)
- **CUDA**: 12.7
- **Driver**: 565.90

## Installed Software
```
✓ Git 2.52.0
✓ Python 3.11.9
✓ Docker Desktop 28.5.2
✓ WSL2 2.6.1
✓ Windows Terminal
```

## Essential Commands

### Refresh Environment
```powershell
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
```

### Validate Setup
```powershell
python scripts/validate_setup.py
```

### First Run
```powershell
.\setup-gaming.ps1
```

### Start Services
```powershell
# All services
docker-compose -f docker-compose.gaming.yml up -d

# Just Jupyter
docker-compose -f docker-compose.gaming.yml up -d jupyter redis postgres

# View logs
docker-compose -f docker-compose.gaming.yml logs -f
```

### Stop Services
```powershell
docker-compose -f docker-compose.gaming.yml down
```

### Monitor GPU
```powershell
nvidia-smi -l 1
```

## URLs
- Jupyter: http://localhost:8888 (token: `gaming-llm`)
- API Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

## Important Files
- `GAMING-LAPTOP-SETUP.md` - Full setup guide
- `INSTALLATION-SUMMARY.md` - What was installed
- `setup-gaming.ps1` - Automated setup
- `scripts/validate_setup.py` - Validation
- `configs/config.gaming.yaml` - Model config
- `docker-compose.gaming.yml` - Services

## Memory Optimizations
- Batch size: 4 (reduced from 8)
- Max frames: 12 (reduced from 16)
- Gradient checkpointing: Enabled
- Mixed precision: FP16

## Troubleshooting
```powershell
# Docker not found
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# GPU test
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Restart Docker
Restart-Service docker
```

## Next Steps
1. Restart computer
2. Start Docker Desktop
3. Run `python scripts/validate_setup.py`
4. Run `.\setup-gaming.ps1`
5. Open http://localhost:8888
