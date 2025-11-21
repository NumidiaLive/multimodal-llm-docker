# Multimodal LLM Docker - Gaming Laptop Setup Script
# RTX 2060 6GB VRAM Configuration

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "   Multimodal LLM Docker - Gaming Laptop Setup   " -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

# Refresh environment variables to pick up newly installed software
Write-Host "Refreshing environment variables..." -ForegroundColor Yellow
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# Check prerequisites
Write-Host "Checking prerequisites..." -ForegroundColor Yellow

# Docker
Write-Host "  [*] Docker: " -NoNewline
try {
    $dockerVersion = docker --version
    Write-Host $dockerVersion -ForegroundColor Green
} catch {
    Write-Host "NOT INSTALLED" -ForegroundColor Red
    Write-Host "    Please install Docker Desktop and restart this script" -ForegroundColor Red
    exit 1
}

# Python
Write-Host "  [*] Python: " -NoNewline
try {
    $pythonVersion = python --version
    Write-Host $pythonVersion -ForegroundColor Green
} catch {
    Write-Host "NOT INSTALLED" -ForegroundColor Red
    Write-Host "    Please install Python 3.11 and restart this script" -ForegroundColor Red
    exit 1
}

# Git
Write-Host "  [*] Git: " -NoNewline
try {
    $gitVersion = git --version
    Write-Host $gitVersion -ForegroundColor Green
} catch {
    Write-Host "NOT INSTALLED" -ForegroundColor Red
    exit 1
}

# NVIDIA GPU
Write-Host "  [*] NVIDIA GPU: " -NoNewline
try {
    $nvidiaVersion = nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    Write-Host $nvidiaVersion -ForegroundColor Green
} catch {
    Write-Host "NOT DETECTED" -ForegroundColor Yellow
    Write-Host "    GPU acceleration will not be available" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Step 1: Installing Python dependencies" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install basic requirements first (from PyPI)
Write-Host "Installing basic dependencies..." -ForegroundColor Yellow
python -m pip install pyyaml requests numpy

# Install PyTorch with CUDA support
Write-Host "Installing PyTorch with CUDA 12.1 support..." -ForegroundColor Yellow
Write-Host "This may take a few minutes..." -ForegroundColor Gray
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

if ($LASTEXITCODE -eq 0) {
    Write-Host "  [OK] Python dependencies installed" -ForegroundColor Green
} else {
    Write-Host "  [!] Some dependencies may have failed to install" -ForegroundColor Yellow
    Write-Host "      You can continue, but some features may not work" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Step 2: Verifying Docker Desktop" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# Check if Docker Desktop is running
Write-Host "Checking Docker Desktop status..." -ForegroundColor Yellow
try {
    docker ps | Out-Null
    Write-Host "  [OK] Docker Desktop is running" -ForegroundColor Green
} catch {
    Write-Host "  [X] Docker Desktop is NOT running" -ForegroundColor Red
    Write-Host ""
    Write-Host "Docker Desktop must be running to build images." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To fix this:" -ForegroundColor Cyan
    Write-Host "  1. Open Docker Desktop from the Start menu" -ForegroundColor White
    Write-Host "  2. Wait for the whale icon in the system tray to turn green" -ForegroundColor White
    Write-Host "  3. Verify it says 'Docker Desktop is running'" -ForegroundColor White
    Write-Host ""
    Write-Host "First-time Docker Desktop setup:" -ForegroundColor Cyan
    Write-Host "  1. Accept the service agreement" -ForegroundColor White
    Write-Host "  2. Choose 'Use WSL 2 instead of Hyper-V' (recommended)" -ForegroundColor White
    Write-Host "  3. Wait for Docker to complete initialization" -ForegroundColor White
    Write-Host "  4. Settings > General > Enable 'Use the WSL 2 based engine'" -ForegroundColor White
    Write-Host ""
    Write-Host "After Docker Desktop is running, run this script again." -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Step 3: Building Docker images" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

Write-Host "This will build the Docker images for your gaming laptop." -ForegroundColor Yellow
Write-Host "This may take 15-30 minutes depending on your internet connection." -ForegroundColor Yellow
Write-Host ""
Write-Host "Note: You can skip this now and build later when needed." -ForegroundColor Gray
Write-Host ""

$build = Read-Host "Build Docker images now? (y/n)"
if ($build -eq "y" -or $build -eq "Y") {
    Write-Host "Building images with gaming laptop configuration..." -ForegroundColor Yellow
    Write-Host ""
    
    # Verify Docker is still running before building
    try {
        docker info | Out-Null
    } catch {
        Write-Host "  [X] Docker daemon stopped responding" -ForegroundColor Red
        Write-Host "      Please ensure Docker Desktop is running" -ForegroundColor Yellow
        exit 1
    }
    
    docker-compose -f docker-compose.gaming.yml build
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "  [OK] Docker images built successfully" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "  [X] Error building Docker images" -ForegroundColor Red
        Write-Host ""
        Write-Host "Common issues:" -ForegroundColor Yellow
        Write-Host "  - Docker Desktop stopped during build" -ForegroundColor White
        Write-Host "  - Network connectivity issues" -ForegroundColor White
        Write-Host "  - Insufficient disk space" -ForegroundColor White
        Write-Host ""
        Write-Host "You can try building again later with:" -ForegroundColor Cyan
        Write-Host "  docker-compose -f docker-compose.gaming.yml build" -ForegroundColor Gray
        Write-Host ""
        exit 1
    }
} else {
    Write-Host "  [SKIPPED] You can build images later with:" -ForegroundColor Yellow
    Write-Host "    docker-compose -f docker-compose.gaming.yml build" -ForegroundColor Gray
}

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Step 4: Creating data directories" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

$directories = @("data", "data/cache", "data/datasets", "models", "logs")
foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-Host "  [+] Created $dir" -ForegroundColor Green
    } else {
        Write-Host "  [OK] $dir already exists" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Your gaming laptop is ready for multimodal LLM development!" -ForegroundColor Green
Write-Host ""
Write-Host "Hardware Configuration:" -ForegroundColor Cyan
Write-Host "  - GPU: NVIDIA RTX 2060 with 6GB VRAM" -ForegroundColor White
Write-Host "  - CUDA Version: 12.7" -ForegroundColor White
Write-Host "  - Optimized for GPU-accelerated inference and training" -ForegroundColor White
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "  1. Start the services:" -ForegroundColor White
Write-Host "     docker-compose -f docker-compose.gaming.yml up -d" -ForegroundColor Yellow
Write-Host ""
Write-Host "  2. Access Jupyter Lab:" -ForegroundColor White
Write-Host "     http://localhost:8888 (token: gaming-llm)" -ForegroundColor Yellow
Write-Host ""
Write-Host "  3. Access API Documentation:" -ForegroundColor White
Write-Host "     http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host ""
Write-Host "  4. Run the quickstart notebook:" -ForegroundColor White
Write-Host "     Open notebooks/quickstart.ipynb in Jupyter" -ForegroundColor Yellow
Write-Host ""
Write-Host "  5. Monitor GPU usage:" -ForegroundColor White
Write-Host "     nvidia-smi -l 1" -ForegroundColor Yellow
Write-Host ""
Write-Host "For more information, see README.md and DEPLOYMENT.md" -ForegroundColor Gray
Write-Host ""
