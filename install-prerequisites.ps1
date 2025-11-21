# Multimodal LLM Docker - Prerequisites Installation Script
# Installs all required software using winget with machine scope

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "   Prerequisites Installation via WinGet          " -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "WARNING: Not running as Administrator" -ForegroundColor Yellow
    Write-Host "For machine-wide installation, please run PowerShell as Administrator" -ForegroundColor Yellow
    Write-Host ""
    $continue = Read-Host "Continue anyway with user scope? (y/n)"
    if ($continue -ne "y" -and $continue -ne "Y") {
        exit 0
    }
    $scope = "--scope user"
} else {
    Write-Host "Running as Administrator - will install with machine scope" -ForegroundColor Green
    $scope = "--scope machine"
}

Write-Host ""
Write-Host "Checking winget availability..." -ForegroundColor Yellow

# Check winget is available
try {
    $wingetVersion = winget --version
    Write-Host "  winget version: $wingetVersion" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: winget is not available" -ForegroundColor Red
    Write-Host "  Please install App Installer from Microsoft Store" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Step 1: Installing Git" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

try {
    $gitInstalled = git --version 2>$null
    Write-Host "  Git already installed: $gitInstalled" -ForegroundColor Green
} catch {
    Write-Host "  Installing Git for Windows..." -ForegroundColor Yellow
    winget install -e --id Git.Git $scope --accept-package-agreements --accept-source-agreements
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  Git installed successfully" -ForegroundColor Green
    } else {
        Write-Host "  Failed to install Git" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Step 2: Installing Python 3.11" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

try {
    $pythonInstalled = python --version 2>$null
    if ($pythonInstalled -match "3\.1[1-9]") {
        Write-Host "  Python already installed: $pythonInstalled" -ForegroundColor Green
    } else {
        throw "Old version or not found"
    }
} catch {
    Write-Host "  Installing Python 3.11..." -ForegroundColor Yellow
    winget install -e --id Python.Python.3.11 $scope --accept-package-agreements --accept-source-agreements
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  Python 3.11 installed successfully" -ForegroundColor Green
    } else {
        Write-Host "  Failed to install Python" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Step 3: Installing Docker Desktop" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

try {
    $dockerInstalled = docker --version 2>$null
    Write-Host "  Docker already installed: $dockerInstalled" -ForegroundColor Green
} catch {
    Write-Host "  Installing Docker Desktop..." -ForegroundColor Yellow
    Write-Host "  This is a large download (570MB) and may take several minutes..." -ForegroundColor Yellow
    
    winget install -e --id Docker.DockerDesktop $scope --accept-package-agreements --accept-source-agreements
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  Docker Desktop installed successfully" -ForegroundColor Green
        Write-Host ""
        Write-Host "  IMPORTANT: You must restart your computer for Docker to work properly" -ForegroundColor Yellow
    } else {
        Write-Host "  Failed to install Docker Desktop" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Step 4: Installing Windows Terminal (Optional)" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

try {
    # Check if Windows Terminal is installed
    $wtInstalled = Get-AppxPackage -Name Microsoft.WindowsTerminal 2>$null
    if ($wtInstalled) {
        Write-Host "  Windows Terminal already installed" -ForegroundColor Green
    } else {
        throw "Not found"
    }
} catch {
    Write-Host "  Installing Windows Terminal..." -ForegroundColor Yellow
    winget install -e --id Microsoft.WindowsTerminal $scope --accept-package-agreements --accept-source-agreements
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  Windows Terminal installed successfully" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Step 5: Setting up WSL2" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

try {
    $wslStatus = wsl --status 2>$null
    if ($wslStatus) {
        Write-Host "  WSL2 already installed" -ForegroundColor Green
        wsl --status
    } else {
        throw "Not installed"
    }
} catch {
    Write-Host "  Installing WSL2..." -ForegroundColor Yellow
    Write-Host "  This requires administrator privileges..." -ForegroundColor Yellow
    
    if ($isAdmin) {
        wsl --install --no-distribution
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  WSL2 installed successfully" -ForegroundColor Green
            Write-Host "  IMPORTANT: You must restart your computer for WSL2 to work" -ForegroundColor Yellow
        } else {
            Write-Host "  Failed to install WSL2" -ForegroundColor Red
        }
    } else {
        Write-Host "  SKIPPED: Administrator privileges required for WSL2 installation" -ForegroundColor Yellow
        Write-Host "  Please run this command manually as Administrator:" -ForegroundColor Yellow
        Write-Host "    wsl --install --no-distribution" -ForegroundColor Cyan
    }
}

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Installation Summary" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

# Refresh environment variables
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# Check what's installed
$installed = @()
$needsRestart = $false

try {
    $gitVer = git --version 2>$null
    $installed += "  [OK] Git: $gitVer"
} catch {
    $installed += "  [X] Git: Not found"
}

try {
    $pyVer = python --version 2>$null
    $installed += "  [OK] Python: $pyVer"
} catch {
    $installed += "  [X] Python: Not found (may need PATH refresh)"
}

try {
    $dockerVer = docker --version 2>$null
    $installed += "  [OK] Docker: $dockerVer"
} catch {
    $installed += "  [!] Docker: Not found (restart required)"
    $needsRestart = $true
}

try {
    $wslVer = wsl --version 2>$null
    $installed += "  [OK] WSL2: Installed"
} catch {
    $installed += "  [!] WSL2: Not found (restart may be required)"
}

try {
    $nvidiaSmi = nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
    $installed += "  [OK] NVIDIA GPU: $nvidiaSmi"
} catch {
    $installed += "  [!] NVIDIA GPU: Not detected"
}

foreach ($item in $installed) {
    Write-Host $item
}

Write-Host ""
if ($needsRestart) {
    Write-Host "==================================================" -ForegroundColor Yellow
    Write-Host "  ACTION REQUIRED: RESTART YOUR COMPUTER" -ForegroundColor Yellow
    Write-Host "==================================================" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Several components require a system restart to work properly:" -ForegroundColor Yellow
    Write-Host "  - Docker Desktop" -ForegroundColor White
    Write-Host "  - WSL2" -ForegroundColor White
    Write-Host "  - Updated PATH environment variables" -ForegroundColor White
    Write-Host ""
    Write-Host "After restarting:" -ForegroundColor Cyan
    Write-Host "  1. Start Docker Desktop from the Start menu" -ForegroundColor White
    Write-Host "  2. Wait for Docker to fully start (green whale icon)" -ForegroundColor White
    Write-Host "  3. Run: .\setup-gaming.ps1" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host "==================================================" -ForegroundColor Green
    Write-Host "  Installation Complete!" -ForegroundColor Green
    Write-Host "==================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Close and reopen PowerShell to refresh PATH" -ForegroundColor White
    Write-Host "  2. Start Docker Desktop" -ForegroundColor White
    Write-Host "  3. Run: .\setup-gaming.ps1" -ForegroundColor White
    Write-Host ""
}

Write-Host "For more information, see GAMING-LAPTOP-SETUP.md" -ForegroundColor Gray
Write-Host ""
