"""
Validate gaming laptop setup for multimodal LLM Docker project.
Tests GPU availability, Docker setup, and Python dependencies.
"""

import sys
import subprocess
from pathlib import Path

def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def check_python_version():
    """Check Python version."""
    print("\n[*] Python Version:")
    print(f"    {sys.version}")
    
    major, minor = sys.version_info[:2]
    if major == 3 and minor >= 8:
        print("    ✓ Python version is compatible")
        return True
    else:
        print("    ✗ Python 3.8+ required")
        return False

def check_pytorch_cuda():
    """Check PyTorch and CUDA availability."""
    print("\n[*] PyTorch & CUDA:")
    
    try:
        import torch
        print(f"    PyTorch Version: {torch.__version__}")
        print(f"    CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"    CUDA Version: {torch.version.cuda}")
            print(f"    GPU Count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"    GPU {i}: {name} ({memory:.1f}GB)")
            
            # Test GPU tensor creation
            try:
                x = torch.rand(5, 3).cuda()
                print("    ✓ GPU tensor creation successful")
                return True
            except Exception as e:
                print(f"    ✗ GPU tensor creation failed: {e}")
                return False
        else:
            print("    ✗ CUDA not available")
            return False
            
    except ImportError:
        print("    ✗ PyTorch not installed")
        print("    Install with: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return False

def check_transformers():
    """Check Transformers library."""
    print("\n[*] Transformers Library:")
    
    try:
        import transformers
        print(f"    Version: {transformers.__version__}")
        print("    ✓ Transformers installed")
        return True
    except ImportError:
        print("    ✗ Transformers not installed")
        print("    Install with: pip install transformers")
        return False

def check_docker():
    """Check Docker installation."""
    print("\n[*] Docker:")
    
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"    {result.stdout.strip()}")
        
        # Check if Docker is running
        try:
            subprocess.run(
                ["docker", "ps"],
                capture_output=True,
                check=True
            )
            print("    ✓ Docker is running")
            return True
        except subprocess.CalledProcessError:
            print("    ✗ Docker is not running")
            print("    Please start Docker Desktop")
            return False
            
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("    ✗ Docker not installed")
        return False

def check_gpu_docker():
    """Check Docker GPU support."""
    print("\n[*] Docker GPU Support:")
    
    try:
        result = subprocess.run(
            ["docker", "run", "--rm", "--gpus", "all", 
             "nvidia/cuda:12.1.0-base-ubuntu22.04", "nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("    ✓ Docker GPU support working")
            # Parse GPU info from nvidia-smi output
            for line in result.stdout.split('\n'):
                if 'NVIDIA' in line or 'GeForce' in line:
                    print(f"    {line.strip()}")
            return True
        else:
            print("    ✗ Docker GPU support not working")
            print(f"    Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("    ✗ Docker GPU test timed out")
        return False
    except Exception as e:
        print(f"    ✗ Docker GPU test failed: {e}")
        return False

def check_project_structure():
    """Check project directory structure."""
    print("\n[*] Project Structure:")
    
    required_dirs = [
        "docker",
        "src",
        "configs",
        "notebooks",
        "scripts"
    ]
    
    required_files = [
        "docker-compose.gaming.yml",
        "configs/config.gaming.yaml",
        "README.md"
    ]
    
    all_good = True
    
    for dir_name in required_dirs:
        path = Path(dir_name)
        if path.exists():
            print(f"    ✓ {dir_name}/")
        else:
            print(f"    ✗ {dir_name}/ missing")
            all_good = False
    
    for file_name in required_files:
        path = Path(file_name)
        if path.exists():
            print(f"    ✓ {file_name}")
        else:
            print(f"    ✗ {file_name} missing")
            all_good = False
    
    return all_good

def check_optional_packages():
    """Check optional packages."""
    print("\n[*] Optional Packages:")
    
    packages = {
        "numpy": "numpy",
        "pandas": "pandas",
        "pillow": "PIL",
        "opencv": "cv2",
        "librosa": "librosa",
        "fastapi": "fastapi",
    }
    
    for name, import_name in packages.items():
        try:
            __import__(import_name)
            print(f"    ✓ {name}")
        except ImportError:
            print(f"    ○ {name} (optional)")

def main():
    """Run all validation checks."""
    print_header("Gaming Laptop Setup Validation")
    print("\nRTX 2060 Configuration")
    print("Checking multimodal LLM Docker environment...")
    
    results = []
    
    # Run checks
    results.append(("Python Version", check_python_version()))
    results.append(("PyTorch & CUDA", check_pytorch_cuda()))
    results.append(("Transformers", check_transformers()))
    results.append(("Docker", check_docker()))
    results.append(("Project Structure", check_project_structure()))
    
    # Check Docker GPU only if Docker is working
    if results[3][1]:  # Docker check passed
        results.append(("Docker GPU Support", check_gpu_docker()))
    
    # Check optional packages
    check_optional_packages()
    
    # Summary
    print_header("Validation Summary")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status:8} - {check_name}")
    
    print(f"\n  Score: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n  🎉 Your gaming laptop is ready for multimodal LLM development!")
        print("\n  Next steps:")
        print("    1. Run: docker-compose -f docker-compose.gaming.yml up -d")
        print("    2. Open: http://localhost:8888 (token: gaming-llm)")
        print("    3. Try: notebooks/quickstart.ipynb")
        return 0
    else:
        print("\n  ⚠️  Some checks failed. Please address the issues above.")
        print("\n  See GAMING-LAPTOP-SETUP.md for troubleshooting.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
