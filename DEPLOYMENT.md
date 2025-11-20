# Deployment Guide for Multimodal LLM

## Quick Setup on Target Devices

### 1. Clone Repository on Target Device

```bash
# On your Ubuntu NUC or Gaming Laptop
git clone https://github.com/YOUR_USERNAME/multimodal-llm-docker.git
cd multimodal-llm-docker
```

### 2. Choose Configuration

#### For Intel NUC (CPU-optimized):
```bash
# Start with NUC configuration
docker-compose -f docker-compose.nuc.yml up -d

# Or use the configuration file
cp configs/config.nuc.yaml configs/config.yaml
docker-compose up -d jupyter redis
```

#### For Gaming Laptop (GPU-optimized):
```bash
# Start with Gaming configuration  
docker-compose -f docker-compose.gaming.yml up -d

# Or use the configuration file
cp configs/config.gaming.yaml configs/config.yaml
docker-compose up -d
```

### 3. Download Models
```bash
# Download required models
docker-compose run --rm model-downloader

# Or run specific model downloads
python scripts/download_models.py --config configs/config.yaml
```

### 4. Access Services

- **Jupyter Lab**: http://localhost:8888 (token: multimodal-llm)
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Grafana Dashboard**: http://localhost:3000 (admin/multimodal)

### 5. Test the Setup

Run the quickstart notebook:
```bash
# Open Jupyter and run notebooks/quickstart.ipynb
# Or test via API
curl http://localhost:8000/health
```

## Hardware-Specific Optimizations

### Intel NUC Optimizations
- Smaller models (distilgpt2, whisper-tiny)
- CPU-only inference with quantization
- Reduced batch sizes and memory usage
- Single worker processes

### Gaming Laptop Optimizations  
- Larger models with GPU acceleration
- Mixed precision training
- Parallel processing capabilities
- Enhanced monitoring and logging

## Troubleshooting

### Common Issues
1. **Port conflicts**: Change ports in docker-compose files
2. **Memory issues**: Reduce model sizes or batch sizes
3. **GPU not detected**: Install nvidia-docker runtime
4. **Model download fails**: Check internet connection and disk space

### Performance Tuning
- Adjust `max_frames` for video processing
- Modify `batch_size` based on available memory
- Enable/disable optional models based on needs

## Development Workflow

1. **Local Development**: Use Jupyter for experimentation
2. **API Testing**: Test endpoints with Postman or curl
3. **Model Customization**: Fine-tune models for your domain
4. **Deployment**: Use appropriate docker-compose configuration

## Next Steps

1. **Customize Models**: Replace with domain-specific models
2. **Add Training**: Implement custom training pipelines
3. **Scale Deployment**: Use orchestration tools like Kubernetes
4. **Monitor Performance**: Set up comprehensive monitoring