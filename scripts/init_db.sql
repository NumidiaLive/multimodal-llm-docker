-- Initialize database tables for multimodal LLM

-- Users table for API authentication
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- API keys table
CREATE TABLE IF NOT EXISTS api_keys (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    key_hash VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Request logs table
CREATE TABLE IF NOT EXISTS request_logs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER,
    request_size_bytes BIGINT,
    response_size_bytes BIGINT,
    processing_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ip_address INET,
    user_agent TEXT
);

-- Model usage statistics
CREATE TABLE IF NOT EXISTS model_usage (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    model_type VARCHAR(50) NOT NULL, -- text, audio, video, multimodal
    model_name VARCHAR(255) NOT NULL,
    input_size_bytes BIGINT,
    output_size_bytes BIGINT,
    processing_time_ms INTEGER,
    device VARCHAR(20), -- cpu, cuda, mps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Training jobs table
CREATE TABLE IF NOT EXISTS training_jobs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    job_name VARCHAR(255) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending', -- pending, running, completed, failed
    config JSONB,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    error_message TEXT,
    metrics JSONB
);

-- Model metadata table
CREATE TABLE IF NOT EXISTS model_metadata (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(255) UNIQUE NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    version VARCHAR(50),
    size_mb INTEGER,
    parameters_count BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT,
    config JSONB
);

-- System metrics table
CREATE TABLE IF NOT EXISTS system_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    cpu_usage_percent FLOAT,
    memory_usage_percent FLOAT,
    memory_used_gb FLOAT,
    gpu_usage_percent FLOAT,
    gpu_memory_used_gb FLOAT,
    disk_usage_percent FLOAT,
    active_requests INTEGER,
    total_requests_per_minute INTEGER
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_request_logs_created_at ON request_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_request_logs_user_id ON request_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_model_usage_created_at ON model_usage(created_at);
CREATE INDEX IF NOT EXISTS idx_model_usage_user_id ON model_usage(user_id);
CREATE INDEX IF NOT EXISTS idx_training_jobs_user_id ON training_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON training_jobs(status);
CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp);

-- Insert default admin user (password: admin123)
-- Note: In production, use a secure password hash
INSERT INTO users (username, email, password_hash) VALUES 
('admin', 'admin@multimodal-llm.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewHrM5JVc6r3.4em')
ON CONFLICT (username) DO NOTHING;

-- Insert some sample model metadata
INSERT INTO model_metadata (model_name, model_type, version, size_mb, parameters_count, description) VALUES
('microsoft/DialoGPT-small', 'text', '1.0', 100, 117000000, 'Small conversational model for text generation'),
('distilgpt2', 'text', '1.0', 80, 82000000, 'Distilled GPT-2 model for efficient text generation'),
('whisper-base', 'audio', '1.0', 142, 74000000, 'Whisper base model for speech recognition'),
('whisper-tiny', 'audio', '1.0', 39, 39000000, 'Whisper tiny model for fast speech recognition'),
('openai/clip-vit-base-patch32', 'video', '1.0', 151, 151000000, 'CLIP model for image-text understanding')
ON CONFLICT (model_name) DO NOTHING;