# Use official Unsloth Docker image
FROM unsloth/unsloth:latest

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/transformers

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace/

# Create necessary directories
RUN mkdir -p /workspace/checkpoints /workspace/outputs /workspace/logs

# Set permissions
RUN chmod -R 755 /workspace

# Expose ports for Jupyter (optional)
EXPOSE 8888

# Set environment variables for training
ENV TOKENIZERS_PARALLELISM=false
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/transformers

# Default command
CMD ["python", "train.py"]