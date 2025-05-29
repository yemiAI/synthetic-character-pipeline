# Use CUDA-enabled PyTorch base
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip git wget libgl1 libglib2.0-0 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip install --upgrade pip

# Copy project
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set default entrypoint
CMD ["python", "inference/generate.py"]
