# Use TensorFlow CPU base image (lighter, no CUDA)
FROM tensorflow/tensorflow:2.12.0

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    python3-opencv \
    tzdata && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training script and setup
COPY lithostrain.py .
COPY setup.py .

# Optional: if using setup.py to define a module
# RUN pip install .

# Set the default command
ENTRYPOINT ["python", "lithostrain.py"]
