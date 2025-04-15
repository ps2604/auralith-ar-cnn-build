FROM tensorflow/tensorflow:2.13.0-gpu

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install minimal OpenCV dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy training code
COPY . .

# Set entrypoint
ENTRYPOINT ["python", "train.py"]