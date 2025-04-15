# Use Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy necessary files
COPY prism_train.py ./
COPY requirements.txt ./
COPY setup.py ./


# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt



# Set entrypoint
ENTRYPOINT ["python", "-m", "prism_train"]