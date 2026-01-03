# Use Python 3.10
FROM python:3.10-slim

# Set working directory to /app
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY ml-service/requirements.txt .

# Install dependencies
# We use the CPU version of TensorFlow to keep the image smaller and faster
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code from ml-service directory
COPY ml-service/ .

# Create a non-root user (good practice for HF Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Hugging Face Spaces expects the app to listen on port 7860
EXPOSE 7860

# Start command
# We use 2 workers and 4 threads here since HF Spaces has decent RAM (16GB)
CMD ["gunicorn", "--workers", "2", "--threads", "4", "--timeout", "120", "--bind", "0.0.0.0:7860", "app:app"]
