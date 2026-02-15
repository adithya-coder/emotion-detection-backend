# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set environment variables for faster startup
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    DEBIAN_FRONTEND=noninteractive

# Set the working directory in the container
WORKDIR /app

# Install necessary system dependencies for OpenCV (minimal set)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY requirements_fixed.txt /app/requirements.txt

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip

# Install Python dependencies in one layer
RUN pip install --no-cache-dir \
    Flask==3.1.0 \
    flask-cors==5.0.0 \
    numpy==2.0.2 \
    opencv-python-headless==4.10.0.84 \
    Pillow==11.0.0 \
    gunicorn==23.0.0 \
    tensorflow-cpu==2.18.0 \
    && pip cache purge

# Copy the rest of the application
COPY . /app

# Make port 5000 available
EXPOSE 5000

# Health check with longer timeout for startup
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:5000/', timeout=5)" || exit 1

# Run gunicorn with optimized settings
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--timeout", "120", \
     "--workers", "1", \
     "--threads", "2", \
     "--worker-class", "gthread", \
     "--preload", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info", \
     "app:app"]
