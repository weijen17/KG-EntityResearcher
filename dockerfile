FROM python:3.12-slim

WORKDIR /app

# Set environment variables (adds metadata)
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

RUN mkdir -p outputs

# Dummy port, even if not actually used
EXPOSE 5000

# May be needed to prevent error on docker-compose up
# VOLUME ["/app/outputs", "/app/src"]

# Run the application
CMD ["python", "main.py"]

# for flask application
# CMD ["python", "app.py"]