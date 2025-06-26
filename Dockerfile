# Use an official slim Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

# Set working directory
WORKDIR /app

# Install OS dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        poppler-utils \
        build-essential \
        gcc \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY ./src ./src

# Expose FastAPI default port
EXPOSE 8000

# Launch Uvicorn server
CMD ["uvicorn", "src.research_assistant.main:app", "--host", "0.0.0.0", "--port", "8000"]
