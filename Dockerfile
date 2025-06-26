# --- Stage 1: Builder ---
# This stage is for installing system dependencies and Python packages into a virtual environment.
FROM python:3.11-slim as builder

# Set environment variables to prevent Python from writing .pyc files and to buffer output
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# --- Install System Dependencies ---
# This is the crucial step to install Tesseract OCR and Poppler utilities.
# `--no-install-recommends` prevents installation of optional packages, keeping the layer smaller.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        poppler-utils \
        # Add build-essential & gcc in case any python package needs to compile C code
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# --- Create and Prepare the Virtual Environment ---
# This creates a virtual environment in a standard location, /opt/venv
RUN python -m venv /opt/venv

# Set the PATH environment variable to use the venv's binaries for subsequent commands.
# This effectively "activates" the venv for the rest of this build stage.
ENV PATH="/opt/venv/bin:$PATH"

# Copy the requirements file and install Python dependencies into the virtual environment
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# --- Stage 2: Final Production Image ---
# This stage creates the final, lean image for runtime. It starts from a fresh base
# to ensure no build-time tools are included.
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Set environment variables for runtime
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# --- Install ONLY the necessary runtime system dependencies ---
# We install Tesseract and Poppler again, but not the build tools (like gcc).
# This keeps the final image as small as possible.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# --- Copy the virtual environment from the builder stage ---
# This brings in all our Python dependencies without having to reinstall them.
COPY --from=builder /opt/venv /opt/venv

# --- Copy the application source code ---
# Ensure your .dockerignore file is correctly set up.
COPY ./src ./src

# --- Activate the virtual environment for the container's runtime ---
# This ensures that when the CMD runs, it uses the Python interpreter and packages from our venv.
ENV PATH="/opt/venv/bin:$PATH"

# Make port 8000 available to the world outside this container
EXPOSE 8000

# --- Run the application using Gunicorn
# Gunicorn is a more robust production server that manages Uvicorn workers.
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "src.research_assistant.main:app", "--bind", "0.0.0.0:8000"]
