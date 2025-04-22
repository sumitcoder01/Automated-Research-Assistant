# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source code into the container at /app
# Copies ./src directory from host to /app/src in container
COPY ./src ./src

# Set the PYTHONPATH environment variable
# Add /app/src to the path so imports like `from research_assistant...` work correctly
# inside your application code.
ENV PYTHONPATH=/app/src

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run uvicorn when the container launches
# Uvicorn finds the app relative to WORKDIR (/app), hence 'src.research_assistant.main:app'
# PYTHONPATH=/app/src ensures the internal imports within the app code are resolved.
CMD ["uvicorn", "src.research_assistant.main:app", "--host", "0.0.0.0", "--port", "8000"]
