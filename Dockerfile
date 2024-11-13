# Use a minimal Python 3.11 image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the project files, excluding files in .dockerignore
COPY . /app

# Upgrade pip and install dependencies
RUN pip install --upgrade pip 

# Install the package itself using setuptools
RUN pip install .

# Set up entrypoint
CMD ["python", "run.py"]