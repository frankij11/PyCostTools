# Multi-stage Dockerfile for testing PyCost across multiple Python versions
ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install -r requirements-dev.txt

# Copy the entire project
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create a non-root user
RUN useradd --create-home --shell /bin/bash pycost
RUN chown -R pycost:pycost /app
USER pycost

# Default command runs tests
CMD ["python", "-m", "pytest", "tests/", "-v", "--cov=pycost", "--cov-report=term-missing"]

# Testing stage - runs all examples
FROM base as test-examples

# Run all examples to ensure they work
RUN echo "Testing examples..." && \
    cd examples && \
    python super_simple.py && \
    python simple.py && \
    python demo_program.py && \
    python manufacturing_lot_example.py && \
    python model_analysis_example.py && \
    echo "All examples completed successfully!"

# Production stage - minimal image for deployment
FROM python:${PYTHON_VERSION}-slim as production

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copy only necessary files
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pycost/ ./pycost/
COPY setup.py pyproject.toml README.md LICENSE.txt ./

# Install the package
RUN pip install .

# Create non-root user
RUN useradd --create-home --shell /bin/bash pycost
USER pycost

# Default command
CMD ["python", "-c", "import pycost; print('PyCost installed successfully!')"] 