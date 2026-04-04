# 🌌 Project Sambhav — Backend Deployment (HuggingFace Docker SDK)
# Section 26.2: Production Deployment Blueprint
# ------------------------------------------------------------

FROM python:3.10-slim

# Set environment variables for production
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PORT 7860
ENV DEPLOY_ENV=production

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    git \
    git-lfs \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Initialize git-lfs for HF Hub models
RUN git lfs install

# Copy dependency definition
COPY requirements.txt .

# Install dependencies (use cache where possible)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the core project source code
COPY . .

# HF Spaces requires a user with UID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH

# Expose the mandatory HF Spaces port
EXPOSE 7860

# Simple health check endpoint defined in main.py
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:7860/health || exit 1

# Start the FastAPI server using uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
