# Use lightweight Python
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies (required for XGBoost)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "8000"]
