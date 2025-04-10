# Dockerfile for SegFormer API
FROM python:3.9-slim

# Add compiler for PyTorch inductor backend
RUN apt-get update && apt-get install -y g++ cmake

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Expose port and run API
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]