# Use official Python slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements if you have
COPY requirements.txt .

# Install dependency
RUN pip install --no-cache-dir -r requirements.txt

# Copy code..
COPY . .

# Expose FastAPI default port
EXPOSE 8000

# Start the app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

