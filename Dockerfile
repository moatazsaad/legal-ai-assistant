# Use official slim Python 3.11 image as base
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Upgrade pip and install Python dependencies including multipart for form handling
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir python-multipart

# Copy the rest of the application code into container
COPY . .

# Expose port 8000 for FastAPI
EXPOSE 8000

# Run FastAPI app using Uvicorn
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
