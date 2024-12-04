FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose port for web interface
EXPOSE 8000

# Command to run the web application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]