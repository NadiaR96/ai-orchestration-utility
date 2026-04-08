# Dockerfile
FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy repo files
COPY . .

# Default command (run tests)
CMD ["python", "-m", "unittest", "discover", "-s", "tests"]