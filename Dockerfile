# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 1️⃣ Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2️⃣ Copy and run NLTK setup
COPY utils/setup_nltk.py ./utils/
RUN python ./utils/setup_nltk.py

# 3️⃣ Copy repo files
COPY . .

# 4️⃣ Default command: run unit tests (excludes integration)
CMD ["python", "-m", "unittest", "discover", "-s", "backend/tests/unit", "-p", "test_*.py"]