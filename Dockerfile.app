FROM python:3.9.20-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools first
RUN pip install --upgrade pip setuptools wheel

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "apps/app.py", "--server.port=8501", "--server.address=0.0.0.0"]