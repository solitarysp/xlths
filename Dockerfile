# Multi-stage Dockerfile cho Mini-Shazam (DSP Audio Fingerprinting)
FROM python:3.11-slim

# Cài đặt ffmpeg để hỗ trợ giải mã file mp3, m4a, flac, wav
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements và cài đặt python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn và dữ liệu
COPY . .

# Mặc định chạy demo nhận diện âm thanh
CMD ["python", "main_demo.py"]
