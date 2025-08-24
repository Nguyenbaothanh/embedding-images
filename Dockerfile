# Base image
FROM python:3.10-slim

# Cài đặt gói hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Tạo thư mục làm việc
WORKDIR /app

# Copy file requirements.txt (nếu có)
COPY requirements.txt .

# Cài đặt Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code
COPY . .

# Mở port (mặc định)
EXPOSE 9999

# Cho phép override port qua biến môi trường PORT (Cloud Run/Render/Railway)
ENV PORT=9999

# Chạy app (tôn trọng biến PORT nếu được set)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-9999}"]
