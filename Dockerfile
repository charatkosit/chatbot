# ใช้ base image จาก Python
FROM python:3.11-slim

# ตั้งค่า working directory
WORKDIR /app

# คัดลอกไฟล์ requirements.txt ไปที่ working directory
COPY requirements.txt .

# ติดตั้ง dependencies
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกไฟล์ที่เหลือไปที่ working directory
COPY . .
COPY data /data

# กำหนด environment variable สำหรับ Flask
ENV FLASK_APP=agent_line_hist.py
ENV FLASK_RUN_PORT=5100

# เปิด port 5100
EXPOSE 5100

# รัน Flask app
CMD ["flask", "run", "--host=0.0.0.0", "--port=5100"]
