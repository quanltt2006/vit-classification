# 1. Chọn hệ điều hành và môi trường Python có sẵn (Giống như sắm cái bếp)
# Bản "slim" giúp dung lượng Image nhẹ hơn rất nhiều
FROM python:3.10-slim

# 2. Tạo một thư mục làm việc bên trong Container (Dọn dẹp một góc bàn bếp)
WORKDIR /app

# 3. Copy file chứa danh sách thư viện vào trước
# Lưu ý: Cần có file requirements.txt nằm cùng chỗ với Dockerfile
COPY requirements.txt .

# 4. Cài đặt các thư viện AI/ML (Mua gia vị)
# Lệnh --no-cache-dir giúp xoá file rác sau khi cài, làm Image nhẹ hơn
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy toàn bộ file code của bạn vào thư mục /app trong Container (Cho nguyên liệu vào nồi)
COPY . .

# 6. Lệnh khởi chạy code khi Container bật lên (Bật bếp)
# (Dùng Exec form chuẩn như bạn đã tìm hiểu)
CMD ["python", "train.py"]