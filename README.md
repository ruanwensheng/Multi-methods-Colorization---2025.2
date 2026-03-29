## CLONE AND SET UP

# 1. clone repo về máy
git clone https://github.com/ruanwensheng/Multi-methods-Colorization---2025.2.git
cd colorization2025.2

# 2. Tạo venv riêng trên máy tính
python -m venv venv
venv\Scripts\activate

# 3. cài đặt thư viện, lưu ý nên dùng Python 3.13.5
pip install -r requirements

# 4. chuyển sang branch của mình, ví dụ bạn làm example-based
git checkout feature/example

# 5. kiểm tra env Ok chưa
python -c "import cv2, numpy, gradio, mlflow; print('hihi')"

## MỖI LẦN LÀM VIỆC

# 1. activate venv
venv\Scripts\activate

# 2. cập nhật code mới từ develop
git checkout feature/example
git fetch origin
git merge origin/develop    

# 3. check xem có thư viện mới không
pip install -r requirements.txt

# 4. sau khi code xong, đảm bảo code chạy đc ko lỗi
git add src/example_based/
git add requirements.txt
git commit -m "feat: example-based color ......"

git push origin feature/example
# lưu ý: ko push trực tiếp lên develop, cái đó sẽ để pull request và merge trên giao diện
# 5. khi thêm thư viện mới
pip install scikit-learn
pip freeze > requirements.txt

git add requirements.txt src/example_based/<my_new_file.py>

git push origin feature/example


