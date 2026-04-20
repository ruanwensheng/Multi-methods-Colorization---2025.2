## Clone and Set Up

1. **Clone repository về máy**
   ```bash
   git clone https://github.com/ruanwensheng/Multi-methods-Colorization---2025.2.git
   cd colorization2025.2
   ```

2. **Tạo venv riêng trên máy tính**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
có thể dùng conda env    

3. **Cài đặt thư viện**  
   *Lưu ý: nên dùng Python 3.13.5*
   ```bash
   pip install -r requirements
   ```

4. **Chuyển sang branch của bạn**  
   *Ví dụ bạn làm example-based*
   ```bash
   git checkout feature/example
   ```

5. **Kiểm tra môi trường OK chưa**
   ```bash
   python -c "import cv2, numpy, gradio, mlflow; print('hihi')"
   ```

---

## Mỗi lần làm việc

1. **Activate venv**
   ```bash
   venv\Scripts\activate
   # conda env: conda activate <tên env on your local>
   ```

2. **Cập nhật code mới từ develop**
   ```bash
   git checkout feature/example  # về nhánh của bạn
   git fetch origin              # tải toàn bộ file từ github về 
   git merge origin/develop      # merge code trên develop vs nhánh đang làm    
   ```

3. **Check xem có thư viện mới không**
   ```bash
   pip install -r requirements.txt
   ```

4. **Sau khi code xong, đảm bảo code chạy được không lỗi**
   ```bash
   git add src/example_based/
   git add requirements.txt
   git commit -m "feat: example-based color ......"
   git push origin feature/example    # code xong thì push lên branch của mình
   ```
   > **Lưu ý:** không push trực tiếp lên `develop`, cái đó sẽ để pull request và merge trên giao diện.



