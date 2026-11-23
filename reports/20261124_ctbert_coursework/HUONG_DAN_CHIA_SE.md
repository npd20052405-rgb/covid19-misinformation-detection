# File bàn giao cho nhóm

Dùng báo cáo phân tích, bảng CSV và hình trong gói này để viết Word. Bảng báo cáo và hình có thể xem độc lập với app.

Mở Notebook 19–22 bằng VS Code để xem quy trình CT-BERT. Tại thư mục gốc, chạy python -m streamlit run apps/coursework_v1/app.py --server.fileWatcherType none sau khi chuẩn bị Python 3.11, các gói trong apps/coursework_v1/requirements.txt và PyTorch CUDA phù hợp.

App cần checkpoint, tokenizer, calibrator và file khóa tại các đường dẫn tương đối ghi trong apps/coursework_v1/model_registry.json. Gói chia sẻ gồm mã, số liệu tổng hợp và hình. Trên máy đã có các artifact dự án, app dùng trực tiếp chúng. Trên máy chỉ giải nén gói, có thể mở bảng tổng quan, còn dự đoán cần những artifact cục bộ đã xác minh.

Khi đưa lên GitHub, dùng các file trong gói chia sẻ. Dữ liệu gốc, nhãn Test theo mẫu, checkpoint lớn và dự đoán theo từng mẫu được giữ trong dự án cục bộ. Kiểm tra quyền phân phối riêng nếu cần chuyển dữ liệu hoặc trọng số cho thành viên.
