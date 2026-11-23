# Kết quả cuối của các thí nghiệm CT-BERT

So sánh sử dụng Validation 2.139 mẫu và Test mirror 2.140 mẫu đã từng dùng. Mỗi CT-BERT fit 3.792 mẫu, chọn epoch trên 702 mẫu nội bộ, hiệu chỉnh sigmoid trên C, và giữ seed 42. Weighted Mean và Choquet dùng xác suất của CT-BERT gốc và DistilBERT, với hệ số đã khóa trên M. Biến thể URL và emoji là mô hình đơn với checkpoint riêng. Không ghép biến thể này vào các ensemble cũ.

| Phương pháp | Weighted-F1 Val | Weighted-F1 Test | Lỗi Val | Lỗi Test |
|---|---:|---:|---:|---:|
| CT-BERT gốc, đã hiệu chỉnh | 98,223% | 98,271% | 38 | 37 |
| CT-BERT + DistilBERT, Weighted Mean | 98,270% | 98,271% | 37 | 37 |
| CT-BERT + DistilBERT, Choquet | 98,176% | 98,271% | 39 | 37 |
| CT-BERT xử lý URL và emoji, đã hiệu chỉnh | 98,457% | 98,318% | 33 | 36 |

- **Tiền xử lý.** URL và emoji được thử như một gói. So với CT-BERT gốc đã hiệu chỉnh, biến thể mới sửa 11 lỗi và tạo 6 lỗi trên Val, sửa 7 lỗi và tạo 6 lỗi trên Test. Chênh lệch Weighted-F1 lần lượt là +0,234 và +0,047 điểm phần trăm. Khoảng bootstrap ghép cặp theo mẫu 95% lần lượt là [−0,140; +0,609] và [−0,280; +0,374] điểm phần trăm. Mẫu gần trùng hoặc cùng nguồn có thể phụ thuộc, nên đây là khoảng tham khảo. Một seed và kết quả trên các tập đã được dùng trước đây giới hạn khả năng suy rộng.
- **Hiệu chỉnh.** Với CT-BERT gốc trên Val, hiệu chỉnh làm Log loss giảm từ 0,10986 xuống 0,06258 và Brier giảm từ 0,01707 xuống 0,01511. Biến thể URL và emoji đã hiệu chỉnh đạt Log loss 0,05407 và Brier 0,01313 trên Val. Checkpoint và cách hiệu chỉnh được giữ cố định trước lượt Test.
- **Ensemble.** Weighted Mean giữ nguyên nhãn dự đoán của CT-BERT gốc trên toàn bộ Test mirror. Choquet sửa một lỗi và tạo một lỗi so với Weighted Mean. Choquet có Brier thấp hơn nhẹ (0,015083 so với 0,015145), trong khi Log loss của hai phép kết hợp gần bằng nhau (0,064407 và 0,064398). Các kết quả chưa hỗ trợ kết luận Choquet tăng F1.
- **Chọn checkpoint.** Biến thể mới chọn epoch 2, với Train/Val nội bộ loss 0,02012/0,06679, Weighted-F1 99,578%/98,573%, gap 0,04667 và 0,01005. Hai gap dưới 0,1 là điều kiện chọn checkpoint. Thời gian huấn luyện khoảng 34 phút 47 giây, suy luận Test 23,94 giây.
- **Đối chiếu benchmark.** Weighted-F1 Test mirror tốt nhất 98,318%, thấp hơn điểm Test công bố 98,69% của Glazkova et al. khoảng 0,372 điểm phần trăm. Hai nguồn có cùng 2.140 mẫu theo tài liệu, nhưng từng ID, văn bản và nhãn của mirror chưa được xác minh với bản chính thức. [Bài báo g2tmn](https://arxiv.org/abs/2012.11967)
- **Rò rỉ và tổng quát hóa.** Val chính loại một mẫu trùng Train đã xác minh. Test mirror giữ nguyên 2.140 mẫu và có hai trường hợp trùng chuẩn hóa với Train đã biết. Điểm ghi nhận trên dữ liệu đã xem trước đây, nên hiệu quả trên dữ liệu mới và độ ổn định qua seed vẫn cần kiểm chứng.

Bảng đầy đủ và ma trận dưới dạng số nằm trong metrics_all_validation_test.csv. Cột split ghi tập đánh giá. Thời gian có phạm vi đo riêng trong timing_existing_models.csv và ctbert_training_and_timing.csv. Ô trống biểu thị đại lượng chưa đo hoặc chưa lưu. Phân tích ghép cặp được lưu trong paired_analysis.csv.
