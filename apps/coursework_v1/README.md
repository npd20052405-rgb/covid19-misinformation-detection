# Streamlit — một giao thức thống nhất

## Chạy trên Windows

Tại root dự án:

```powershell
& 'C:\Program Files\Python311\python.exe' -m streamlit run apps/coursework_v1/app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false --server.fileWatcherType none
```

Mở http://127.0.0.1:8501. Nếu server cũ còn chạy, Ctrl+C rồi chạy lại. Watcher đã tắt nên refresh có thể chưa nạp module mới. Không tạo server trùng cổng.

## Ba trang

- **Tổng quan:** một bảng Validation 2.139 / Test 2.140, 11 mô hình đơn và 4 Ensemble chính. Không có menu ba giao thức, không hiển thị Dummy.
- **Dự đoán:** nhập tiếng Anh, chọn nhiều mô hình, xem kết quả cùng lúc. Inference thật qua đúng preprocessing, vectorizer/tokenizer, checkpoint, calibrator và pooling; không dùng cached prediction giả làm mô hình.
- Trong trang **Dự đoán**, chọn **Tải file CSV** để xử lý nhiều dòng. Cột bắt buộc: `text`; tùy chọn: `id`, `expected_label`, `source_url`. File phải UTF-8, tối đa2MB/100dòng/200.000ký tự tổng. Nhãn tham chiếu chỉ được đối chiếu sau inference, không đưa vào mô hình. Có bảng tổng hợp, xác suất chi tiết và tải kết quảCSV. Văn bản chỉ xử lý cục bộ.
- Bộ đọc tự nhận dấu phẩy, chấm phẩy hoặc tab dựa trên header; bỏ khoảng trắng và không phân biệt hoa/thường ở tên cột. Không thay đổi nội dung văn bản. Chặn header trùng và cấu trúc cột không nhất quán.
- File `demo_5_real_5_fake.csv` có5đoạn Real và5câu Fake không trùng nguyên văn/sau chuẩn hóa trong Train nguồn6.420 hoặc Validation nguồn. RoBERTa-Base dự đoán đúng10/10 trên bộ minh họa đã thử này; DistilBERT và SVM+DistilBERT Choquet đúng9/10. Đây là bộ trình diễn được chọn sau khi quan sát kết quả, không là benchmark hoặc bằng chứng tổng quát hóa; cùng sự kiện có thể đã xuất hiện trong Train.
- **LIME:** giải thích cục bộ bằng cách ẩn từ và fit surrogate. Không phải giải thích nhân quả hoặc bằng chứng xác minh sự thật.

### Kiểm tra lỗi LIME và giới hạn câu nhập mới (23/10/2026)

LIME giữ xác suất dự đoán đơn tại đúng văn bản gốc khi batch có sai số số học nhỏ (tối đa 0,0002). Các văn bản nhiễu vẫn được suy luận thật; sai số lớn hơn hoặc đổi nhãn sẽ chặn giải thích. Thông số chi tiết hiển thị sai số batch đã quan sát. Không thay checkpoint, calibration, threshold hoặc bảng kết quả nghiên cứu.

Bộ 10 câu người dùng cung cấp cho thấy 13/15 lựa chọn dự đoán tất cả là Fake (5/10 đúng theo nhãn tham chiếu); GCN đúng 6/10, GAT đúng 7/10. Đây là kiểm tra thách thức nhỏ, không phải benchmark ngẫu nhiên. Parity trên mẫu M và thứ tự nhãn đã kiểm chứng; không tự quy kết lỗi này là đảo nhãn. Hiệu năng dataset cao chưa bảo đảm phân biệt đúng các phát biểu y khoa mới. Không dùng các câu này để chỉnh ngưỡng hoặc fit lại mô hình. Bằng chứng: `results/20261023_app_diagnostics_v1/`.

## Giao thức

Mọi mô hình thành viên fit cùng Train nội bộ 3.792 mẫu. Tập 702 mẫu nội bộ dùng chọn epoch; C=642 dùng calibration, M=642 dùng học Ensemble. Neural chọn checkpoint có cả hai absolute gap loss/F1 dưới 0,1. ML dùng cấu hình cố định, không có epoch; báo gap thật sau fit. Ngân sách kiến trúc khác nhau được công bố rõ. Không refit trên 4.494/6.420 mẫu, không multi-seed/5-fold.

Validation loại đúng một mẫu trùng hoàn toàn đã xác minh. Test giữ nguyên 2.140 mẫu từ mirror được duyệt; có hai mẫu trùng sau chuẩn hóa với Train đã biết. Chưa có checksum chính thức chứng nhận mirror tương đương tuyệt đối. Test đã được xem từ trước nên đây là đánh giá lại tập đã biết; tham số khóa trước đánh giá, không tuning bằng Validation/Test. Một seed và gap nhỏ chưa chứng minh ổn định hoặc hết overfitting.

## Kiểm thử

```powershell
& 'C:\Program Files\Python311\python.exe' -B -m unittest discover -s apps/coursework_v1 -p test_unified.py -v
```

Kiểm tra một giao thức, ghép metric đúng Val/Test, parity 11 mô hình trên M, pooling, đầu vào rỗng/quá dài, AppTest nhiều mô hình và LIME thật. Môi trường Windows/GPU hiện tại đã kiểm chứng; chưa kiểm chứng triển khai Transformer trên máy chỉ CPU.

Nguồn: `results/20261022_unified_coursework_v1/`. Registry và checksum: `model_registry.json`, `release.json`. Bảng metric: `metrics_unified.csv`. Dependencies: `requirements.txt`; không tự thay driver hoặc package hệ thống.

T5/TinyLlama chưa có checkpoint thành công nên không xuất hiện như mô hình khả dụng. App là minh họa phân loại văn bản, không phải hệ thống xác minh sự thật y khoa hoặc tư vấn y tế.

## Cập nhật CT-BERT
App có một phương pháp đề xuất là CT-BERT xử lý URL và emoji, với hiệu chỉnh sigmoid. Các baseline và ensemble đối chứng không dùng CT-BERT vẫn có thể chọn. Bảng bổ sung và checkpoint của phương pháp đề xuất được xác minh bằng checksum trong release.json và model_registry.json.
