# Bộ hình hiện hành — một giao thức thống nhất

Hình 03, 04, 05 bằng tiếng Việt; các hình khác bằng tiếng Anh. Tất cả 650 DPI. Sơ đồ 03–05 được vẽ bằng Matplotlib, không sử dụng ảnh AI. Hình 1 và PPT chờ lệnh riêng.

- `02_du_lieu_va_do_dai.png`
- `03_phan_tach_du_lieu_va_leakage.png`
- `04_tien_xu_ly_den_xac_suat.png`
- `05_cac_nhom_mo_hinh.png`
- `06_train_dev_loss_f1_01.png`
- `06_train_dev_loss_f1_02.png`
- `06_train_dev_loss_f1_03.png`
- `07_ket_qua_validation.png`
- `08_validation_va_test.png`
- `09_confusion_Val_01.png`
- `09_confusion_Val_02.png`
- `09_confusion_Val_03.png`
- `09_confusion_Val_04.png`
- `09_confusion_Val_05.png`
- `09_confusion_Val_06.png`
- `10_confusion_Test_01.png`
- `10_confusion_Test_02.png`
- `10_confusion_Test_03.png`
- `10_confusion_Test_04.png`
- `10_confusion_Test_05.png`
- `10_confusion_Test_06.png`
- `11_calibration_metrics.png`
- `11_reliability_01.png`
- `11_reliability_02.png`
- `12_ensemble_va_choquet.png`
- `13_xai_lime.png`
- `14_streamlit_demo.png`

Hình 06: lịch sử thật của bảy mô hình neural, Y từ 0–1 và đường Selected epoch. ML không có đường epoch giả. Hình 07/08: cùng 15 lựa chọn chính như app. Hình 09/10: đủ 26 cấu hình đã khóa trên Validation/Test, gồm các đối chứng không đưa hết vào app. Hình 11: xác suất sau calibration; không tự coi chênh lệch raw/cal là đã được kiểm chứng trên tập cuối. Hình 13: LIME thật trên một mẫu M; không là giải thích nhân quả. Hình 14: sơ đồ chức năng, không phải screenshot. Test đã được xem từ trước, không gọi là Test độc lập mới.

## Bổ sung cho CT-BERT
- `06_train_dev_loss_f1_04_ctbert.png` — Loss và F1 theo epoch, có đường checkpoint được chọn.
- `08_validation_va_test_ctbert.png` — Bốn cấu hình CT-BERT trên Validation và Test mirror.
- `09_confusion_ctbert.png` và `10_confusion_ctbert.png` — Ma trận nhầm lẫn cho từng tập.
- `11_reliability_ctbert.png` — Đường reliability cho xác suất Real.
Checksum các hình bổ sung nằm trong manifest của đợt CT-BERT.
