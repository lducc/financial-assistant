# Hệ Thống Tìm Kiếm Tài Liệu Tài Chính - ViFinQA (R2AI2026 Compliant Document Retrieval)

Dự án này là một hệ thống tìm kiếm tài liệu (Document Retrieval) hoạt động hoàn toàn bằng tập luật (rule-only), được thiết kế để tìm kiếm chính xác các báo cáo tài chính (báo cáo riêng/báo cáo hợp nhất) phù hợp cho từng câu hỏi trong bộ dữ liệu ViFinQA.

Hệ thống được thiết kế để hoạt động độc lập, có thể tái lập kết quả 100%, không sử dụng các nhãn dự đoán từ các phiên bản trước hoặc các API mô hình ngôn ngữ lớn đóng (closed LLM).

> [!NOTE]
> Gói submission tạo ra từ dự án này cố ý để trống các trường `answer`, `relevant_tables`, `evidence`, và `pandas_query`. Mục tiêu duy nhất của dự án này là tối ưu hóa và đo lường độ chính xác tìm kiếm tài liệu (Document Precision, Recall, và MRR).

---

## 🛠️ Tính Năng Nổi Bật của Hệ Thống

1. **Nhận diện thực thể thông minh (Entity Detection):** Gộp kết quả từ tất cả các bộ nhận diện thực thể thay vì chỉ lấy mã cổ phiếu hoặc tên viết tắt (alias) đầu tiên tìm thấy.
2. **Hỗ trợ tên lịch sử & thương hiệu:** Tích hợp danh sách tên lịch sử và thương hiệu của các công ty đã được đối chiếu và xác minh trực tiếp từ các báo cáo tài chính thực tế.
3. **Tránh nhận diện sai lệch:** Ưu tiên ánh xạ tên cụ thể hơn là các token thương hiệu trùng với mã cổ phiếu khác (ví dụ: cụm từ `"Chứng khoán FPT"` sẽ ánh xạ chính xác đến mã `FTS` thay vì tập đoàn mẹ `FPT`).
4. **Xử lý ngữ cảnh đối tác:** Loại trừ các công ty liên kết hoặc đối tác được nhắc tới trong câu hỏi mà không làm mất đi mã cổ phiếu của công ty chủ thể cần truy vấn.
5. **Độ bao phủ theo năm:** Tạo cấu trúc truy vấn chính xác cho từng cặp `(mã cổ phiếu, năm)` được yêu cầu.
6. **Ưu tiên phạm vi báo cáo (Scope):** Tự động giải quyết phạm vi báo cáo theo thứ tự ưu tiên: Khớp phạm vi chính xác (riêng/hợp nhất) -> Phạm vi không xác định -> Phạm vi ngược lại.
7. **Hỗ trợ phạm vi hỗn hợp (Mixed Scope):** Nhận diện và xử lý các câu hỏi yêu cầu cả báo cáo riêng và báo cáo hợp nhất trên các năm khác nhau trong cùng một câu hỏi.

---

## 📁 Cấu Trúc Thư Mục Dự Án

```text
├── src/
│   └── compliant_docs/       # Mã nguồn chính xử lý trích xuất thực thể, chuẩn hóa và truy vấn
│       ├── __init__.py
│       ├── aliases.py        # Định nghĩa và chuẩn hóa tên viết tắt, thương hiệu, tên lịch sử
│       ├── catalog.py        # Quản lý danh mục báo cáo tài chính và danh sách doanh nghiệp
│       ├── normalize.py      # Tiền xử lý văn bản câu hỏi tiếng Việt (bỏ dấu, viết thường)
│       ├── ollama.py         # Module tùy chọn hỗ trợ gọi mô hình cục bộ
│       ├── parser.py         # Trích xuất mã cổ phiếu, năm, và phạm vi báo cáo từ câu hỏi
│       ├── retriever.py      # Logic tìm kiếm tài liệu phù hợp từ cơ sở dữ liệu
│       └── submission.py     # Tạo gói kết quả submission.zip theo định dạng chuẩn
├── tests/
│   └── test_pipeline.py      # Các bộ kiểm thử tự động (pytest)
├── sources/
│   └── pseudo_gt_21_filtered_rules.txt  # File phân tích luật lọc từ dữ liệu
├── output/                   # Thư mục lưu kết quả đầu ra (chứa file .gitkeep để theo dõi)
├── requirements.txt          # Các thư viện phụ thuộc
├── run.py                    # Script chạy toàn bộ pipeline chính
└── README.md                 # Tài liệu hướng dẫn sử dụng
```

---

## 🚀 Hướng Dẫn Cài Đặt và Chạy

### 1. Chuẩn bị môi trường & Dữ liệu
Dự án yêu cầu cài đặt Python 3.11+ và thư viện phụ thuộc:
```powershell
pip install -r requirements.txt
```

Hệ thống giả định cấu trúc dữ liệu của cuộc thi `ViFinQA_data` nằm ở thư mục cùng cấp với thư mục dự án này:
```text
├── ViFinQA_data/             # Thư mục dữ liệu (nằm ngoài repo Git này)
│   ├── code_stock.csv        # Danh sách mã cổ phiếu và tên doanh nghiệp
│   ├── financial_statements/ # Thư mục chứa các báo cáo tài chính dạng PDF/Text
│   └── questions/
│       └── questions.jsonl   # File chứa câu hỏi kiểm tra
└── [Thư mục dự án này]/     # Chứa mã nguồn dự án (run.py, src, tests...)
```

### 2. Chạy Kiểm Thử (Tests)
Để đảm bảo các quy tắc trích xuất hoạt động chính xác và không bị lỗi rollback (regression):
```powershell
python -m pytest tests -q
```

### 3. Chạy Pipeline tạo file Submission
Chạy lệnh sau để quét toàn bộ câu hỏi và tạo gói kết quả tìm kiếm tài liệu tài chính:
```powershell
python run.py --no-llm --full-year-coverage
```
* **`--no-llm`**: Chạy thuần tập luật (rule-based), không cần kết nối API hoặc chạy mô hình bên ngoài (độ chính xác và tính tái lập đạt 100%).
* **`--full-year-coverage`**: Tối ưu hóa việc tìm kiếm báo cáo cho tất cả các năm được truy vấn.

Kết quả cuối cùng sẽ được ghi vào file: `output/submission.zip`.

---

## 🛠️ Sử Dụng Mô Hình LLM Cục Bộ (Tùy chọn)
Nếu muốn kích hoạt khả năng phân tích các thực thể bị mơ hồ (ambiguous) bằng mô hình ngôn ngữ lớn cục bộ:
1. Đảm bảo bạn đã cài đặt [Ollama](https://ollama.com/) và đã tải mô hình:
   ```powershell
   ollama pull qwen2.5:7b
   ```
2. Chạy pipeline bỏ tham số `--no-llm`:
   ```powershell
   python run.py --full-year-coverage
   ```
Hệ thống sẽ chỉ gọi `qwen2.5:7b` cho các câu hỏi mà tập luật phát hiện ra các thực thể bị mơ hồ và không thể tự giải quyết. Kết quả phản hồi sẽ được lưu đệm (cache) trong file `output/llm_cache.jsonl` để tránh gọi lại nhiều lần.

---

## 📊 Kết Quả và Dữ Liệu Đầu Ra (Artifacts)
Sau khi pipeline hoàn thành, các file báo cáo chi tiết sẽ xuất hiện trong thư mục `output/`:
* `output/diagnostics.jsonl`: Nhật ký chi tiết cho từng câu hỏi (chứa mã cổ phiếu nhận diện, các năm, phạm vi báo cáo phát hiện được và tài liệu được chọn).
* `output/report_catalog.json`: Danh mục các báo cáo hợp lệ được quét từ thư mục dữ liệu gốc.
* `output/alias_evidence.json`: Đường dẫn tài liệu thực tế chứng minh cho việc ánh xạ các tên viết tắt (alias) thủ công.
* `output/provenance.json`: Metadata lưu cấu hình, thông số, mã băm (SHA256) của câu hỏi và dữ liệu đầu vào phục vụ kiểm tra tính chính xác.