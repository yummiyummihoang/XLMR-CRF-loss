# XLM-RoBERTa base + CRF + Hierarchical SupCon

Notebook huấn luyện và đánh giá nhận dạng thực thể tiếng Việt (NER) trên PAP_NER, sử dụng **`xlm-roberta-base` + CRF + Hierarchical Supervised Contrastive Loss**. Cấu hình mặc định hướng tới Vast.ai với GPU RTX 3090 24 GB.

- [Mở notebook](baselineHierarchicalSupCon_XLMR_Base_Vast_RTX3090.ipynb)
- [Tải model/checkpoint trên Google Drive](https://drive.google.com/drive/folders/1tJ7miHhjHn3odFpXaUSnG7YyKooH7Ek8?usp=sharing)
- [Dataset trên Hugging Face](https://huggingface.co/datasets/levandai1704/PAP_NER_dedup_70_15_15)

## Kiến trúc và loss

Luồng xử lý: tokenizer XLM-R → encoder XLM-R → căn biểu diễn subword về từ → dropout + linear → CRF → giải mã Viterbi thành chuỗi nhãn BIO.

Nhánh SupCon đưa biểu diễn thực thể qua projection head `768 → 256 → 128`. Chỉ các vị trí từ hợp lệ có nhãn khác `O` được đưa vào contrastive loss. Với chế độ hierarchical:

- Cặp có cùng nhãn BIO, ví dụ `B-CQ` và `B-CQ`, có trọng số dương bằng 1.
- Cặp có cùng loại thực thể nhưng khác tiền tố BIO, ví dụ `B-CQ` và `I-CQ`, có trọng số dương bằng `alpha = 0.3`.
- Mỗi biểu diễn không được ghép dương với chính nó.

```text
L_total = L_CRF + lambda_supcon × L_SupCon
lambda_supcon = 0.3; temperature = 0.07; alpha = 0.3
```

Notebook còn chứa lớp `PhoBertCrf` và một số tiện ích từ baseline gốc. Luồng train/test hiện tại chọn **`XLMRobertaCrf`**, còn dự đoán mẫu dùng `XLMRobertaViTagger`; backbone được tải là **`xlm-roberta-base`**. Lớp PhoBERT cũ không được khởi tạo trong các cell chạy chính.

## Dữ liệu

Train, dev và test đều dùng dataset `levandai1704/PAP_NER_dedup_70_15_15`, cố định revision:

```text
e4af303d22f878e5827c94861f50f747a6b3ed45
```

| Split | Tên file được notebook yêu cầu | Tên file sau khi tải |
| --- | --- | --- |
| Train | `train_data_dedup_70_15_15.txt` | `train_data.txt` |
| Dev | `dev_data_dedup_70_15_15.txt` | `dev_data.txt` |
| Test | `test_data_dedup_70_15_15.txt` | `test_data.txt` |

Notebook kiểm tra tên file trong revision trước khi tải. Train/dev được tải ở Bước 4; test được tải ở Bước 6, sau huấn luyện. Dữ liệu dùng định dạng CoNLL: mỗi dòng là một từ, cột đầu chứa từ, cột thứ tư chứa nhãn NER, dòng trống phân cách câu.

Bộ nhãn gồm `O` và các cặp `B-`/`I-` cho `CQ`, `ĐT`, `VBPL`, `NG`, `SL`. `B-` đánh dấu bắt đầu thực thể, `I-` đánh dấu phần tiếp theo, `O` là ngoài thực thể.

## Chạy huấn luyện

1. Upload notebook lên Jupyter của Vast.ai có GPU CUDA, ví dụ RTX 3090 24 GB.
2. Chạy các cell theo thứ tự. Bước 2 cài dependencies; Bước 3 nạp mã nguồn; Bước 3.1 kiểm tra forward bằng mô hình nhỏ; Bước 4 tải train/dev.
3. Chạy Bước 5 để huấn luyện. Bước 6 tải test và đánh giá checkpoint tốt nhất; Bước 7 in các metric entity-level; Bước 8 dự đoán câu mẫu đã tách từ.

Cần kết nối mạng để cài thư viện, tải tokenizer/backbone và dataset. Mã nguồn train/loss được nhúng trực tiếp trong notebook, không cần upload repo phụ. Ví dụ dự đoán nhận câu đã tách từ, nối các tiếng trong từ ghép bằng `_`.

| Tham số | Giá trị mặc định ở cell train |
| --- | --- |
| Backbone | `xlm-roberta-base` |
| Độ dài đầu vào tối đa | 256 subword tokens |
| Train / eval batch size | 128 / 128 |
| Số epoch tối đa | 20 |
| Learning rate encoder | `2e-5` |
| Learning rate classifier, CRF, projection head | `1e-3` |
| Warmup | 0.1 tổng số training steps |
| Weight decay | 0.01 |
| Adam epsilon | `1e-6` |
| Gradient clipping | 1.0 |
| Early stopping | 3 epoch liên tiếp không tăng dev BIO-Macro-F1 |
| Seed | 42 |
| Mixed precision / gradient checkpointing | FP16 / bật |

Nếu thiếu VRAM, giảm `--train_batch_size` và `--eval_batch_size` trong cell train. Có thể tăng `--gradient_accumulation_steps` để bù batch hiệu dụng khi giảm train batch size; việc này thay đổi cấu hình chạy so với mặc định.

Thư mục làm việc mặc định là `/workspace/xlmr_base_hsupcon` nếu có `/workspace`, hoặc `xlmr_base_hsupcon` dưới thư mục hiện tại. Có thể đặt biến môi trường `NER_WORK_ROOT` trước Bước 3 để đổi vị trí này. Các đường dẫn bên dưới tính từ thư mục làm việc:

```text
outputs/xlmr_base_crf_hierarchical_supcon_vast_3090/
├── best_model.pt
├── train.log
└── confusion_matrix.png  # tạo khi đánh giá test
```

`best_model.pt` lưu `model` (state dict), `args` và `classes`. Checkpoint tốt nhất được chọn theo **BIO-Macro-F1 trên dev**; log train được ghi ra cell và `train.log`.

## Dùng model từ Google Drive, không huấn luyện lại

1. Tải checkpoint tương ứng từ [thư mục Drive](https://drive.google.com/drive/folders/1tJ7miHhjHn3odFpXaUSnG7YyKooH7Ek8?usp=sharing), rồi upload vào máy chạy notebook. Checkpoint cần đúng định dạng `best_model.pt` nêu trên và được huấn luyện với kiến trúc/bộ nhãn tương ứng.
2. Chạy Bước 2, Bước 3, Bước 3.1 và Bước 4 để khởi tạo môi trường, mã nguồn và các biến cấu hình. **Bỏ qua cell train ở Bước 5.**
3. Trước Bước 6, chép checkpoint vào vị trí notebook đang dùng:

```python
from pathlib import Path
import shutil

uploaded_checkpoint = Path("/duong/dan/da/upload/best_model.pt")  # sửa đường dẫn
checkpoint_path = Path(OUTPUT_DIR) / "best_model.pt"
checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
if uploaded_checkpoint.resolve() != checkpoint_path.resolve():
    shutil.copyfile(uploaded_checkpoint, checkpoint_path)
assert checkpoint_path.is_file(), checkpoint_path
```

4. Chạy Bước 6 để tải đúng test và nạp model, Bước 7 để tính metric, Bước 8 để dự đoán mẫu.

Link Drive là nơi tải model do chủ repo cung cấp. Repo không chứa trọng số, và không có số đo hiệu năng được suy đoán từ checkpoint.

## Các metric được báo cáo

### Token-level: log train/dev và classification report

Mỗi từ hợp lệ được xem là một mẫu phân loại BIO. Padding không được tính.

| Metric | Ý nghĩa |
| --- | --- |
| BIO-Accuracy | Tỷ lệ từ có nhãn BIO dự đoán đúng, bao gồm `O`. |
| Precision của một nhãn | Số từ dự đoán đúng nhãn đó chia cho tổng số từ được dự đoán mang nhãn đó. |
| Recall của một nhãn | Số từ dự đoán đúng nhãn đó chia cho tổng số từ thực sự mang nhãn đó. |
| F1 của một nhãn | `2 × precision × recall / (precision + recall)`; bằng 0 nếu mẫu số bằng 0. |
| BIO-Macro-F1 | Trung bình F1 giữa các nhãn BIO có trong classification report, bao gồm `O` khi xuất hiện. Đây là metric chọn `best_model.pt`. |
| Weighted avg | Trung bình metric của các nhãn, có trọng số theo số từ gold của từng nhãn (`support`). |
| Confusion matrix | Hàng là nhãn gold, cột là nhãn dự đoán; mỗi hàng được chuẩn hóa theo tổng số mẫu của nhãn gold đó. |

Token-level F1 và entity-level F1 đo hai đơn vị khác nhau. Nhãn `O` thường nhiều, nên accuracy cao chưa đủ để kết luận khả năng nhận diện đúng toàn bộ thực thể.

### Entity-level: cell đánh giá paper với `nervaluate`

Cell Bước 7 giữ ranh giới câu, chuyển chuỗi BIO thành các span thực thể và in Precision/Recall/F1 cho bốn tiêu chí:

| Tiêu chí | Điều kiện khớp |
| --- | --- |
| **Strict** | Đúng cả ranh giới bắt đầu/kết thúc và loại thực thể. |
| **Exact** | Đúng hoàn toàn ranh giới; không xét loại khi tính overall. |
| **Partial** | Không xét loại khi tính overall; đúng ranh giới nhận đủ điểm, chồng lấn một phần nhận nửa điểm. |
| **Entity Type** (`ent_type`) | Đúng loại và có chồng lấn; không bắt buộc đúng hoàn toàn ranh giới. |

Ví dụ gold là `[Ủy ban nhân dân]_CQ`: dự đoán đúng span nhưng gán loại khác được tính đúng theo Exact/Partial overall, nhưng sai theo Strict/Entity Type. Dự đoán span `[Ủy ban]_CQ` được điểm một phần theo Partial và có thể được tính đúng theo Entity Type; Strict/Exact yêu cầu đủ ranh giới.

Các cột đếm trong bảng chi tiết:

| Cột | Ý nghĩa |
| --- | --- |
| `COR` | Thực thể khớp đúng theo tiêu chí đang xét. |
| `INC` | Có ghép khớp/chồng lấn nhưng sai yêu cầu của tiêu chí. |
| `PAR` | Chồng lấn một phần được chấp nhận theo tiêu chí Partial. |
| `MIS` | Thực thể gold bị bỏ sót. |
| `SPU` | Thực thể dự đoán thừa, không ghép được với gold. |
| `POS` | Tổng thực thể gold: `COR + INC + PAR + MIS`. |
| `ACT` | Tổng thực thể dự đoán: `COR + INC + PAR + SPU`. |

Với Strict/Exact, `precision = COR / ACT`, `recall = COR / POS`. Partial dùng `COR + 0.5 × PAR` ở tử số. Entity Type dùng cách tính của thư viện cho `ent_type` (cũng hỗ trợ thành phần `0.5 × PAR`; các cặp đúng loại và chồng lấn được tính vào `COR`). Mẫu số bằng 0 trả về 0. F1 là trung bình điều hòa của precision và recall.

- **Micro**: lấy kết quả overall của evaluator, gộp các thực thể trên toàn bộ tập đánh giá.
- **Macro**: tính riêng từng metric Precision/Recall/F1 cho mỗi loại `CQ`, `ĐT`, `VBPL`, `NG`, `SL`, sau đó lấy trung bình đều trên 5 loại. Loại không xuất hiện ở cả gold và prediction nhận điểm 0. Macro-F1 là trung bình các F1 theo loại, không phải F1 tính từ macro-precision và macro-recall.
- Bảng per-type do `nervaluate` tính sau khi lọc từng loại. Vì vậy Exact/Partial per-type và macro vẫn phản ánh việc phân nhóm theo loại, dù Exact/Partial overall bỏ qua loại khi ghép span.

Kết quả nằm trong hai DataFrame `summary` và `details`. Có thể lưu sau cell metrics:

```python
summary.to_csv(Path(OUTPUT_DIR) / "entity_metrics_summary.csv", index=False)
details.to_csv(Path(OUTPUT_DIR) / "entity_metrics_by_type.csv", index=False)
```

**Phạm vi đánh giá:** đầu vào được truncate tại 256 subword tokens; bộ căn nhãn chỉ giữ các từ đầy đủ trước điểm cắt. Các metric phản ánh phần từ được giữ lại, không bao gồm phần câu bị cắt. Khi so sánh thực nghiệm, cần giữ cùng split, revision, cách cắt câu/độ dài và phiên bản thư viện.

Cell test thông thường còn in báo cáo BIO và kết quả CoNLL kế thừa baseline. Dùng bảng entity-level ở Bước 7 khi báo cáo bốn tiêu chí và micro/macro vì cell này giữ riêng từng câu.

## Các lỗi đã xử lý trong notebook

- Bước 6 tải file test của **cùng dataset và revision với train/dev**, lưu dưới tên `test_data.txt` trước khi gọi `test()`.
- Cell metrics đọc được dictionary kết quả của `nervaluate` mới và tuple của API cũ; chuyển các đối tượng metric thành dictionary trước khi truy cập các trường.
- Fast tokenizer và offset mapping được dùng để căn nhãn từ với subword XLM-R.

Notebook được kiểm tra cấu trúc và cú pháp; các nhánh chuyển đổi API metrics được kiểm tra bằng dữ liệu giả lập. Repo chưa kèm kết quả benchmark được xác minh từ một lần chạy GPU đầy đủ.

## Tham khảo

- [XLM-RoBERTa base](https://huggingface.co/FacebookAI/xlm-roberta-base)
- [PAP_NER dedup 70/15/15](https://huggingface.co/datasets/levandai1704/PAP_NER_dedup_70_15_15)
- [nervaluate](https://github.com/MantisAI/nervaluate) — [các tiêu chí ghép thực thể](https://github.com/MantisAI/nervaluate/blob/main/src/nervaluate/strategies.py), [công thức tính metric](https://github.com/MantisAI/nervaluate/blob/main/src/nervaluate/entities.py).
- Mã CRF/SupCon, trainer và tiện ích trong notebook kế thừa baseline `PhoBERT-Legal-NER/vphoberttagger`; phần chọn backbone, tokenizer và căn nhãn được điều chỉnh cho XLM-R.
