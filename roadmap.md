# Roadmap ViT Pet-Project — Milestone 6 → 13

> Bối cảnh: bạn đã xong M1–M5 (Data Pipeline, ViT from scratch, Training Engine, Config System, Logging). Từ đây project chuyển từ "train được" sang "dùng được, đưa được cho người khác, deploy được" — đúng năng lực một AI Engineer 2026 cần có.

---

## MILESTONE 6 — Evaluation

### Mục tiêu
Tách hoàn toàn việc **đánh giá model** ra khỏi việc **train model**. Một AI engineer giỏi phải đánh giá được model của người khác mà không cần train lại, và phải hiểu accuracy không nói lên toàn bộ câu chuyện.

### Vì sao quan trọng
Trong công việc thật, bạn thường nhận một checkpoint từ đồng nghiệp/model pretrained và phải trả lời: "model này tốt tới đâu, yếu ở lớp nào, có nên deploy không?" — đó chính là công việc của `evaluate.py`.

### Học
- **Accuracy** — tỉ lệ đúng tổng thể, dễ đánh lừa nếu dữ liệu mất cân bằng (imbalanced).
- **Precision / Recall** — với 37 giống chó mèo, hiểu class nào bị nhầm nhiều với class nào.
- **F1-score** — trung bình điều hòa của Precision/Recall, dùng khi cần cân bằng cả hai.
- **Confusion Matrix** — ma trận 37×37, trực quan hóa để thấy cặp giống bị nhầm lẫn (vd 2 giống mèo lông giống nhau).
- **ROC / AUC** — hiểu rằng ROC chuẩn cho binary classification; với multi-class dùng one-vs-rest hoặc macro-average.
- **Macro vs Micro vs Weighted average** — khi có 37 class, cách gộp metric ảnh hưởng lớn tới con số cuối cùng.

### Code cần viết
```
evaluate.py
utils/
  └── metrics.py   (mở rộng file đã có ở M3)
```

`evaluate.py` cần:
- Nhận `--config` và `--checkpoint` qua argparse (KHÔNG cần train trước).
- Load model + checkpoint, load test_loader từ `data_loaders/oxford_pet.py`.
- Chạy inference toàn bộ test set, thu thập `y_true`, `y_pred`, `y_prob`.
- In ra: accuracy, precision/recall/f1 (macro + weighted), classification report theo từng class.
- Vẽ confusion matrix lưu vào `outputs/{exp_name}/confusion_matrix.png`.
- Vẽ ROC curve (macro-average one-vs-rest) lưu vào `outputs/{exp_name}/roc_curve.png`.

Gợi ý thư viện: `sklearn.metrics` (accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report, roc_curve, auc), `matplotlib`/`seaborn` để vẽ.

### Deliverable
```bash
python evaluate.py --config configs/vit_base.yaml --checkpoint checkpoints/best.pth
```
Chạy được **độc lập hoàn toàn** với `train.py`.

### Done checklist
- [ ] Không cần train vẫn evaluate được (chỉ cần checkpoint có sẵn)
- [ ] In được accuracy, precision, recall, f1 (macro + per-class)
- [ ] Xuất được confusion matrix dạng ảnh
- [ ] Xuất được ROC curve
- [ ] Kết quả log lại vào `logs/` giống cách train.py đang làm

### Sai lầm thường gặp
- ❌ Chỉ nhìn accuracy tổng, bỏ qua per-class — với 37 class, có thể model đoán đúng 90% nhưng "chết" hoàn toàn ở 3-4 giống hiếm.
- ❌ Evaluate trên tập train hoặc val thay vì test — con số sẽ lạc quan giả tạo.
- ❌ Copy-paste lại code load data/model từ train.py — nên tái sử dụng qua import, không duplicate logic.

---

## MILESTONE 7 — Inference

### Mục tiêu
Biến model từ "chỉ chạy được trong repo" thành "chạy được với 1 ảnh bất kỳ do người dùng đưa vào" — đây là bước đầu tiên hướng tới sản phẩm thật.

### Học
- Tiền xử lý ảnh input giống hệt lúc train/eval (cùng resize, cùng normalize) — sai chỗ này là lỗi âm thầm rất phổ biến.
- `torch.no_grad()` + `model.eval()` cho inference.
- Softmax để ra xác suất thay vì chỉ logits.
- Batch inference cho folder ảnh (xử lý nhiều ảnh cùng lúc để nhanh hơn).
- Xử lý ảnh lỗi/định dạng lạ mà không crash toàn bộ chương trình.

### Code cần viết
```
infer.py
utils/
  └── image_utils.py   (load ảnh, preprocess dùng chung train/eval/infer)
```

`infer.py` hỗ trợ 3 chế độ qua argparse:
```bash
python infer.py --checkpoint checkpoints/best.pth --image dog.jpg
python infer.py --checkpoint checkpoints/best.pth --images dog.jpg cat.jpg
python infer.py --checkpoint checkpoints/best.pth --folder ./test_images/
```

Output mẫu (đúng format bạn note trong roadmap gốc):
```
dog.jpg
  → Golden Retriever   99.6%
  → Labrador Retriever  0.3%
  → Beagle              0.1%
```

Nên trả về top-k (vd top-3) thay vì chỉ top-1, để thấy được độ tự tin của model.

### Quan trọng: tách preprocessing thành 1 nguồn sự thật duy nhất
Đây là lỗi cực kỳ phổ biến khi lên production: transform lúc infer không khớp transform lúc eval → model "tự nhiên" dự đoán sai hết dù checkpoint không có vấn đề gì. Viết 1 hàm `get_eval_transform(img_size)` dùng chung cho cả `oxford_pet.py`, `evaluate.py`, `infer.py`.

### Deliverable
- [ ] Predict được 1 ảnh
- [ ] Predict được nhiều ảnh cùng lúc
- [ ] Predict được cả 1 folder
- [ ] Output rõ ràng: tên class + % confidence

### Sai lầm thường gặp
- ❌ Viết lại transform riêng cho infer.py, lệch với eval → bug âm thầm khó phát hiện nhất trong cả project.
- ❌ Không convert ảnh về RGB (`Image.open(path).convert("RGB")`) — ảnh PNG có alpha channel hoặc ảnh grayscale sẽ làm model input sai shape.
- ❌ Quên `model.eval()` → Dropout/BatchNorm (nếu có) hoạt động sai, kết quả không ổn định giữa các lần chạy.

---

## MILESTONE 8 — Refactor (Notebook → Production Code)

### Mục tiêu
Đây là milestone bạn tự đánh giá "cực kỳ quan trọng" — đúng vậy. Đây là ranh giới giữa "code chạy được" và "code người khác đọc/maintain được". Kỹ năng này quyết định bạn có được coi là engineer hay chỉ là "người chạy notebook".

### Học
- **Clean Architecture** cho ML project: tách rõ `data` / `model` / `engine` / `configs` / `utils` — bạn đã làm đúng hướng từ M1, giờ siết chặt lại triệt để.
- **Type hints** (`typing`): mọi hàm public phải có type cho input/output.
- **Docstring**: theo chuẩn Google style hoặc NumPy style, nhất quán toàn repo.
- **Single Responsibility**: mỗi file/hàm chỉ làm đúng 1 việc.
- **Dependency injection kiểu đơn giản**: truyền `config` xuống thay vì hardcode, truyền `device` thay vì gọi `.cuda()` khắp nơi.

### Việc cần làm cụ thể
1. Thêm type hint cho toàn bộ hàm trong `model/`, `engine/`, `data_loaders/`, `utils/`.
   ```python
   def get_dataloaders(config: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
       ...
   ```
2. Thêm docstring cho mọi class/hàm public — đặc biệt `ViT`, `Trainer`/`train_one_epoch`, `get_dataloaders`.
3. Refactor `train.py` để **dưới 100 dòng** — bí quyết: chuyển toàn bộ logic (build model, build optimizer, vòng lặp train) vào 1 class `Trainer` trong `engine/trainer.py`, `train.py` chỉ còn phần "lắp ráp" (load config → khởi tạo → gọi `trainer.fit()`).
4. Chạy `mypy` hoặc ít nhất `ruff`/`flake8` để bắt lỗi style tự động.
5. Viết `pyproject.toml` hoặc `setup.cfg` cấu hình linter (không bắt buộc nhưng rất "production").

### Deliverable
- [ ] `train.py` **< 100 dòng**
- [ ] Mọi hàm public có type hint
- [ ] Mọi class/hàm public có docstring
- [ ] `engine/trainer.py` có class `Trainer` gói toàn bộ logic train/validate/save/resume

### Sai lầm thường gặp
- ❌ Refactor xong nhưng behavior thay đổi (bug regression) — luôn chạy lại `evaluate.py` sau refactor để confirm số liệu không đổi.
- ❌ Docstring chỉ ghi lại tên biến bằng lời, không mô tả **tại sao** hàm tồn tại.
- ❌ Giảm số dòng `train.py` bằng cách nhồi nhiều lệnh trên 1 dòng (dùng `;`) — đây là gian lận, không phải refactor.

---

## MILESTONE 9 — timm (Transfer Learning)

### Mục tiêu
Hiểu vì sao ngoài đời gần như không ai train ViT from scratch trên vài nghìn ảnh — bạn vừa tự tay implement ViT (M2) để **hiểu cơ chế**, giờ dùng pretrained backbone để **có kết quả tốt thật sự**.

### Học
- Khái niệm **pretrained weight**, **backbone**, **transfer learning**, **fine-tuning** vs **feature extraction** (freeze backbone, chỉ train head).
- Thư viện `timm` (PyTorch Image Models) — kho hơn 1000 kiến trúc pretrained.
- So sánh: ViT tự viết train từ đầu trên ~7000 ảnh Oxford Pet sẽ overfit/kết quả kém hơn nhiều so với ViT pretrained fine-tune.

### Code cần viết
```
models/
  ├── vit.py          (ViT tự viết, đã có từ M2)
  └── timm_model.py    (wrapper cho timm)
```

```python
import timm

def build_model(config: dict):
    backend = config["model"]["backend"]
    if backend == "self":
        from model.vit import ViT
        return ViT(...)
    elif backend == "timm":
        return timm.create_model(
            config["model"]["timm_name"],       # vd "vit_base_patch16_224"
            pretrained=True,
            num_classes=config["model"]["num_classes"],
        )
```

Cập nhật config:
```yaml
model:
  backend: "timm"        # "self" | "timm"
  timm_name: "vit_base_patch16_224"
```

### Deliverable
- [ ] Đổi `backend: self` ↔ `backend: timm` trong config, `train.py` chạy đúng cả 2 trường hợp không sửa code.
- [ ] So sánh (ghi vào README hoặc report ngắn): accuracy của self-ViT vs timm-ViT trên cùng test set, cùng số epoch.

### Sai lầm thường gặp
- ❌ Fine-tune với learning rate giống hệt lúc train from scratch — pretrained model cần lr nhỏ hơn nhiều (thường 1e-4 → 1e-5), lr lớn sẽ phá vỡ pretrained weight ("catastrophic forgetting").
- ❌ Quên resize ảnh đúng kích thước mà backbone pretrained yêu cầu (một số timm model dùng 224, số khác 384).
- ❌ Không normalize theo đúng mean/std mà backbone được pretrain (thường ImageNet mean/std — bạn đã dùng đúng từ M1, may mắn khớp sẵn).

---

## MILESTONE 10 — Hugging Face

### Mục tiêu
Học hệ sinh thái phổ biến nhất hiện nay bên cạnh timm — `transformers` của Hugging Face — để 1 pipeline của bạn chạy được với **3 backend khác nhau**: tự viết, timm, Hugging Face. Đây là kỹ năng thể hiện rõ khả năng "đọc doc thư viện mới, tích hợp vào hệ thống có sẵn" — thứ nhà tuyển dụng 2026 quan tâm hơn cả việc bạn thuộc bao nhiêu công thức.

### Học
- `AutoImageProcessor` — thay thế cho `transforms.Compose` tự viết, tự động lấy đúng preprocessing config của checkpoint.
- `AutoModelForImageClassification` — load bất kỳ model classification nào từ Hugging Face Hub chỉ bằng tên string.
- Khác biệt input/output format giữa `transformers` và `torchvision`/`timm` (HF thường trả về object có `.logits` thay vì tensor thẳng).

### Code cần viết
```
models/
  └── hf_model.py
```

```python
from transformers import AutoImageProcessor, AutoModelForImageClassification

def build_hf_model(model_name: str, num_classes: int):
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(
        model_name, num_labels=num_classes, ignore_mismatched_sizes=True
    )
    return model, processor
```

Viết 1 lớp adapter để cả 3 backend (`self`, `timm`, `hf`) đều expose cùng interface `forward(x) -> logits` cho `engine/trainer.py` dùng chung, không cần biết đang chạy backend nào.

### Deliverable
- [ ] Config `backend: hf` chạy được qua cùng 1 `train.py`, `evaluate.py`, `infer.py` như 2 backend kia.
- [ ] Bảng so sánh 3 backend: accuracy, số tham số, thời gian train/epoch, thời gian inference 1 ảnh.

### Sai lầm thường gặp
- ❌ Trộn lẫn processor của HF với DataLoader/Dataset tự viết mà không kiểm tra output shape khớp nhau.
- ❌ Quên `ignore_mismatched_sizes=True` khi số class khác với model gốc (thường pretrain trên ImageNet-1000, bạn có 37 class) → lỗi shape mismatch khi load weight.
- ❌ Coi `outputs.logits` giống hệt tensor thường — cần `.logits` trước khi đưa vào loss function.

---

## MILESTONE 11 — Docker

### Mục tiêu
Đảm bảo "chạy được trên máy tôi" → "chạy được ở bất kỳ đâu". Đây là kỹ năng bắt buộc, không phải optional, cho mọi vị trí AI engineer 2026.

### Học
- Khác biệt `CMD` vs `ENTRYPOINT`.
- Multi-stage build để giảm size image (đặc biệt quan trọng với PyTorch — image gốc rất nặng).
- `.dockerignore` để không copy `data/`, `checkpoints/`, `__pycache__` vào image.
- CPU-only vs GPU (CUDA) base image — cân nhắc image nào phù hợp mục đích deploy.

### Code cần viết
```
Dockerfile
.dockerignore
```

Ví dụ khung Dockerfile (CPU, cho mục đích inference/API là chính, train nặng vẫn nên chạy ngoài Docker hoặc trên máy có GPU + base image cuda):
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "infer.py"]
```

`.dockerignore` gợi ý:
```
data/
checkpoints/
outputs/
logs/
__pycache__/
*.pyc
.git/
venv/
```

### Deliverable
```bash
docker build -t vit-classification .
docker run --rm vit-classification
```

### Done checklist
- [ ] `docker build` chạy không lỗi
- [ ] `docker run` chạy được (ít nhất chạy được `infer.py` với 1 ảnh mẫu đóng gói sẵn trong image, hoặc mount volume ảnh từ ngoài vào)
- [ ] Image không chứa `data/` (dataset) hay `checkpoints/` cũ thừa thãi — kiểm tra bằng `docker images` xem size hợp lý chưa

### Sai lầm thường gặp
- ❌ Không có `.dockerignore` → build cực chậm, image cực nặng vì copy cả dataset vào.
- ❌ Cài `torch` không pin version → build hôm nay thành công, build tuần sau lỗi vì torch ra bản mới không tương thích.
- ❌ Hardcode path tuyệt đối kiểu `/home/quan/...` trong code → chạy trong container sẽ lỗi vì path không tồn tại.

---

## MILESTONE 12 — FastAPI

### Mục tiêu
Biến model từ "script chạy tay" thành "service" — bước cuối cùng để model thực sự được người khác (frontend, mobile app, đồng nghiệp) gọi tới qua HTTP.

### Học
- Khái niệm REST endpoint, `POST` request nhận file upload.
- `UploadFile` của FastAPI để nhận ảnh qua multipart/form-data.
- Load model **1 lần duy nhất lúc server khởi động** (không load lại mỗi request — lỗi hiệu năng rất phổ biến của người mới).
- Trả JSON response chuẩn, có status code hợp lý (400 nếu file không phải ảnh, 500 nếu lỗi server).
- (Nên biết thêm) `async def` vs `def` trong FastAPI — khi nào cần async thật sự.

### Code cần viết
```
app.py  (hoặc api/main.py)
```

```python
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io

app = FastAPI(title="ViT Pet Classifier")

model = None  # load 1 lần lúc startup

@app.on_event("startup")
def load_model_once():
    global model
    model = build_model_and_load_checkpoint(...)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File phải là ảnh")

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    result = run_inference(model, image)   # tái sử dụng hàm từ infer.py, KHÔNG viết lại
    return {"predictions": result}
```

Output JSON mẫu:
```json
{
  "predictions": [
    {"label": "Golden Retriever", "confidence": 0.996},
    {"label": "Labrador Retriever", "confidence": 0.003}
  ]
}
```

### Deliverable
```bash
uvicorn app:app --reload
# test:
curl -X POST -F "file=@dog.jpg" http://localhost:8000/predict
```
Upload ảnh → nhận JSON.

### Done checklist
- [ ] `/predict` nhận file ảnh, trả JSON đúng format
- [ ] Model chỉ load 1 lần khi start server, không load lại mỗi request
- [ ] Có validate input (từ chối file không phải ảnh) với status code phù hợp
- [ ] Test thử qua `curl` hoặc Swagger UI tự động (`/docs`)

### Sai lầm thường gặp
- ❌ Load model bên trong hàm `predict()` → mỗi request chậm vài giây không cần thiết, production sẽ sập khi có nhiều request cùng lúc.
- ❌ Không xử lý exception (ảnh lỗi, file rỗng) → server crash thay vì trả lỗi 400/500 gọn gàng.
- ❌ Viết lại toàn bộ logic inference riêng cho API thay vì tái sử dụng `infer.py` — vi phạm đúng nguyên tắc Single Source of Truth đã học ở M8.

---

## MILESTONE 13 — Polish

### Mục tiêu
Milestone quyết định người khác (nhà tuyển dụng, đồng nghiệp) có đánh giá cao project của bạn hay không — code tốt mà không polish thì vẫn trông như project tập tành.

### Việc cần làm

**1. README.md hoàn chỉnh** gồm:
- Mô tả bài toán 1 đoạn ngắn (Oxford-IIIT Pet, 37 giống chó mèo).
- Badge (Python version, License, nếu có CI thì thêm badge CI).
- Sơ đồ kiến trúc / pipeline (ảnh hoặc ASCII diagram).
- Hướng dẫn cài đặt (đã có, giữ nguyên).
- Bảng kết quả: accuracy/F1 của cả 3 backend (self/timm/hf) trên test set.
- Hướng dẫn train / evaluate / infer / chạy Docker / chạy API — mỗi cái 1 lệnh copy-paste chạy được ngay.
- Screenshot demo (vd ảnh confusion matrix, hoặc screenshot gọi API trả kết quả).

**2. LICENSE** — chọn MIT (phổ biến nhất, dễ hiểu) hoặc Apache 2.0 nếu quan tâm patent clause.

**3. `examples/`** — 3-5 ảnh mẫu để người dùng test ngay `infer.py` mà không cần tự tìm ảnh.

**4. Directory structure rõ ràng trong README**, ví dụ:
```
vit-classification/
├── configs/          # cấu hình YAML
├── data_loaders/      # dataset, augmentation
├── model/             # ViT tự viết + timm/HF wrapper
├── engine/             # training loop, Trainer class
├── utils/              # logger, checkpoint, metrics
├── examples/            # ảnh mẫu để test infer.py
├── train.py
├── evaluate.py
├── infer.py
├── app.py               # FastAPI server
├── Dockerfile
└── README.md
```

**5. Citation** — nếu dùng dataset/paper của người khác (Oxford-IIIT Pet, paper ViT gốc "An Image is Worth 16x16 Words"), trích dẫn đúng chuẩn trong README:
```
@article{dosovitskiy2020vit,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and others},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}
```

### Deliverable / bài test cuối cùng
> Một người **chưa từng gặp bạn**, chỉ có link repo, `git clone` về và chạy được — không cần hỏi bạn thêm bất kỳ câu nào.

### Done checklist
- [ ] README có đủ: mô tả, cài đặt, kết quả, hướng dẫn chạy từng phần
- [ ] LICENSE tồn tại
- [ ] `examples/` có ảnh mẫu, `infer.py` chạy được ngay với ảnh đó
- [ ] Directory structure được document
- [ ] Citation cho dataset + paper gốc
- [ ] Tự thử nghiệm: xóa hết local, `git clone` lại từ đầu vào thư mục mới, làm đúng theo README, xác nhận chạy được 100%

### Sai lầm thường gặp
- ❌ README viết cho chính mình (giả định người đọc đã biết ngữ cảnh) thay vì viết cho người lạ hoàn toàn.
- ❌ Không test lại bằng clone sạch — rất nhiều project "chạy được" chỉ vì máy bạn có sẵn file/env cũ mà repo không hề chứa.
- ❌ Quên `.gitignore` dữ liệu/checkpoint nặng → repo nặng hàng GB, không ai muốn clone.

---

## Tổng kết lộ trình

| Milestone | Trọng tâm | Kỹ năng AI Engineer tương ứng |
|---|---|---|
| 6 | Evaluation | Đánh giá model độc lập, hiểu metric ngoài accuracy |
| 7 | Inference | Đưa model ra khỏi notebook, phục vụ input thật |
| 8 | Refactor | Viết code production, không chỉ code chạy được |
| 9 | timm | Transfer learning, tận dụng pretrained ecosystem |
| 10 | Hugging Face | Tích hợp thư viện phổ biến nhất hiện nay |
| 11 | Docker | Đóng gói, chạy nhất quán mọi môi trường |
| 12 | FastAPI | Biến model thành service thực sự |
| 13 | Polish | Giao tiếp qua code — kỹ năng bị đánh giá thấp nhưng quyết định ấn tượng đầu tiên |

Sau M13, project này đã đủ chất lượng để đưa vào CV/portfolio như một pet-project hoàn chỉnh, thể hiện toàn bộ vòng đời ML: data → model → train → eval → serve → deploy → document.