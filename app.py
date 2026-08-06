"""
app.py - Milestone 12: Serve model qua FastAPI.

Chạy:
    uvicorn app:app --reload

Test:
    curl -X POST -F "file=@dog.jpg" http://localhost:8000/predict

Swagger UI tự động (thử API ngay trên trình duyệt, không cần curl):
    http://localhost:8000/docs
"""
import io
from contextlib import asynccontextmanager

import torch
import yaml
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image

from model.factory import build_model
from data_loaders.oxford_pet import get_class_names
from infer import predict_images

# Đổi 2 đường dẫn này theo checkpoint bạn muốn serve
CONFIG_PATH = "configs/vit_timm.yaml"
CHECKPOINT_PATH = "checkpoints/timm_best.pth"

# Nơi lưu model/config sau khi load - dùng dict đơn giản thay vì biến global
# rải rác nhiều chỗ, dễ theo dõi vòng đời của state.
state: dict = {}

# Trang web tối giản: 1 form upload + JS gọi /predict, không cần framework
# frontend riêng, không cần file .html tách biệt - đủ dùng cho demo/portfolio.
HTML_PAGE = """
<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>ViT Pet Classifier</title>
<style>
  body { font-family: sans-serif; max-width: 480px; margin: 40px auto; padding: 0 16px; }
  h2 { margin-bottom: 4px; }
  #result { margin-top: 20px; white-space: pre-wrap; font-size: 15px; }
  img#preview { max-width: 100%; margin-top: 12px; display: none; border-radius: 8px; }
  button { margin-top: 10px; padding: 8px 16px; cursor: pointer; }
</style>
</head>
<body>
  <h2>ViT Pet Classifier</h2>
  <p>Upload 1 ảnh chó/mèo để model đoán giống.</p>
  <input type="file" id="fileInput" accept="image/*">
  <br>
  <button onclick="predict()">Dự đoán</button>
  <img id="preview">
  <div id="result"></div>

<script>
const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const resultDiv = document.getElementById("result");

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (file) {
    preview.src = URL.createObjectURL(file);
    preview.style.display = "block";
  }
  resultDiv.textContent = "";
});

async function predict() {
  const file = fileInput.files[0];
  if (!file) {
    alert("Chọn 1 ảnh trước đã");
    return;
  }
  resultDiv.textContent = "Đang dự đoán...";

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/predict", { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok) {
      resultDiv.textContent = "Lỗi: " + (data.detail || "không rõ nguyên nhân");
      return;
    }

    resultDiv.textContent = data.predictions
      .map(p => `${p.label}: ${(p.confidence * 100).toFixed(1)}%`)
      .join("\\n");
  } catch (err) {
    resultDiv.textContent = "Lỗi kết nối: " + err;
  }
}
</script>
</body>
</html>
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ----- STARTUP: load model đúng 1 LẦN DUY NHẤT -----
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = config["train"]["device"] if torch.cuda.is_available() else "cpu"

    model = build_model(config, device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    state["model"] = model
    state["config"] = config
    state["device"] = device
    state["class_names"] = get_class_names(config)

    print(f"Model đã sẵn sàng - backend={config['model'].get('backend', 'self')}, device={device}")

    yield  # server phục vụ request trong lúc này

    # ----- SHUTDOWN -----
    state.clear()


app = FastAPI(title="ViT Pet Classifier", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def index():
    """Trang web đơn giản để upload ảnh và xem kết quả, không cần Swagger UI."""
    return HTML_PAGE


@app.get("/health")
def health():
    """Kiểm tra server còn sống và model đã load chưa - dùng cho load balancer/monitoring sau này."""
    return {"status": "ok", "device": state.get("device", "chưa load")}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    """
    Nhận 1 ảnh, trả về top-3 giống chó/mèo kèm % tin cậy.

    Cố tình dùng `def` thường, KHÔNG `async def`: torch inference là code
    đồng bộ/blocking (CPU hoặc GPU). Nếu để async def rồi gọi thẳng torch bên
    trong, sẽ chặn event loop của FastAPI, mọi request khác phải xếp hàng chờ.
    Với def thường, FastAPI tự chạy hàm này trong threadpool riêng.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File phải là ảnh (jpg, png, ...)")

    try:
        image_bytes = file.file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Không đọc được ảnh, file có thể bị hỏng")

    try:
        preds = predict_images(
            state["model"], [image], state["config"], state["class_names"],
            state["device"], top_k=3,
        )[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi predict: {e}")

    return {
        "predictions": [
            {"label": label, "confidence": round(conf, 4)} for label, conf in preds
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)