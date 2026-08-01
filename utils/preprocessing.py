"""
utils/preprocessing.py - Milestone 10

Xác định img_size/mean/std ĐÚNG cho từng backend:
- self / timm: dùng chuẩn ImageNet cố định.
- hf: lấy trực tiếp từ AutoImageProcessor của chính model đó, vì mỗi model
  Hugging Face có thể được pretrain với preprocessing khác nhau.

Đây là nguồn duy nhất trả lời "tiền xử lý đúng cho model này là gì", dùng chung
bởi data_loaders/oxford_pet.py (train/val/test) VÀ utils/image_utils.py (infer.py).
Không nơi nào khác được tự quyết định mean/std/img_size.
"""

import transformers

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_preprocessing_config(config: dict) -> dict:
    backend = config["model"].get("backend", "self")

    if backend == "hf":
        from transformers import AutoImageProcessor

        processor = AutoImageProcessor.from_pretrained(config["model"]["hf_name"])
        size = processor.size.get("height") or processor.size.get("shortest_edge")
        return {
            "img_size": size or config["model"]["img_size"],
            "mean": processor.image_mean,
            "std": processor.image_std,
        }

    return {
        "img_size": config["model"]["img_size"],
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    }