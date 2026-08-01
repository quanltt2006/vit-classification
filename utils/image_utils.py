from pathlib import Path

import torch
from PIL import Image

from data_loaders.oxford_pet import build_eval_transform

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def load_image(path: str):
    return Image.open(path).convert('RGB')


def preprocess_image(image: Image.Image, prep_config: dict) -> torch.Tensor:
    """Trả về tensor [1, 3, H, W] sẵn sàng đưa vào model. prep_config từ get_preprocessing_config()."""
    transform = build_eval_transform(prep_config["img_size"], prep_config["mean"], prep_config["std"])
    tensor = transform(image)
    return tensor.unsqueeze(0)

def find_images_in_folder(folder: str) -> list:
    folder_path = Path(folder)
    return sorted([
        str(p) for p in folder_path.iterdir()
        if p.suffix.lower() in VALID_EXTENSIONS
    ])


