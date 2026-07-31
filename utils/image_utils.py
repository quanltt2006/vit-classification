from pathlib import Path

import torch
from PIL import Image

from data_loaders.oxford_pet import build_eval_transform

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def load_image(path: str):
    return Image.open(path).convert('RGB')


def preprocess_image(image: Image.Image, img_size: int) -> torch.Tensor:
    transform = build_eval_transform(img_size)
    tensor = transform(image)
    return tensor.unsqueeze(0)

def find_images_in_folder(folder: str) -> list:
    folder_path = Path(folder)
    return sorted([
        str(p) for p in folder_path.iterdir()
        if p.suffix.lower() in VALID_EXTENSIONS
    ])


