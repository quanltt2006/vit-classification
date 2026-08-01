import argparse

import torch
import yaml

from model.vit import ViT
from data_loaders.oxford_pet import get_class_names
from utils.image_utils import load_image, preprocess_image, find_images_in_folder
from model.factory import build_model
import torch.nn as nn
from utils.preprocessing import get_preprocessing_config

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)




def load_model_for_inference(config: dict, checkpoint_path: str, device: str) -> ViT:
    model = build_model(config, device)

    ckpt = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"])

    model.eval()

    return model

@torch.no_grad
def predict_batch(model: nn.Module, image_paths: list, config: dict, class_names: list,
                   device: str, top_k: int = 3) -> dict:

    tensors, valid_paths = [], []
    prep_config = get_preprocessing_config(config)


    for path in image_paths:
        try:
            image = load_image(path)
            tensor = preprocess_image(image, prep_config)
            tensors.append(tensor)
            valid_paths.append(path)
        except Exception as e: 
            print(f"[Bỏ qua] {path}: không đọc được ảnh ({e})")

    if not tensors: 
        return {}

    batch = torch.cat(tensors, dim = 0 ).to(device)
    logits = model(batch)
    probs = torch.softmax(logits, dim = 1)

    top_probs, top_indices = probs.topk(top_k, dim = 1 )

    results = {}


    for i, path in enumerate(valid_paths):
        preds = [
            (class_names[idx.item()], prob.item())
            for prob, idx in zip(top_probs[i], top_indices[i])
        ]
        results[path] = preds

    return results
def print_results(results: dict):
    for path, preds in results.items():
        print(f"\n{path}")
        for label, conf in preds:
            print(f"  → {label:<30s} {conf*100:5.1f}%")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, help="Đường dẫn 1 ảnh")
    parser.add_argument("--images", type=str, nargs="+", help="Nhiều ảnh, cách nhau bởi dấu cách")
    parser.add_argument("--folder", type=str, help="Folder chứa ảnh")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    if not any([args.image, args.images, args.folder]):
        parser.error("Cần truyền 1 trong 3: --image, --images, hoặc --folder")

    config = load_config(args.config)
    device = config["train"]["device"] if torch.cuda.is_available() else "cpu"

    model = load_model_for_inference(config, args.checkpoint, device)
    class_names = get_class_names(config)

    if args.image:
        image_paths = [args.image]
    elif args.images:
        image_paths = args.images
    else:
        image_paths = find_images_in_folder(args.folder)
        if not image_paths:
            print(f"Không tìm thấy ảnh hợp lệ trong {args.folder}")
            return

    results = predict_batch(model, image_paths, config, class_names, device, top_k=args.top_k)
    print_results(results)


if __name__ == "__main__":
    main()