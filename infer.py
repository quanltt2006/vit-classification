"""
infer.py - Milestone 7: Inference trên ảnh thật, ngoài dataset.

Chạy:
    # 1 ảnh
    python infer.py --checkpoint checkpoints_tiny/best.pth --config configs/vit_tiny_scratch.yaml --image dog.jpg

    # nhiều ảnh chỉ định
    python infer.py --checkpoint checkpoints_tiny/best.pth --config configs/vit_tiny_scratch.yaml --images dog.jpg cat.jpg

    # cả 1 folder
    python infer.py --checkpoint checkpoints_tiny/best.pth --config configs/vit_tiny_scratch.yaml --folder ./examples/
"""
import argparse

import torch
import torch.nn as nn
import yaml

from model.factory import build_model
from data_loaders.oxford_pet import get_class_names
from utils.preprocessing import get_preprocessing_config
from utils.image_utils import load_image, preprocess_image, find_images_in_folder


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model_for_inference(config: dict, checkpoint_path: str, device: str):
    model = build_model(config, device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()  # BẮT BUỘC: tắt Dropout/BatchNorm hành xử kiểu training
    return model


@torch.no_grad()
def predict_images(model: nn.Module, images: list, config: dict, class_names: list,
                    device: str, top_k: int = 3) -> list:
    """
    Predict trên danh sách PIL.Image ĐÃ CÓ SẴN TRONG RAM (không đọc từ đĩa).
    Đây là hàm lõi, dùng chung bởi predict_batch() (CLI) VÀ app.py (API) -
    tránh viết lại logic inference 2 lần cho 2 giao diện khác nhau.

    Trả về list, mỗi phần tử là [(label, confidence), ...] tương ứng 1 ảnh input.
    """
    prep_config = get_preprocessing_config(config)
    tensors = [preprocess_image(image, prep_config) for image in images]

    batch = torch.cat(tensors, dim=0).to(device)
    logits = model(batch)
    probs = torch.softmax(logits, dim=1)
    top_probs, top_indices = probs.topk(top_k, dim=1)

    results = []
    for i in range(len(images)):
        preds = [
            (class_names[idx.item()], prob.item())
            for prob, idx in zip(top_probs[i], top_indices[i])
        ]
        results.append(preds)

    return results


def predict_batch(model: nn.Module, image_paths: list, config: dict, class_names: list,
                   device: str, top_k: int = 3) -> dict:
    """
    Wrapper cho CLI: đọc ảnh từ đường dẫn file, bỏ qua ảnh lỗi, rồi gọi predict_images().
    Trả về dict: {path: [(label, confidence), ...]}
    """
    images, valid_paths = [], []

    for path in image_paths:
        try:
            images.append(load_image(path))
            valid_paths.append(path)
        except Exception as e:
            print(f"[Bỏ qua] {path}: không đọc được ảnh ({e})")

    if not images:
        return {}

    preds_list = predict_images(model, images, config, class_names, device, top_k=top_k)
    return dict(zip(valid_paths, preds_list))


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