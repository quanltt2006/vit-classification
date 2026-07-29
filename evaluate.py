import argparse
import os

import torch
import yaml
import matplotlib
matplotlib.use("Agg")  # không cần màn hình hiển thị, chỉ lưu file ảnh
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from model.vit import ViT
from data_loaders.oxford_pet import get_dataloaders
from utils.logger import setup_logger
from utils.metrics import (
    compute_classification_metrics,
    compute_classification_report,
    compute_confusion_matrix,
    compute_macro_roc,
)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(config: dict, device: str) -> ViT:
    """Khởi tạo model với đúng kiến trúc ghi trong config (giống hệt lúc train)."""
    return ViT(
        img_size=config["model"]["img_size"],
        patch_size=config["model"]["patch_size"],
        in_chans=3,
        n_classes=config["model"]["num_classes"],
        embed_dim=config["model"]["embed_dim"],
        depth=config["model"]["depth"],
        n_heads=config["model"]["num_heads"],
    ).to(device)

@torch.no_grad()
def run_inference(model: ViT, loader, device: str):
    """Chạy toàn bộ test set qua model 1 lần, thu thập nhãn thật/dự đoán/xác suất."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    pbar = tqdm(
        loader,
        desc="Inference",
        unit="batch",
        total=len(loader),
        ncols=100
    )

    for images, labels in pbar:
        images = images.to(device)

        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        all_labels.append(labels.numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

        pbar.set_postfix(
            batch_size=images.size(0),
            processed=len(all_labels) * images.size(0)
        )

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    return y_true, y_pred, y_prob


def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: str):
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

def plot_roc_curve(all_fpr: np.ndarray, mean_tpr: np.ndarray, macro_auc: float, save_path: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(all_fpr, mean_tpr, linewidth=2,
            label=f"Macro-average ROC (AUC = {macro_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (one-vs-rest, macro-average)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    device = config["train"]["device"] if torch.cuda.is_available() else "cpu"
    exp_name = config.get("exp_name", "vit_experiment_1")

    logger = setup_logger(log_dir="logs", name=f"{exp_name}_eval")
    out_dir = f"outputs/{exp_name}"
    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model = build_model(config, device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Checkpoint từ epoch {ckpt.get('epoch', '?')}, best_acc lúc train={ckpt.get('best_acc', '?')}")

    # Chỉ cần test_loader — evaluate.py KHÔNG đụng gì tới train/val loader.
    _, _, test_loader = get_dataloaders(config)

    try:
        class_names = test_loader.dataset.classes
    except AttributeError:
        class_names = [str(i) for i in range(config["model"]["num_classes"])]

    logger.info(f"Đang chạy inference trên {len(test_loader.dataset)} ảnh test...")
    y_true, y_pred, y_prob = run_inference(model, test_loader, device)

    # --- Toàn bộ tính toán gọi từ utils/metrics.py ---
    metrics = compute_classification_metrics(y_true, y_pred)

    logger.info("===== KẾT QUẢ TỔNG QUAN =====")
    logger.info(f"Accuracy:            {metrics['accuracy']*100:.2f}%")
    logger.info(
        f"Macro    Precision/Recall/F1: {metrics['precision_macro']*100:.2f}% / "
        f"{metrics['recall_macro']*100:.2f}% / {metrics['f1_macro']*100:.2f}%"
    )
    logger.info(
        f"Weighted Precision/Recall/F1: {metrics['precision_weighted']*100:.2f}% / "
        f"{metrics['recall_weighted']*100:.2f}% / {metrics['f1_weighted']*100:.2f}%"
    )
    if abs(metrics["f1_macro"] - metrics["f1_weighted"]) > 0.05:
        logger.info(
            "⚠️  Macro F1 và Weighted F1 lệch nhau >5% → có class hiếm bị model bỏ rơi, "
            "xem classification_report.txt"
        )

    report = compute_classification_report(y_true, y_pred, class_names)
    report_path = os.path.join(out_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Classification report (theo từng class) lưu tại: {report_path}")

    cm = compute_confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_path)
    logger.info(f"Confusion matrix lưu tại: {cm_path}")

    all_fpr, mean_tpr, macro_auc = compute_macro_roc(y_true, y_prob, config["model"]["num_classes"])
    roc_path = os.path.join(out_dir, "roc_curve.png")
    plot_roc_curve(all_fpr, mean_tpr, macro_auc, roc_path)
    logger.info(f"ROC curve (macro AUC={macro_auc:.3f}) lưu tại: {roc_path}")

    logger.info("Evaluation hoàn tất.")


if __name__ == "__main__":
    main()
