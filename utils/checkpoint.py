import os
import torch


def save_checkpoint(state: dict, checkpoint_dir: str, filename: str = "last.pth"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)
    return path


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    """
    Load checkpoint và khôi phục model/optimizer/scheduler.
    Trả về (start_epoch, best_acc) để vòng lặp train tiếp tục đúng chỗ.
    """
    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    start_epoch = ckpt.get("epoch", -1) + 1
    best_acc = ckpt.get("best_acc", 0.0)
    return start_epoch, best_acc
