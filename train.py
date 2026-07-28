import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter

from model.vit import ViT
from data_loaders.oxford_pet import get_dataloaders
from engine.trainer import train_one_epoch, evaluate
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.logger import setup_logger


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_scheduler(optimizer, config):
    sch_cfg = config["train"].get("scheduler", {"type": "none"})
    sch_type = sch_cfg.get("type", "none")

    if sch_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["train"]["epochs"]
        )
    if sch_type == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sch_cfg.get("step_size", 10),
            gamma=sch_cfg.get("gamma", 0.1),
        )
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/vit_base.yaml")
    parser.add_argument(
        "--resume", type=str, default=None, help="Path tới checkpoint để resume"
    )
    args = parser.parse_args()
    config = load_config(args.config)

    device = config["train"]["device"]
    exp_name = config.get("exp_name", "vit_experiment_1")

    logger = setup_logger(log_dir="logs", name=exp_name)
    writer = SummaryWriter(log_dir=f"outputs/{exp_name}")

    train_loader, val_loader, test_loader = get_dataloaders(config)

    # Model nhận ĐẦY ĐỦ tham số từ config — trước đây patch_size/embed_dim/
    # depth/num_heads trong yaml bị bỏ qua, model luôn chạy default.
    model = ViT(
        img_size=config["model"]["img_size"],
        patch_size=config["model"]["patch_size"],
        in_chans=3,
        n_classes=config["model"]["num_classes"],
        embed_dim=config["model"]["embed_dim"],
        depth=config["model"]["depth"],
        n_heads=config["model"]["num_heads"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=config["train"]["weight_decay"],
    )
    scheduler = build_scheduler(optimizer, config)

    start_epoch = 0
    best_acc = 0.0
    patience = config["train"].get("patience", 5)
    patience_counter = 0

    if args.resume:
        start_epoch, best_acc = load_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )
        logger.info(f"Resumed từ epoch {start_epoch}, best_acc={best_acc:.2f}%")

    checkpoint_dir = config["train"]["checkpoint_dir"]

    for epoch in range(start_epoch, config["train"]["epochs"]):
        logger.info(f"Epoch {epoch + 1}/{config['train']['epochs']}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}% | lr={current_lr:.6f}"
        )

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/acc", train_acc, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/acc", val_acc, epoch)
        writer.add_scalar("train/lr", current_lr, epoch)

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "best_acc": best_acc,
            "config": config,
        }
        save_checkpoint(state, checkpoint_dir, filename="last.pth")

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            state["best_acc"] = best_acc
            save_checkpoint(state, checkpoint_dir, filename="best.pth")
            logger.info(f"Saved best model (val_acc={best_acc:.2f}%)")
        else:
            patience_counter += 1
            logger.info(f"Không cải thiện: {patience_counter}/{patience}")
            if patience_counter >= patience:
                logger.info("Early stopping triggered.")
                break

    writer.close()
    logger.info(f"Training xong. Best val_acc={best_acc:.2f}%")


if __name__ == "__main__":
    main()
