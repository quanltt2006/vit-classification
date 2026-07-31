import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter

from model.vit import ViT
from data_loaders.oxford_pet import get_dataloaders
from engine.trainer import Trainer
from utils.logger import setup_logger
from model.factory import build_model


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
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
    parser.add_argument("--config", type=str, default="configs/local.yaml")
    parser.add_argument(
        "--resume", type=str, default=None, help="Path tới checkpoint để resume"
    )
    args = parser.parse_args()
    config = load_config(args.config)

    device = config["train"]["device"]
    exp_name = config.get("exp_name", "vit_experiment_1")

    logger = setup_logger(log_dir="logs", name=exp_name)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Config yêu cầu device='cuda' nhưng torch.cuda.is_available() = False. "
            "Kiểm tra Colab Runtime > Change runtime type > GPU, và đã Restart runtime chưa."
        )
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        logger.info(f"Đang dùng GPU: {torch.cuda.get_device_name(0)}")

    writer = SummaryWriter(log_dir=f"outputs/{exp_name}")
    train_loader, val_loader, _ = get_dataloaders(config)

    model = build_model(config, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=config["train"]["weight_decay"],
    )

    scheduler = build_scheduler(optimizer, config)

    trainer = Trainer(model, optimizer, scheduler, criterion, config, logger, writer, device)
    if args.resume:
        trainer.resume(args.resume)
    trainer.fit(train_loader, val_loader)
    writer.close()

if __name__ == "__main__":
    main()