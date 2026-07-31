import torch
from tqdm import tqdm

from utils.metrics import calculate_accuracy
from utils.checkpoint import save_checkpoint, load_checkpoint


class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion, 
                 config, logger, writer, device):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.config = config
        self.logger = logger
        self.writer = writer
        self.device = device

        use_amp = config["train"].get("amp", False) and device == "cuda"
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None

        self.logger.info(f"Mixed precision (AMP): {'BẬT' if use_amp else 'TẮT'}")

        self.checkpoint_dir = config["train"]["checkpoint_dir"]

        self.patience = config["train"].get("patience", 5)
        self.start_epoch = 0 
        self.best_acc = 0.0
        self.patience_counter = 0

    def train_one_epoch(self, loader) -> tuple:
        self.model.train()
        running_loss, running_acc = 0.0, 0.0

        for images, labels in tqdm(loader, desc="Training"):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            if self.scaler is not None:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            acc = calculate_accuracy(outputs, labels)[0]
            running_loss += loss.item()
            running_acc += acc.item()

        n = len(loader)
        return running_loss / n, running_acc / n
    @torch.no_grad()
    def validate(self, loader) -> tuple:
        self.model.eval()
        running_loss, running_acc = 0.0, 0.0

        for images, labels in tqdm(loader, desc="Validating"):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            acc = calculate_accuracy(outputs, labels)[0]

            running_loss += loss.item()
            running_acc += acc.item()

        n = len(loader)
        return running_loss / n, running_acc / n
    def save(self, epoch: int, filename: str):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_acc": self.best_acc,
            "config": self.config,
        }
        save_checkpoint(state, self.checkpoint_dir, filename=filename)
    def resume(self, checkpoint_path: str):
        self.start_epoch, self.best_acc = load_checkpoint(
        checkpoint_path, self.model, self.optimizer, self.scheduler, self.device)
        self.logger.info(f"Resumed từ epoch {self.start_epoch}, best_acc={self.best_acc:.2f}%")
    def fit(self, train_loader, val_loader):
        epochs = self.config["train"]["epochs"]

        for epoch in range(self.start_epoch, epochs):
            self.logger.info(f"Epoch {epoch + 1}/{epochs}")

            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            if self.scheduler is not None:
                self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.logger.info(
                f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}% | lr={current_lr:.6f}"
            )

            self.writer.add_scalar("train/loss", train_loss, epoch)
            self.writer.add_scalar("train/acc", train_acc, epoch)
            self.writer.add_scalar("val/loss", val_loss, epoch)
            self.writer.add_scalar("val/acc", val_acc, epoch)
            self.writer.add_scalar("train/lr", current_lr, epoch)

            self.save(epoch, filename="last.pth")

            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.patience_counter = 0
                self.save(epoch, filename="best.pth")
                self.logger.info(f"Saved best model (val_acc={self.best_acc:.2f}%)")
            else:
                self.patience_counter += 1
                self.logger.info(f"Không cải thiện: {self.patience_counter}/{self.patience}")
                if self.patience_counter >= self.patience:
                    self.logger.info("Early stopping triggered.")
                    break

        self.logger.info(f"Training xong. Best val_acc={self.best_acc:.2f}%")