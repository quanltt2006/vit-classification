import torch
import torch.nn as nn
import torch.optim as optim
from model.vit import ViT
from data_loaders.oxford_pet import get_dataloaders # Sửa lại tên folder cho đúng thực tế của bạn
from engine.trainer import train_one_epoch, evaluate
import yaml
import argparse
from torch.utils.tensorboard import SummaryWriter 




def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/vit_base.yaml')
    args = parser.parse_args()
    config = load_config(args.config)

    train_loader, val_loader = get_dataloaders(config)
    writer = SummaryWriter(log_dir='logs/vit_experiment_1')




    model = ViT(
        img_size=config['model']['img_size'], 
        n_classes=config['model']['num_classes']
    ).to(config['train']['device'])

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(), 
        lr=float(config['train']['lr']), 
        weight_decay=config['train']['weight_decay']
    ) 

    best_acc = 0.0

    for epoch in range(config['train']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['train']['epochs']}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, config['train']['device']
        )

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, config['train']['device']
        )

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc: 
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved model!!")
    writer.close()


if __name__ == "__main__":
    main()