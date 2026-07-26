import torch
import torch.nn as nn
import torch.optim as optim
from model.vit import ViT
from data_loaders.oxford_pet import get_dataloaders
from engine.trainer import train_one_epoch, evaluate


def main():
    config = {
        'image_size': 224,
        'batch_size': 32,
        'lr': 1e-4,
        'epochs': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }


    train_loader, val_loader = get_dataloaders(config)


    model = ViT(img_size=224, n_classes=37).to(config['device'])


    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr = config['lr'], weight_decay= 0.05) 

    best_acc = 0.0


    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")

        train_loss, train_acc = train_one_epoch(model, train_loader,
                                                criterion, optimizer,
                                                config['device'])

        val_loss, val_acc = evaluate(model, val_loader, criterion, 
                                     config['device'])

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc: 
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved model!!")

if __name__ == "__main__":
    main()