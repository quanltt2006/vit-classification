import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(config):
    train_transform = transforms.Compose([
        transforms.Resize((config['model']['img_size'], config['model']['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((config['model']['img_size'], config['model']['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.OxfordIIITPet(
        root='data', split='trainval', download=True, transform=train_transform
    )
    
    test_dataset = datasets.OxfordIIITPet(
        root='data', split='test', download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config['data']['batch_size'], shuffle=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=config['data']['batch_size'], shuffle=False
    )

    return train_loader, test_loader

