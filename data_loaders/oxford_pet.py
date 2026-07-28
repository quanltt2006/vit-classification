import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset


def get_dataloaders(config):
    """
    Trả về train_loader, val_loader, test_loader.

    QUAN TRỌNG:
    - train/val được tách từ split 'trainval' của Oxford-IIIT Pet.
    - test luôn dùng split 'test' riêng biệt, không bao giờ dùng để chọn best model.
    - train có augmentation, val/test KHÔNG augmentation (để đánh giá công bằng).
    """
    img_size = config['model']['img_size']
    batch_size = config['data']['batch_size']
    num_workers = config['data'].get('num_workers', 2)
    val_split = config['data'].get('val_split', 0.1)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Tải 2 bản trainval: 1 bản có augment (dùng cho train),
    # 1 bản không augment (dùng cho val) — cùng ảnh, khác transform.
    trainval_for_train = datasets.OxfordIIITPet(
        root='data', split='trainval', download=True, transform=train_transform
    )
    trainval_for_val = datasets.OxfordIIITPet(
        root='data', split='trainval', download=True, transform=eval_transform
    )

    n_total = len(trainval_for_train)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(
        range(n_total), [n_train, n_val], generator=generator
    )

    train_dataset = Subset(trainval_for_train, train_subset.indices)
    val_dataset = Subset(trainval_for_val, val_subset.indices)

    test_dataset = datasets.OxfordIIITPet(
        root='data', split='test', download=True, transform=eval_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
