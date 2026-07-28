"""
Chạy: python -m data_loaders.test_data
Kiểm tra nhanh pipeline dữ liệu trả về đúng shape và có đủ 3 loader.
"""
import yaml
from data_loaders.oxford_pet import get_dataloaders

with open("configs/vit_base.yaml", "r") as f:
    config = yaml.safe_load(f)

train_loader, val_loader, test_loader = get_dataloaders(config)

images, labels = next(iter(train_loader))
print(f"Image shape: {images.shape}")  # Kỳ vọng: [batch, 3, img_size, img_size]
print(f"Label shape: {labels.shape}")  # Kỳ vọng: [batch]
print(f"Train batches: {len(train_loader)}")
print(f"Val batches:   {len(val_loader)}")
print(f"Test batches:  {len(test_loader)}")
print("Data Pipeline thành công!")
