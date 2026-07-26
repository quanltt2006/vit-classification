from oxford_pet import get_dataloaders

# Giả lập config (Milestone 4 sẽ làm xịn hơn)
config = {
    'image_size': 224,
    'batch_size': 32
}

train_loader, test_loader = get_dataloaders(config)

# Lấy thử 1 batch
images, labels = next(iter(train_loader))
print(f"Image shape: {images.shape}") # Kỳ vọng: [32, 3, 224, 224]
print(f"Label shape: {labels.shape}") # Kỳ vọng: [32]
print("Data Pipeline thành công!")