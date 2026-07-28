import torch
from model.vit import ViT

def test_vit():
    img_size = 224
    batch_size = 4
    n_classes = 37 # Theo bộ Oxford Pet
    
    model = ViT(img_size=img_size, n_classes=n_classes)
    
    # Tạo dữ liệu giả (Dummy data)
    x = torch.randn(batch_size, 3, img_size, img_size)
    
    # Forward pass
    output = model(x)
    
    # Kiểm tra shape
    assert output.shape == (batch_size, n_classes)
    print("Test passed! Output shape is correct.")

if __name__ == "__main__":
    test_vit()