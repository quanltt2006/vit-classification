import torch
import torch.nn as nn

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(BottleneckBlock, self).__init__()
        # Lớp 1x1 Conv (Bottleneck) để giảm chi phí tính toán
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        
        # Lớp 3x3 Conv để trích xuất đặc trưng
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x # Lưu giữ đặc trưng gốc
        
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        
        # Phép Nối (Concatenation) đặc trưng cũ và mới theo chiều kênh (dim=1)
        return torch.cat([identity, out], 1)


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            # Mỗi layer sau nhận vào (số kênh ban đầu + i * số kênh tăng thêm)
            layers.append(BottleneckBlock(in_channels + i * growth_rate, growth_rate))
        
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DenseNet(nn.Module):
    def __init__(self, num_blocks, growth_rate, num_classes):
        super(DenseNet, self).__init__()
        
        # 1. Stem Layer: Xử lý ảnh đầu vào ban đầu
        self.conv1 = nn.Conv2d(3, 2 * growth_rate, kernel_size=7, padding=3, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(2 * growth_rate)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # 2. Dense Blocks và Transition Layers
        self.dense_blocks = nn.ModuleList()
        in_channels = 2 * growth_rate
        
        for i, num_layers in enumerate(num_blocks):
            # Thêm một Dense Block
            self.dense_blocks.append(DenseBlock(num_layers, in_channels, growth_rate))
            in_channels += num_layers * growth_rate
            
            # Nếu không phải block cuối cùng, thêm Transition Layer để nén dữ liệu
            if i != len(num_blocks) - 1:
                out_channels = in_channels // 2
                transition = nn.Sequential(
                    nn.BatchNorm2d(in_channels),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.AvgPool2d(kernel_size=2, stride=2)
                )
                self.dense_blocks.append(transition)
                in_channels = out_channels

        # 3. Final Layers: Phân loại đầu ra
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.pool2 = nn.AdaptiveAvgPool2d((1, 1)) # Dùng Adaptive để linh hoạt kích thước ảnh
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        # Đi qua Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        # Đi qua các Dense Block và Transition
        for block in self.dense_blocks:
            x = block(x)

        # Kết thúc mạng
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        
        x = torch.flatten(x, 1) # Chuyển thành vector phẳng
        x = self.fc(x)
        
        return x