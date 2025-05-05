import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from .cnn import SimpleCNN
# 没有分类头的CNN模型，作为backbone使用
# 除了分类头外和cnn中的SimpleCNN完全相同
class SimpleCNN_no_head(nn.Module):
    def __init__(self):
        super(SimpleCNN_no_head, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 输出大小: 16x16x16
        x = self.pool(self.relu(self.conv2(x)))  # 输出大小: 8x8x32
        x = x.view(-1, 32 * 8 * 8)  # 展平
        x = self.relu(self.fc1(x))
        return x

class RotationNet(nn.Module):
    """基于旋转预测的自监督学习模型"""
    def __init__(self, backbone=None):
        super(RotationNet, self).__init__()
        self.backbone = SimpleCNN_no_head() 
        self.clissifier = nn.Linear(128, 4) # 分类头，4个类别对应4种旋转

    
    def forward(self, x):
        # 前向传播
        x = self.backbone(x) # 提取特征 
        x = self.clissifier(x) # 分类头
        return x


def generate_rotations(image):
    # 生成旋转图像和对应的标签
    """为每张图像生成4种旋转 (0°, 90°, 180°, 270°)"""
    rotations = []
    labels = []
    
    # 原始图像 - 0°
    rotations.append(image)
    labels.append(0)
    
    # 旋转90°
    rot90 = transforms.functional.rotate(image, 90)
    rotations.append(rot90)
    labels.append(1)
    
    # 旋转180°
    rot180 = transforms.functional.rotate(image, 180)
    rotations.append(rot180)
    labels.append(2)
    
    # 旋转270°
    rot270 = transforms.functional.rotate(image, 270)
    rotations.append(rot270)
    labels.append(3)
    
    return rotations, labels