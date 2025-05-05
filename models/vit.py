import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel
from torchvision import transforms
import timm

class PretrainedViT(nn.Module):
    """使用预训练的Vision Transformer模型，适配CIFAR-10"""
    def __init__(self, model_name='vit_tiny_patch16_224', num_classes=10, img_size=32):
        super(PretrainedViT, self).__init__()
        
        # 加载预训练模型
        self.model = timm.create_model(model_name, pretrained=True)
        
        # 替换分类头以匹配CIFAR-10的类别数
        self.embedding_dim = self.model.head.in_features
        self.model.head = nn.Linear(self.embedding_dim, num_classes)
        
        # 由于预训练模型通常设计用于224x224图像，我们需要调整CIFAR-10的32x32图像
        self.img_size = img_size
        
        # 定义图像预处理转换
        self.transform = transforms.Compose([
            transforms.Resize(224),  # 调整为模型预期的尺寸
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])
        
        # 初始化新的分类头
        nn.init.zeros_(self.model.head.bias)
        
    def forward(self, x):
        # 调整图像大小以匹配预训练模型的预期输入
        x = self.transform(x)
        
        # 前向传播通过预训练模型
        x = self.model(x)
        
        return x