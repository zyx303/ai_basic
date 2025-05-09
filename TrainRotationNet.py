import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import warnings 
import os

from models import RotationNet, generate_rotations
from utils.data_loader import load_cifar10
from utils.train_utils import train_model, evaluate_model
from utils.train_utils import plot_training_history

# 创建自监督数据集
class RotationDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        
    def __len__(self):
        return len(self.original_dataset)
        
    def __getitem__(self, idx):
        image, _ = self.original_dataset[idx]  # 忽略原始标签
        rotated_images, rotation_labels = generate_rotations(image)
        
        # 随机选择一个旋转版本
        choice = np.random.randint(0, 4)
        return rotated_images[choice], rotation_labels[choice]

# 主函数
def main():
    # 设置参数
    batch_size = 128
    epochs = 20
    learning_rate = 0.001
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    save_directory = './checkpoints/self_supervised'
    os.makedirs(save_directory, exist_ok=True)
    
    print(f"使用设备: {device}")
    
    # 加载数据
    train_loader, valid_loader, test_loader, classes = load_cifar10(
        use_augmentation=False,  # 不使用数据增强，因为我们要做旋转
        batch_size=batch_size
    )
    
    # 创建自监督数据集和加载器
    ss_train_dataset = RotationDataset(train_loader.dataset)
    ss_valid_dataset = RotationDataset(valid_loader.dataset)
    
    ss_train_loader = DataLoader(
        ss_train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    ss_valid_loader = DataLoader(
        ss_valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # 创建自监督模型
    model = RotationNet().to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 训练模型
    print("开始自监督预训练...")
    trained_model, history = train_model(
        model, ss_train_loader, ss_valid_loader, criterion, optimizer, scheduler,
        num_epochs=epochs, device=device, save_dir=save_directory
    )
    
    
    # 可视化训练历史
    plot_training_history(history, title="Self-Supervised Training History")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()