import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# 导入项目中的模块
from models import SimpleMLP, DeepMLP, ResidualMLP, SimpleCNN, MediumCNN, VGGStyleNet, SimpleResNet
from utils import load_cifar10, set_seed

def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler=None,
                num_epochs=10, device=None, save_dir='./checkpoints'):
    """训练模型并记录性能指标"""
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    start_time = time.time()
    model = model.to(device)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'epoch_times': []
    }

    best_val_acc = 0.0

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f"Epoch {epoch+1}/{num_epochs}")

        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 统计
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # 计算训练指标
        train_loss = train_loss / len(train_loader.sampler)
        train_acc = train_correct / train_total

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # 统计
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # 计算验证指标
        val_loss = val_loss / len(valid_loader.sampler)
        val_acc = val_correct / val_total

        # 更新学习率
        if scheduler:
            scheduler.step()

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # 记录每个epoch的时间
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        history['epoch_times'].append(epoch_time)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{save_dir}/{model.__class__.__name__}_best.pth")

        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
        print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
        print(f"本轮用时: {epoch_time:.2f}s")
        print("-" * 50)

    # 计算总训练时间
    total_time = time.time() - start_time
    print(f"总训练时间: {total_time:.2f}s")

    return model, history

def evaluate_model(model, test_loader, criterion, device=None):
    """评估模型在测试集上的性能"""
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model.eval()

    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 统计
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    # 计算测试指标
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = test_correct / test_total

    return test_loss, test_acc

def model_complexity(model, input_size=(3, 32, 32), batch_size=128, device=None):
    """计算模型参数量和推理时间"""
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model.eval()

    # 计算参数量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 创建随机输入
    dummy_input = torch.randn(batch_size, *input_size).to(device)

    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # 计时
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)
    end_time = time.time()

    inference_time = (end_time - start_time) / 100

    return num_params, inference_time

def compare_models():
    """比较不同模型的性能"""
    # 设置随机种子
    set_seed()

    # 检查是否有可用的GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    train_loader, valid_loader, test_loader, classes = load_cifar10(
        use_augmentation=True,
        batch_size=128
    )

    # 定义要比较的模型
    models = {
        'SimpleMLP': SimpleMLP(),
        'DeepMLP': DeepMLP(dropout_rate=0.5, use_bn=True, use_dropout=True),
        'ResidualMLP': ResidualMLP(activation='relu'),
        'SimpleCNN': SimpleCNN(),
        'MediumCNN': MediumCNN(use_bn=True),
        'VGGStyleNet': VGGStyleNet(),
        'SimpleResNet': SimpleResNet(num_blocks=[2, 2, 2])
    }

    # 存储结果
    results = {}

    # 训练和评估每个模型
    for model_name, model in models.items():
        print(f"\n开始训练 {model_name}...")

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

        # 计算模型复杂度
        print(f"\n分析 {model_name} 复杂度...")
        num_params, inference_time = model_complexity(model, device=device)

        # 训练模型
        _, history = train_model(
            model, train_loader, valid_loader, criterion, optimizer, scheduler,
            num_epochs=15, device=device, save_dir='./checkpoints'
        )

        # 在测试集上评估模型
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)

        print(f"{model_name} 测试准确率: {test_acc:.4f}")

        # 存储结果
        results[model_name] = {
            'history': history,
            'test_acc': test_acc,
            'params': num_params,
            'inf_time': inference_time
        }

    # 比较模型性能
    model_names = list(results.keys())
    test_accs = [results[name]['test_acc'] for name in model_names]
    params = [results[name]['params'] / 1e6 for name in model_names]  # 转换为百万
    inf_times = [results[name]['inf_time'] * 1000 for name in model_names]  # 转换为毫秒

    # 创建比较图表
    fig, axes = plt.subplots(3, 1, figsize=(15, 15))

    # 测试准确率比较
    ax = axes[0]
    bars = ax.bar(model_names, test_accs, color='skyblue')
    ax.set_title('Model Test Accuracy Comparison')  # 英文标题
    ax.set_ylabel('Accuracy')  # 英文标签
    ax.set_ylim(0, 1)

    # 添加数值标签
    for bar, acc in zip(bars, test_accs):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{acc:.4f}', ha='center', va='bottom')

    # 参数量比较
    ax = axes[1]
    bars = ax.bar(model_names, params, color='lightgreen')
    ax.set_title('Model Parameter Count Comparison (millions)')  # 英文标题
    ax.set_ylabel('Parameters (M)')  # 英文标签

    # 添加数值标签
    for bar, param in zip(bars, params):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{param:.2f}M', ha='center', va='bottom')

    # 推理时间比较
    ax = axes[2]
    bars = ax.bar(model_names, inf_times, color='salmon')
    ax.set_title('Model Inference Time Comparison (ms/batch)')  # 英文标题
    ax.set_ylabel('Inference time (ms)')  # 英文标签

    # 添加数值标签
    for bar, time in zip(bars, inf_times):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{time:.2f}ms', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()

    # 绘制训练曲线比较
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    # 训练损失比较
    ax = axes[0]
    for name in model_names:
        ax.plot(results[name]['history']['train_loss'], label=f'{name} Training')
        ax.plot(results[name]['history']['val_loss'], '--', label=f'{name} Validation')
    ax.set_title('Training Loss Comparison')  # 英文标题
    ax.set_xlabel('Epoch')  # 英文标签
    ax.set_ylabel('Loss')  # 英文标签
    ax.legend()

    # 验证准确率比较
    ax = axes[1]
    for name in model_names:
        ax.plot(results[name]['history']['val_acc'], label=name)
    ax.set_title('Validation Accuracy Comparison')  # 英文标题
    ax.set_xlabel('Epoch')  # 英文标签
    ax.set_ylabel('Accuracy')  # 英文标签
    ax.legend()

    plt.tight_layout()
    plt.savefig('training_curves_comparison.png')
    plt.show()

    return results

if __name__ == "__main__":
    results = compare_models()
