import torch
import torch.nn as nn
import torch.optim as optim
import os

# 导入项目中的模块
from models import SimpleMLP, DeepMLP, ResidualMLP, SimpleCNN, MediumCNN, VGGStyleNet, SimpleResNet, PretrainedViT
from utils import (
    load_cifar10,
    set_seed,
    train_model,
    evaluate_model,
    plot_training_history,
    visualize_model_predictions,
    visualize_conv_filters,
    model_complexity
)


# 设置参数
model_type = 'simple_mlp'  # 可选: 'simple_mlp', 'deep_mlp', 'residual_mlp', 'simple_cnn', 'medium_cnn', 'vgg_style', 'resnet'
epochs = 20
learning_rate = 0.001
batch_size = 128
use_data_augmentation = True  # CNN通常受益于数据增强
save_directory = './ck'
visualize_filters = True  # 是否可视化卷积核（仅对CNN有效）
visualize_predictions = True  # 是否可视化预测结果

# 设置随机种子
set_seed()

#因为mo平台的提交任务机制，需要手动切换到该文件夹下。
# os.chdir(os.path.expanduser("~/work/Jianhai/lab5"))

# 检查是否有可用的GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 加载数据
train_loader, valid_loader, test_loader, classes = load_cifar10(
    use_augmentation=use_data_augmentation,
    batch_size=batch_size
)

# 初始化选择的模型
if model_type == 'simple_mlp':
    model = SimpleMLP()
    model_name = "SimpleMLP"
elif model_type == 'deep_mlp':
    model = DeepMLP(dropout_rate=0.5, use_bn=True, use_dropout=True)
    model_name = "DeepMLP"
elif model_type == 'residual_mlp':
    model = ResidualMLP(activation='relu')
    model_name = "ResidualMLP"
elif model_type == 'simple_cnn':
    model = SimpleCNN()
    model_name = "SimpleCNN"
elif model_type == 'medium_cnn':
    model = MediumCNN(use_bn=True)
    model_name = "MediumCNN"
elif model_type == 'vgg_style':
    model = VGGStyleNet()
    model_name = "VGGStyleNet"
else:  # resnet
    model = SimpleResNet(num_blocks=[2, 2, 2])
    model_name = "SimpleResNet"

print(f"使用模型: {model_name}")

model = PretrainedViT()

# 计算模型复杂度
# print("\n分析模型复杂度:")
# model_complexity(model, device=device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 可以添加学习率调度器
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# 确保checkpoints目录存在
os.makedirs(save_directory, exist_ok=True)



# 训练模型
trained_model, history = train_model(
    model, train_loader, valid_loader, criterion, optimizer, scheduler,
    num_epochs=epochs, device=device, save_dir=save_directory
)

# 绘制训练历史
plot_training_history(history, title=f"{model_name} Training History")

# 在测试集上评估模型
print("\n在测试集上评估模型:")
test_loss, test_acc = evaluate_model(trained_model, test_loader, criterion, device, classes)

print(f"{model_name} 最终测试准确率: {test_acc:.4f}")

# 如果是CNN模型并且需要可视化卷积核
if visualize_filters and model_type in ['simple_cnn', 'medium_cnn', 'vgg_style', 'resnet']:
    print("\n可视化卷积核:")
    if model_type == 'simple_cnn':
        visualize_conv_filters(trained_model, 'conv1')
    elif model_type == 'medium_cnn':
        visualize_conv_filters(trained_model, 'conv1')
    elif model_type == 'vgg_style':
        visualize_conv_filters(trained_model, 'features.0')
    else:  # resnet
        visualize_conv_filters(trained_model, 'conv1')

# 如果需要可视化模型预测
if visualize_predictions:
    print("\n可视化模型预测:")
    visualize_model_predictions(trained_model, test_loader, classes, device)

print(f"\n{model_name}的训练和评估已完成！")
