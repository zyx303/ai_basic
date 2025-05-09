import re
import json
import sys
from collections import defaultdict

def parse_log(log_content):
    """从日志内容中提取训练结果数据"""
    # 创建存储结果的字典
    results = {}
    
    # 当前处理的模型名称
    current_model = None
    
    # 正则表达式模式
    model_start_pattern = re.compile(r'开始训练 (\w+)\.\.\.')
    model_test_pattern = re.compile(r'(\w+) 测试准确率: (\d+\.\d+)')
    epoch_pattern = re.compile(r'Epoch (\d+)/(\d+)')
    train_metrics_pattern = re.compile(r'训练损失: (\d+\.\d+), 训练准确率: (\d+\.\d+)')
    val_metrics_pattern = re.compile(r'验证损失: (\d+\.\d+), 验证准确率: (\d+\.\d+)')
    epoch_time_pattern = re.compile(r'本轮用时: (\d+\.\d+)s')
    total_time_pattern = re.compile(r'总训练时间: (\d+\.\d+)s')
    
    # 处理每一行日志
    lines = log_content.split('\n')
    for line in lines:
        # 跳过空行
        if not line.strip():
            continue
            
        # 提取时间戳后的内容
        parts = line.split(' ', maxsplit  = 1)  # 分割时间戳和内容
        # if len(parts) < 3:
        #     continue
        content = parts[1]
        
        # 检测开始训练新模型
        model_match = model_start_pattern.search(content)
        if model_match:
            current_model = model_match.group(1)
            results[current_model] = {
                'history': {
                    'train_loss': [],
                    'train_acc': [],
                    'val_loss': [],
                    'val_acc': [],
                    'epoch_times': []
                },
                'test_acc': 0.0,
                'params': 1000000,  # 默认参数值
                'inf_time': 0.01    # 默认推理时间
            }
            continue
        
        # 如果当前正在处理一个模型
        if current_model:
            # 提取测试准确率
            test_match = model_test_pattern.search(content)
            if test_match and test_match.group(1) == current_model:
                results[current_model]['test_acc'] = float(test_match.group(2))
                continue
                
            # 提取训练指标
            train_match = train_metrics_pattern.search(content)
            if train_match:
                results[current_model]['history']['train_loss'].append(float(train_match.group(1)))
                results[current_model]['history']['train_acc'].append(float(train_match.group(2)))
                continue
                
            # 提取验证指标
            val_match = val_metrics_pattern.search(content)
            if val_match:
                results[current_model]['history']['val_loss'].append(float(val_match.group(1)))
                results[current_model]['history']['val_acc'].append(float(val_match.group(2)))
                continue
                
            # 提取每轮用时
            time_match = epoch_time_pattern.search(content)
            if time_match:
                results[current_model]['history']['epoch_times'].append(float(time_match.group(1)))
                continue
    
    return results

def save_results(results, output_path):
    """将结果保存到JSON文件"""
    with open(output_path, 'w') as file:
        json.dump(results, file, indent=4)

def main():
    # # 检查是否提供了日志文件路径
    # if len(sys.argv) > 1:
    #     # 从文件读取日志
    #     log_path = sys.argv[1]
    #     try:
    #         with open(log_path, 'r') as file:
    #             log_content = file.read()
    #     except:
    #         print(f"无法读取文件: {log_path}")
    #         return
    # else:
    #     # 从标准输入读取日志
    #     print("请粘贴日志内容，完成后按Ctrl+D (Unix) 或 Ctrl+Z后回车 (Windows):")
    #     log_content = sys.stdin.read()
    log_path = 'D:/desktop/my_zju_course/ai_basic/lab/ai_basic/reslut/compare_result.txt'
    # import os
    # print(os.path.abspath(log_path))  # 检查绝对路径
    # print(os.path.exists(log_path))  # 检查文件是否存在
    # try:
    with open(log_path, 'r',encoding='utf-8') as file:
        log_content = file.read()
    # except:
    #     print(f"无法读取文件: {log_path}")
    #     return
    # 解析日志获取结果
    results = parse_log(log_content)
    
    # 保存结果到JSON文件
    output_file = 'model_results.json'
    save_results(results, output_file)
    
    print(f"解析完成，结果已保存到 {output_file}")
    
    # 显示提取到的模型列表
    print(f"提取到以下模型的训练结果:")
    for model in results:
        print(f"- {model} (测试准确率: {results[model]['test_acc']})")

if __name__ == "__main__":
    main()