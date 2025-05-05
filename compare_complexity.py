
import warnings
 
# 屏蔽常见的 transformers 和 torch 警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from models import SimpleMLP, DeepMLP, ResidualMLP, SimpleCNN, MediumCNN, VGGStyleNet, SimpleResNet
from utils import model_complexity 
import torch
device = torch.device( 'cuda:0' if torch.cuda.is_available() else 'cpu ')

models = {
'SimpleMLP ': SimpleMLP(), 
'DeepMLP ': DeepMLP(),
'SimpleCNN ': SimpleCNN(),
'MediumCNN ': MediumCNN(),
'VGGStyleNet ': VGGStyleNet(),  
'SimpleResNet ': SimpleResNet()
}

results = {}
for name, model in models.items(): 
    print(f"\n分析{name}复杂度 :")
    params, time = model_complexity(model, device=device) 
    results[name] = { 'params ': params, 'time ': time}
