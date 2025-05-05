from utils.train_utils import visualize_conv_filters 
from models import SimpleCNN 
import torch

trained_model = SimpleCNN()
trained_model.load_state_dict(torch.load("./ck/SimpleCNN_best.pth"))

visualize_conv_filters(trained_model, 'conv1')