import os

import torch
from torchvision import models
from torchvision.models import VGG16_Weights

# vortrainierte Gewichte nach ../models/<modell>/ laden (nur einmalig herunterladen)
weights_info = VGG16_Weights.IMAGENET1K_V1
model_dir = os.path.join("..", "models", "vgg16")
os.makedirs(model_dir, exist_ok=True)
weights_path = os.path.join(model_dir, "vgg16-imagenet1k-v1.pth")
if not os.path.exists(weights_path):
    torch.hub.download_url_to_file(weights_info.url, weights_path)

# Overload lets you choose differnt checkpoints from models
vgg16 = models.vgg16(weights=None)
vgg16.load_state_dict(torch.load(weights_path, map_location="cpu"))

print(vgg16.classifier)

import torch.nn as nn
import torch.nn.functional as F


# New subclass that inherits from nn.Module
# PyTorchs base class for neuronal networks
class SimpleNet(nn.Module):
    # __init__ Initializes model parameters
    # three linear layers - layer output has to match next layer input
    # nn.Linear() is a fully connected layer -> "fc"
    def __init__(self):

        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 32)

    # forward defines how data is routed through the model
    def forward(self, x):

        # remodels input tensor to have two dimensions
        # nescessary if x is a multidimensional image tensor
        # batch size -1 and 2048 values per sample
        x = x.view(-1, 2048)

        # activation function, sets all negative values to 0 does not touch positive values
        # for introducing non-linearity
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # rescales all elements of an n-dimensional output tensor
        # all output values inbetween 0 and 1, all summed up = 1
        x = F.softmax(self.fc3(x), dim=1)
        return x
