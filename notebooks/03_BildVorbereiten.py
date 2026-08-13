import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# Open Image, Create Axis Image, Save Image
img = Image.open("../images/coffee.jpg")
plt.imshow(img)
plt.savefig("../images/coffee_edit.png")


import torch
from torchvision import transforms

# Image is PIL format - convert to Tensor Format!
# Compose a custom transform
myTransform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        # Scales pixel vales from 0-255 to 0-1
        transforms.ToTensor(),
        # Normalize scales them from 0-1 to center 0 with abbreviations depending on mean and std
        # Mean and Std over ALL trained images per channel
        # "global mean" - mean brightness of channeli
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Apply transform and print new image attributes
img_tensor = myTransform(img)
print(type(img_tensor), img_tensor.shape)


# Model requires batches for training
# "Unsqueeze" adds a dimension to tensor - create a batch of dimension 1
# out: torch.Size ([1, 3, 224, 224])
batch = img_tensor.unsqueeze(0)
print(batch.shape)


# Load pretrained AlexNet from 2012
from torchvision import models
from torchvision.models import AlexNet_Weights

# LEGACYCODE  model = models.alexnet(pretrained=True)
# "models" contains architecture like alexnet or resnet50
# "weights" ENUM selects pretrained weights
myModel = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# sets the model to Inference mode
# in training mode some layers like dropout and batch normalization act differently
# Dropout deactivates neurons randomly to prevent overfitting
# Fatal for inference because every output would be different not deterministic
myModel.eval()
myModel.to(device)

# Our batch contained 1 image, so first dimension of output is 1
# Output containes 1000 classes so output dimension is [1, 1000]
y = myModel(batch.to(device))
print(y.shape)

# Evaluate winning class
# tensor([967], device='cuda:0') tensor([22.8561],...
# Tensor Nr. 967 has highest value of 22.8561
y_max, index = torch.max(y, 1)
print(index, y_max)


import urllib.request

url = "https://pytorch.tips/imagenet-labels"
fpath = "imagenet_class_labels.txt"
urllib.request.urlretrieve(url, fpath)

# Creates a list from each line in "f:"
with open("imagenet_class_labels.txt") as f:
    classes = [line.strip() for line in f.readlines()]

print(classes[index])

# Convert raw scores into probabilities
# Apply softmax on Axis 1 of  [1, 1000]
prob = torch.nn.functional.softmax(y, dim=1)[0] * 100
_, indices = torch.sort(y, descending=True)

# Selects top5 indizes for first (and only) image
for idx in indices[0][:5]:
    print(classes[idx], prob[idx].item())
