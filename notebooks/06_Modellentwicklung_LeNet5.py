import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10


# ---------------------------------------------------------
# Datenvorbereitung
# ---------------------------------------------------------

train_data = CIFAR10(root="./train/", train=True, download=True)
test_data = CIFAR10(root="./train/", train=False, download=True)
normalize = transforms.Normalize(
    mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
)

test_transforms = transforms.Compose([transforms.ToTensor(), normalize])
train_transforms = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
)

train_data.transform = train_transforms
test_data.transform = test_transforms

trainloader = torch.utils.data.DataLoader(
    train_data,
    batch_size=128,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)
testloader = torch.utils.data.DataLoader(
    test_data,
    batch_size=128,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)


# new class that inherits from nn.Module
class LeNet5(nn.Module):
    # "def" defines a method; __init__ runs automatically when a new object is instantiated
    # __init__ defines what happens when a new object of this class is created
    # "self": reference to the object instance, allows attribute access across methods
    def __init__(self):

        # "super()" calls nn.Module's own __init__ first, setting up its internal
        # parameter tracking, before this class's own __init__ continues below
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        # output layer count has to match classes (10)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # nelement() gives total element count; dividing by batch size (shape[0])
        # flattens each sample into a 1D vector for the following Linear layers
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


device = "cuda" if torch.cuda.is_available() else "cpu"

# here the actual Instance of the LeNet5 class is created and moved to the device
model = LeNet5().to(device=device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

N_EPOCHS = 10
for epoch in range(N_EPOCHS):
    epoch_loss = 0.0
    for inputs, labels in trainloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        print("Epoch: {} Loss {}".format(epoch, epoch_loss / len(trainloader)))


torch.save(model.state_dict(), "checkpoints/lenet5_cifar10.pth")
