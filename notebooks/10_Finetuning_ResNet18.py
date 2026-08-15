import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision

from torchvision import datasets, models
from torchvision.models import resnet18, ResNet18_Weights

from torchvision import transforms

from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

# ---------------------------------------------------------
# Zip web download and extract
# ---------------------------------------------------------

zipurl = "https://pytorch.tips/bee-zip"
# "with" is equivalent to C# "using"
# garanties that resource is closed/released after code block
with urlopen(zipurl) as myhttpresponse:
    # BytesIO reads raw Byte that handles like a file but sits in RAM
    with ZipFile(BytesIO(myhttpresponse.read())) as myzipfile:
        # ZipFile interprets the data as Zip and can extract
        myzipfile.extractall("../data/beesAndAnts")


# ---------------------------------------------------------
# Data processing
# ---------------------------------------------------------


myNormalize = transforms.Normalize(
    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
)

test_transforms = transforms.Compose([transforms.ToTensor(), myNormalize])

train_transforms = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        myNormalize,
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        myNormalize,
    ]
)

DATA_DIR = "../data/beesAndAnts/hymenoptera_data"
train_data = datasets.ImageFolder(root=f"{DATA_DIR}/train", transform=train_transforms)
val_data = datasets.ImageFolder(root=f"{DATA_DIR}/val", transform=val_transforms)

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=768,
    shuffle=True,
    num_workers=16,
    pin_memory=True,
    persistent_workers=True,
)

val_loader = torch.utils.data.DataLoader(
    val_data,
    batch_size=768,
    shuffle=False,
    num_workers=16,
    pin_memory=True,
    persistent_workers=True,
)


# ---------------------------------------------------------
# Modell
# ---------------------------------------------------------

# Loads model architecture, is part of the pytorch library
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
# Print out features of last layer
print(f"original last layer:{model.fc}")
# Get In features of last layer
num_ftrs = model.fc.in_features
# Create NEW last layer with machting in-features and 2 out
model.fc = nn.Linear(num_ftrs, 2)
print(f"modified last layer:{model.fc}")

from torch.optim.lr_scheduler import StepLR

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

NUM_EPOCHS = 25

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Deletes old gradients (set to zero)
        # Pytorch automatically accumulates gradients
        # Needed for RNN architecture, not here!
        optimizer.zero_grad()

        # Object instance of ResNet
        # Internally calls "model.__call__(inputs)"
        #
        outputs = model(inputs)

        # outputs is now in tensor shape [batch_size, 2]
        # torch.max returns [max value, index] - we just need index
        _, preds = torch.max(outputs, 1)

        # compares outputs (Logits) with labels
        # calculates the loss from median error across the whole batch
        loss = criterion(outputs, labels)

        # pytorch builds a "computational graph" while moving through the layers
        # "Autograd". backward() moves backward through the graph
        # calculates how strong and in which direction each weight has to change
        # in order to miminize loss. These values are the "gradient"
        loss.backward()

        # optimizer applies the gradients to the weights
        # controlled by hyperparameters
        optimizer.step()

        # ".item" extracts a value from the tensor
        # cutting its "connection" to the graph
        running_loss += loss.item() / inputs.size(0)
        running_corrects += torch.sum(preds == labels.data) / inputs.size(0)

    # learning rate is reduced for each epoch
    exp_lr_scheduler.step()
    # running loss and running corrects have accumulated
    # the values across all batches
    # now calc the mean for each batch
    train_epoch_loss = running_loss / len(train_loader)
    train_epoch_acc = running_corrects / len(train_loader)

    # Set model to evaluation mode
    # changes how model is evaluated
    # for example "dropout" is deactivated, all neurons are active
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    # "no_grad" to disable Autograd
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() / inputs.size(0)
            running_corrects += torch.sum(preds == labels.data) / inputs.size(0)

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = running_corrects.double() / len(val_loader)
    print(
        "Train: Loss: {:.4f} Acc: {:.4f} Val: Loss: {:.4f} Acc: {:.4f}".format(
            train_epoch_loss, train_epoch_acc, epoch_loss, epoch_acc
        )
    )
