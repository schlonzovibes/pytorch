import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint


# ---------------------------------------------------------
# Zip web download and extract (unchanged from before)
# ---------------------------------------------------------
zipurl = "https://pytorch.tips/bee-zip"
with urlopen(zipurl) as myhttpresponse:
    with ZipFile(BytesIO(myhttpresponse.read())) as myzipfile:
        myzipfile.extractall("../data/beesAndAnts")


# ---------------------------------------------------------
# LightningModule: bundles model + train/val logic + optimizer
# ---------------------------------------------------------
class BeesAntsClassifier(L.LightningModule):
    def __init__(self, num_classes=2, lr=0.001, momentum=0.9, step_size=7, gamma=0.1):
        super().__init__()
        # saves all __init__ arguments, accessible later via self.hparams
        self.save_hyperparameters()

        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    # Called automatically by the Trainer for every training batch
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()

        # self.log replaces the manual running_loss/running_corrects bookkeeping
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, on_step=False, prog_bar=True)
        return loss  # Lightning calls backward() and optimizer.step() for you

    # Called automatically for every validation batch (inside torch.no_grad()
    # and with model.eval() already set - both handled by the Trainer)
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()

        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, on_step=False, prog_bar=True)

    # Replaces the manual optimizer + scheduler setup
    def configure_optimizers(self):
        optimizer = SGD(
            self.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum
        )
        scheduler = StepLR(
            optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma
        )
        return [optimizer], [scheduler]


# ---------------------------------------------------------
# Data (unchanged from before)
# ---------------------------------------------------------
myNormalize = transforms.Normalize(
    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
)

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

train_loader = DataLoader(
    train_data,
    batch_size=768,
    shuffle=True,
    num_workers=16,
    pin_memory=True,
    persistent_workers=True,
)
val_loader = DataLoader(
    val_data,
    batch_size=768,
    shuffle=False,
    num_workers=16,
    pin_memory=True,
    persistent_workers=True,
)


# ---------------------------------------------------------
# Training - this replaces the entire manual for-loop
# ---------------------------------------------------------
model = BeesAntsClassifier()

checkpoint_callback = ModelCheckpoint(
    dirpath="../checkpoints",
    filename="resnet18_beesAndAnts_epoch{epoch:02d}_valloss{val_loss:.4f}_trainloss{train_loss:.4f}",
    # metric deciding which model to keep
    monitor="val_loss",
    # better loss has to be smaller, acc would be "max"
    mode="min",
    # saves just the best, k3 would be best three
    save_top_k=1,
    auto_insert_metric_name=False,
)


trainer = L.Trainer(
    max_epochs=25,
    accelerator="auto",  # picks GPU automatically if available, else CPU
    devices="auto",
    logger=False,
    callbacks=[checkpoint_callback],
)

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
