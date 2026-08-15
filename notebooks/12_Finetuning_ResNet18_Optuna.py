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

import optuna

# ---------------------------------------------------------
# Zip web download and extract (unchanged from before)
# ---------------------------------------------------------
zipurl = "https://pytorch.tips/bee-zip"
with urlopen(zipurl) as myhttpresponse:
    with ZipFile(BytesIO(myhttpresponse.read())) as myzipfile:
        myzipfile.extractall("../data/beesAndAnts")


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


class BeesAntsClassifier(L.LightningModule):
    def __init__(self, num_classes=2, lr=0.001, momentum=0.9, step_size=7, gamma=0.1):
        super().__init__()
        self.save_hyperparameters()

        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()

        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()

        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, on_step=False, prog_bar=True)

    def configure_optimizers(self):
        optimizer = SGD(
            self.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum
        )
        scheduler = StepLR(
            optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma
        )
        return [optimizer], [scheduler]


# ---------------------------------------------------------
# Optuna objective - runs many times, saves NOTHING
# ---------------------------------------------------------
def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    momentum = trial.suggest_float("momentum", 0.5, 0.99)
    step_size = trial.suggest_int("step_size", 3, 10)
    gamma = trial.suggest_float("gamma", 0.05, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    model = BeesAntsClassifier(
        lr=lr, momentum=momentum, step_size=step_size, gamma=gamma
    )

    trainer = L.Trainer(
        max_epochs=15,
        accelerator="auto",
        devices="auto",
        logger=False,
        enable_checkpointing=False,  # no files written during the search
        enable_progress_bar=False,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return trainer.callback_metrics["val_loss"].item()


# ---------------------------------------------------------
# Hyperparameter search
# ---------------------------------------------------------
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=25)

print("Best hyperparameters found:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")


# ---------------------------------------------------------
# Final training with the best hyperparameters - saves ONE checkpoint
# ---------------------------------------------------------
best = study.best_params

train_loader = DataLoader(
    train_data,
    batch_size=best["batch_size"],
    shuffle=True,
    num_workers=16,
    pin_memory=True,
    persistent_workers=True,
)
val_loader = DataLoader(
    val_data,
    batch_size=best["batch_size"],
    shuffle=False,
    num_workers=16,
    pin_memory=True,
    persistent_workers=True,
)

final_model = BeesAntsClassifier(
    lr=best["lr"],
    momentum=best["momentum"],
    step_size=best["step_size"],
    gamma=best["gamma"],
)

checkpoint_callback = ModelCheckpoint(
    dirpath="../checkpoints",
    filename="resnet18_beesAndAnts_epoch{epoch:02d}_valloss{val_loss:.4f}_trainloss{train_loss:.4f}",
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    auto_insert_metric_name=False,
)

final_trainer = L.Trainer(
    max_epochs=50,
    accelerator="auto",
    devices="auto",
    logger=False,
    callbacks=[checkpoint_callback],
)

final_trainer.fit(
    final_model, train_dataloaders=train_loader, val_dataloaders=val_loader
)

print(f"Best checkpoint saved at: {checkpoint_callback.best_model_path}")
