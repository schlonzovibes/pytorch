import os
import glob
import numpy as np

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import v2 as transforms
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn_v2,
    MaskRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import Dataset, DataLoader, random_split

from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
from PIL import Image
from scipy import ndimage

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import EarlyStopping

import optuna
from optuna.integration import PyTorchLightningPruningCallback

# ---------------------------------------------------------
# Zip web download and extract
# ---------------------------------------------------------

# ("hello " "world") == ("hello world")
zipurl = (
    "https://raw.githubusercontent.com/abin24/"
    "Surface-Inspection-defect-detection-dataset/master/RoadCracks.zip"
)
with urlopen(zipurl) as myhttpresponse:
    with ZipFile(BytesIO(myhttpresponse.read())) as myzipfile:
        myzipfile.extractall("../data/crackForest")


# ---------------------------------------------------------
# Data
# ---------------------------------------------------------
# No Normalize here: Mask R-CNN carries an internal GeneralizedRCNNTransform
# that normalizes with the ImageNet statistics and resizes on its own.
# A second Normalize would feed the pretrained backbone data it has never seen.
train_transforms = transforms.Compose(
    [
        # converts to tensor with channel axis C,H,W - but remains  uint8 from image
        transforms.ToImage(),
        # converts uint8 to float32 | scale=True maps from pixel 0-255 to 0-1
        transforms.ToDtype(torch.float32, scale=True),
        # both steps are in "ToTensor()" but is deprecated in "transforms.v2"
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ]
)

# 8-connectivity structure. With the default 4-connectivity a diagonal crack
# falls apart into up to 32 fragments instead of 7, which would teach the model
# to find crack pieces rather than cracks.
CONNECTIVITY_8 = np.ones((3, 3), dtype=int)


class CrackForestDataset(Dataset):
    # Yields (image, target) pairs in the format torchvision detection models expect."""

    # "self" references the instance of the object
    # the parameters "image_dir" and "transform" are set to the instance via "self"
    def __init__(self, image_dir, transform, augment=False):
        self.transform = transform
        self.augment = augment

        # CREATE TUPLE LIST FROM DATA
        # Images and masks share one folder and one stem: 001.jpg <-> 001.png
        self.pairs = []
        # gets all *jpg in "image_dir", sorts and appends the filenames to the path
        for image_path in sorted(glob.glob(os.path.join(image_dir, "*.jpg"))):
            # splits path and ending into tuple, gets [0] first element and appends ".png"
            mask_path = os.path.splitext(image_path)[0] + ".png"
            # just executes if a mask for that image exists
            if os.path.exists(mask_path):
                # adds both tuple! lists to the python list "pairs"
                self.pairs.append((image_path, mask_path))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        # Tuple unpacking
        image_path, mask_path = self.pairs[idx]
        image = Image.open(image_path).convert("RGB")

        # ("L") converts to greyscale | mask is 0 = no crack and 255 = crack
        # ">0" gives TRUE if condition is met (crack) and numpy array contains now BOOLS
        foreground = np.array(Image.open(mask_path).convert("L")) > 0

        # A horizontal flip has to move image AND mask together, so it cannot
        # live inside the Compose above
        if self.augment and torch.rand(1).item() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

            # flips the numpy array like the image -> only generates a "view"
            # "copy()" forces those changes to the array
            foreground = np.fliplr(foreground).copy()

        # Mask R-CNN wants one binary mask per instance, so the single binary
        # ground truth gets split into its connected components
        # "ndimage" searches and lables connected "True" areas (like pixels)
        labelled, n_components = ndimage.label(foreground, structure=CONNECTIVITY_8)
        # labelled is array with  same size as "foreground"
        # 0 for background, 1 for crack nr1, 2 for crack nr2, ...
        # n_components is the highes number contained in labelled

        masks, boxes = [], []
        # python loop range(a,b) never gives b | "for a < b"
        for component_id in range(1, n_components + 1):
            # gives "True" where component_id == labelled[n]
            # creates a binary mask for each component
            instance = labelled == component_id

            # "where" gives array coordinates with axis for True elements [row, column]
            ys, xs = np.where(instance)

            # max and min gets the extreme values from each row/column
            # those are the borders for the bounding box
            # Zero-width or zero-height boxes make the RPN loss return NaN
            if xs.max() == xs.min() or ys.max() == ys.min():
                # -> discard element
                continue

            # valid boxes are appended to masks list
            masks.append(instance)

            # boxes are created from x/y max/min values
            boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])

        # if no valid boxes have been created for this image
        if len(boxes) == 0:
            # tensors with zero entries are created, but with defined shape
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            masks_t = torch.zeros((0, *foreground.shape), dtype=torch.uint8)
            labels_t = torch.zeros((0,), dtype=torch.int64)
        else:
            # tensor is created from python list boxes
            # np.array() converts the list beforehand to numpy array
            # faster from np.array to tensor
            # boxes_t   Shape (3, 4)
            boxes_t = torch.as_tensor(np.array(boxes), dtype=torch.float32)
            # same for mask data, 8bit (255 values) more than enough for binary mask
            # masks_t   Shape (3, 320, 480)
            masks_t = torch.as_tensor(np.array(masks), dtype=torch.uint8)
            # Label 0 is reserved for background, so every crack gets label 1
            # labels_t  Shape (3,)
            labels_t = torch.ones((len(boxes),), dtype=torch.int64)

        # {} is python dictionary: key:value
        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "masks": masks_t,
            # every value must be a tensor before processing
            # creates a tiny tensor "Shape(1,)""
            "image_id": torch.tensor([idx]),
            # vectorizes all box data at once (complicated)
            "area": (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0]),
            # iscrows is COCO-Erbe. bin zu müde
            "iscrowd": torch.zeros((len(boxes_t),), dtype=torch.int64),
        }

        # returns a tuple: the transformed image, and the target dictionary
        return self.transform(image), target


def collate_fn(batch):
    """Every image holds a different number of instances, so targets stay tuples
    instead of being stacked into one tensor."""
    return tuple(zip(*batch))


DATA_DIR = "../data/crackForest/RoadCracks/Imgs"

# CrackForest ships without a train/val split, so it is cut here. Two dataset
# instances over the same files with identical seeds produce identical index
# sets - the training half gets augmentation, the validation half does not.
# Subset objects cannot have their transform reassigned afterwards.
data_augmented = CrackForestDataset(DATA_DIR, train_transforms, augment=True)
data_plain = CrackForestDataset(DATA_DIR, val_transforms, augment=False)

n_total = len(data_augmented)
n_val = int(n_total * 0.2)
n_train = n_total - n_val

train_data, _ = random_split(
    data_augmented, [n_train, n_val], generator=torch.Generator().manual_seed(42)
)
_, val_data = random_split(
    data_plain, [n_train, n_val], generator=torch.Generator().manual_seed(42)
)

print(f"CrackForest: {n_total} pairs -> {n_train} train / {n_val} val")

train_loader = DataLoader(
    train_data,
    batch_size=4,
    shuffle=True,
    num_workers=16,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=collate_fn,
)
val_loader = DataLoader(
    val_data,
    batch_size=4,
    shuffle=False,
    num_workers=16,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=collate_fn,
)


class CrackMaskRCNN(L.LightningModule):
    def __init__(
        self,
        num_classes=2,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005,
        step_size=7,
        gamma=0.1,
    ):

        # modern python3 syntax. old one is "super(LightningModule, self).__init__"
        super().__init__()

        # above values "num_classes, lr, momentum,..." are save in self.hparams
        self.save_hyperparameters()

        self.model = maskrcnn_resnet50_fpn_v2(
            weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        )

        # Box head: classification scores + bounding box regression
        # accesses the "roi_heads" which determin "which object" and "where located"
        # "cls_score" is a linear layer. outputs one value per possible class
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        # the COCO pretrained "box_predicator" was trained on 21 COCO classes
        # we need 2 (background + crack)
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Mask head: the 28x28 RoIAlign mask predictor
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        # MaskRCNNPredicator is a class containing two nn.Linear Layers
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, 256, num_classes
        )

        # No self.criterion here - Mask R-CNN computes its losses internally

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch

        # In train mode the model returns a dict of five partial losses:
        # classifier, box regression, mask, RPN objectness, RPN box
        loss_dict = self.model(images, targets)
        loss = sum(loss_dict.values())

        # Lightning cannot infer the batch size from a tuple of dicts
        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            batch_size=len(images),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch

        # In eval mode Mask R-CNN returns predictions instead of losses, so
        # there would be no val_loss at all. Staying in train mode is safe here
        # because the backbone uses FrozenBatchNorm2d - no running statistics
        # are updated - and the model has no dropout.
        self.model.train()
        with torch.no_grad():
            loss_dict = self.model(images, targets)
        loss = sum(loss_dict.values())

        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            batch_size=len(images),
        )

    def configure_optimizers(self):
        optimizer = SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = StepLR(
            optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma
        )
        return [optimizer], [scheduler]


# ---------------------------------------------------------
# Optuna objective - runs many times, saves NOTHING
# ---------------------------------------------------------
def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 2e-2, log=True)
    momentum = trial.suggest_float("momentum", 0.8, 0.95)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    step_size = trial.suggest_int("step_size", 2, 6)
    gamma = trial.suggest_float("gamma", 0.05, 0.5)
    # Mask R-CNN is far heavier per image than ResNet18, so the batches stay small
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,
    )

    model = CrackMaskRCNN(
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        step_size=step_size,
        gamma=gamma,
    )

    trainer = L.Trainer(
        max_epochs=50,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")],
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
# study = optuna.create_study(direction="minimize")
study = optuna.create_study(
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5),
)
study.optimize(objective, n_trials=20)

print("Best hyperparameters found:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")

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
    collate_fn=collate_fn,
)
val_loader = DataLoader(
    val_data,
    batch_size=best["batch_size"],
    shuffle=False,
    num_workers=16,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=collate_fn,
)

final_model = CrackMaskRCNN(
    lr=best["lr"],
    momentum=best["momentum"],
    weight_decay=best["weight_decay"],
    step_size=best["step_size"],
    gamma=best["gamma"],
)

checkpoint_callback = ModelCheckpoint(
    dirpath="../checkpoints",
    filename="maskrcnn_crackForest_epoch{epoch:02d}_valloss{val_loss:.4f}_trainloss{train_loss:.4f}",
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
    callbacks=[checkpoint_callback, early_stop],
)

final_trainer.fit(
    final_model, train_dataloaders=train_loader, val_dataloaders=val_loader
)

print(f"Best checkpoint saved at: {checkpoint_callback.best_model_path}")
