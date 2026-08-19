from itertools import filterfalse
import os
import glob
import time
import numpy as np

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2 as transforms
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn_v2,
    MaskRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import Dataset, DataLoader, random_split

from io import BytesIO
from urllib.request import urlopen, Request
from zipfile import ZipFile
from PIL import Image
from scipy import ndimage

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import EarlyStopping

import optuna

# optuna.integration was moved out of the main package in optuna 4.x
# needs: pip install optuna-integration --break-system-packages
from optuna_integration import PyTorchLightningPruningCallback


# ---------------------------------------------------------
# Zip web download and extract
# ---------------------------------------------------------

# ("hello " "world") == ("hello world")
zipurl = (
    "https://raw.githubusercontent.com/abin24/"
    "Surface-Inspection-defect-detection-dataset/master/RoadCracks.zip"
)
DATA_DIR = "../data/crackForest/RoadCracks/Imgs"
MAX_RETRIES = 5

# Only download once - re-running the script must not hit the network again
if not os.path.exists(DATA_DIR):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # some CDN edges drop the default urllib User-Agent mid transfer,
            # which shows up as http.client.IncompleteRead
            myrequest = Request(zipurl, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(myrequest, timeout=30) as myhttpresponse:
                zip_bytes = myhttpresponse.read()
            with ZipFile(BytesIO(zip_bytes)) as myzipfile:
                myzipfile.extractall("../data/crackForest")
            print(f"Download successful on attempt {attempt}")
            break
        except Exception as error:
            print(f"Attempt {attempt}/{MAX_RETRIES} failed: {error}")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(2 * attempt)
else:
    print(f"Dataset already present at {DATA_DIR}")


# ---------------------------------------------------------
# Data
# ---------------------------------------------------------
# No Normalize here: Mask R-CNN carries an internal GeneralizedRCNNTransform
# that normalizes with the ImageNet statistics and resizes on its own.
# A second Normalize would feed the pretrained backbone data it has never seen.
train_transforms = transforms.Compose(
    [
        # converts to tensor with channel axis C,H,W - but remains uint8 from image
        transforms.ToImage(),
        # converts uint8 to float32 | scale=True maps from pixel 0-255 to 0-1
        transforms.ToDtype(torch.float32, scale=True),
        # both steps are in "ToTensor()" but it is deprecated in "transforms.v2"
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
    """Yields (image, target) pairs in the format torchvision detection models expect."""

    # "self" references the instance of the object
    # the parameters "image_dir" and "transform" are set to the instance via "self"
    def __init__(self, image_dir, transform, augment=False):
        self.transform = transform
        self.augment = augment

        # CREATE TUPLE LIST FROM DATA
        # Images and masks share one folder and one stem: 001.jpg <-> 001.png
        self.pairs = []
        # gets all *.jpg in "image_dir", sorts them, full paths already included
        for image_path in sorted(glob.glob(os.path.join(image_dir, "*.jpg"))):
            # splits path and ending into tuple, gets [0] first element and appends ".png"
            mask_path = os.path.splitext(image_path)[0] + ".png"
            # only keep images that actually have a mask
            if os.path.exists(mask_path):
                # appends ONE tuple as ONE list entry
                self.pairs.append((image_path, mask_path))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        # Tuple unpacking
        image_path, mask_path = self.pairs[idx]
        image = Image.open(image_path).convert("RGB")

        # ("L") converts to greyscale | mask is 0 = no crack and 255 = crack
        # ">0" gives TRUE where the condition is met -> numpy array of BOOLS
        foreground = np.array(Image.open(mask_path).convert("L")) > 0

        # -----------------------------------------------------
        # Augmentation
        # -----------------------------------------------------
        # Every geometric change has to move image AND mask together, which is
        # why none of this can live inside the Compose above.
        # Image stays PIL, mask stays numpy -> different tools, same operation.
        if self.augment:
            # Horizontal flip
            if torch.rand(1).item() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                # np.fliplr only returns a "view" with reversed reading order,
                # "copy()" forces a real, contiguous array (torch needs that)
                foreground = np.fliplr(foreground).copy()

            # Vertical flip
            if torch.rand(1).item() < 0.5:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                foreground = np.flipud(foreground).copy()

            # Rotation and affine are mutually exclusive: stacking both on a
            # 2 pixel wide crack breaks it into fragments that the degenerate
            # box filter below then throws away.
            geometric_choice = torch.rand(1).item()

            if geometric_choice < 0.35:
                # ---------------- Random rotation ----------------
                # kept small: large angles create big black corners (fill=0)
                # which the model would learn as a normal image feature
                angle = float(torch.empty(1).uniform_(-8.0, 8.0).item())

                image = TF.rotate(
                    image,
                    angle,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0,
                )

                # NEAREST for the mask: BILINEAR would invent in-between values
                # that do not exist as classes
                mask_tensor = torch.from_numpy(foreground.astype(np.uint8))
                mask_tensor = TF.rotate(
                    mask_tensor.unsqueeze(0),
                    angle,
                    interpolation=InterpolationMode.NEAREST,
                    fill=0,
                ).squeeze(0)
                foreground = mask_tensor.numpy().astype(bool)

            elif geometric_choice < 0.70:
                # ---------------- Random scale + translate ----------------
                height, width = foreground.shape

                max_dx = int(width * 0.10)
                max_dy = int(height * 0.10)

                translate = [
                    int(torch.randint(-max_dx, max_dx + 1, (1,)).item()),
                    int(torch.randint(-max_dy, max_dy + 1, (1,)).item()),
                ]
                # only scaling up avoids black borders from scale < 1.0
                scale = float(torch.empty(1).uniform_(1.00, 1.15).item())

                image = TF.affine(
                    image,
                    angle=0.0,
                    translate=translate,
                    scale=scale,
                    shear=[0.0, 0.0],
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0,
                )

                mask_tensor = torch.from_numpy(foreground.astype(np.uint8))
                mask_tensor = TF.affine(
                    mask_tensor.unsqueeze(0),
                    angle=0.0,
                    translate=translate,
                    scale=scale,
                    shear=[0.0, 0.0],
                    interpolation=InterpolationMode.NEAREST,
                    fill=0,
                ).squeeze(0)
                foreground = mask_tensor.numpy().astype(bool)

            # remaining ~30%: no geometric distortion at all

        # Mask R-CNN wants one binary mask per instance, so the single binary
        # ground truth gets split into its connected components
        # "ndimage" searches and labels connected "True" areas
        labelled, n_components = ndimage.label(foreground, structure=CONNECTIVITY_8)
        # labelled is an array with the same shape as "foreground"
        # 0 for background, 1 for crack nr1, 2 for crack nr2, ...
        # n_components is the highest number contained in labelled

        masks, boxes = [], []
        # python range(a,b) never yields b | "for a < b"
        for component_id in range(1, n_components + 1):
            # gives "True" where labelled == component_id
            # creates a binary mask for this one component
            instance = labelled == component_id

            # "where" gives coordinates of all True elements [rows, columns]
            ys, xs = np.where(instance)

            # min/max are the borders of the bounding box
            # Zero-width or zero-height boxes make the RPN loss return NaN
            if xs.max() == xs.min() or ys.max() == ys.min():
                # -> discard this component
                continue

            masks.append(instance)
            # torchvision box format is x first: [xmin, ymin, xmax, ymax]
            boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])

        # if no valid instance survived for this image
        if len(boxes) == 0:
            # tensors with ZERO entries, but with correct number of axes
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            masks_t = torch.zeros((0, *foreground.shape), dtype=torch.uint8)
            labels_t = torch.zeros((0,), dtype=torch.int64)
        else:
            # np.array() converts the python list first - much faster than
            # building a tensor element by element from nested lists
            # boxes_t   Shape (N, 4)
            boxes_t = torch.as_tensor(np.array(boxes), dtype=torch.float32)
            # masks_t   Shape (N, 320, 480) - uint8 is plenty for a binary mask
            masks_t = torch.as_tensor(np.array(masks), dtype=torch.uint8)
            # Label 0 is reserved for background, so every crack gets label 1
            # labels_t  Shape (N,) - one class number per instance, not per pixel
            labels_t = torch.ones((len(boxes),), dtype=torch.int64)

        # {} is a python dictionary: key:value
        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "masks": masks_t,
            # every value must be a tensor - a plain int has no .to(device)
            # creates a tiny tensor of Shape (1,)
            "image_id": torch.tensor([idx]),
            # vectorized: [:, 3] takes column 3 of ALL rows at once
            "area": (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0]),
            # COCO legacy: 1 would mark an un-separated crowd to be ignored
            # during evaluation. Everything here is annotated separately -> 0
            "iscrowd": torch.zeros((len(boxes_t),), dtype=torch.int64),
        }

        # returns a tuple: the transformed image, and the target dictionary
        return self.transform(image), target


def collate_fn(batch):
    """Every image holds a different number of instances, so targets stay tuples
    instead of being stacked into one tensor."""
    return tuple(zip(*batch))


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
    # Each worker with pytorch process takes 1-2GB RAM. lower to save RAM
    num_workers=4,
    # Faster on PCIe transfer - not needed on unified RAM
    pin_memory=False,
    # Workers do not die if persistent - needs RAM
    persistent_workers=False,
    collate_fn=collate_fn,
)
val_loader = DataLoader(
    val_data,
    batch_size=4,
    shuffle=False,
    num_workers=4,
    pin_memory=False,
    persistent_workers=False,
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

        # modern python3 syntax. old one is "super(CrackMaskRCNN, self).__init__()"
        super().__init__()

        # stores num_classes, lr, momentum, ... in self.hparams
        # without this, self.hparams.lr in configure_optimizers would crash
        self.save_hyperparameters()

        self.model = maskrcnn_resnet50_fpn_v2(
            weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        )

        # Box head: classification scores + bounding box regression
        # "roi_heads" decides "which object" and "where exactly"
        # "cls_score" is a Linear layer, one output value per possible class
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        # The COCO pretrained box_predictor was built for 91 COCO class ids
        # (80 used object classes + background + gaps). We need 2: background + crack.
        # FastRCNNPredictor holds two nn.Linear layers: cls_score + bbox_pred
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Mask head: produces the 28x28 RoIAlign mask
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        # MaskRCNNPredictor is an nn.Sequential of ConvTranspose2d + ReLU + Conv2d.
        # NO Linear layers here - the mask head outputs an IMAGE, not numbers.
        # ConvTranspose2d upscales 14x14 -> 28x28 (same layer type as in a DCGAN
        # generator). The 256 is dim_reduced: channel count of the middle layer.
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
study = optuna.create_study(
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5),
)
study.optimize(objective, n_trials=20)

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

# stops the run once val_loss has not improved for 5 epochs, so max_epochs
# can be set generously without burning time on overfitting
early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")

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
