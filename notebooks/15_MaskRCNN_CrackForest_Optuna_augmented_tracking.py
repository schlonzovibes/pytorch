"""
Mask R-CNN instance segmentation on the CrackForest / RoadCracks dataset.

Revision notes (v2):
  - Optuna now optimizes a deterministic pixel IoU instead of the stochastic
    multi-task validation loss.
  - Worker teardown crash fixed, study persisted to SQLite, pruner relaxed.
  - bf16 mixed precision, reduced internal rescaling, photometric augmentation.
  - Optional noise-floor measurement to verify that trial differences are real.
"""

import os
import re
import glob
import time
import numpy as np

# NEW: must be set BEFORE torch is imported. With blocking launches every CUDA
# kernel reports its error at the call site instead of a few frames later, so
# the traceback finally points at the real culprit. Costs ~20-30% speed - only
# for debugging, never for a production run.
DEBUG_CUDA = False
if DEBUG_CUDA:
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2 as transforms
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn_v2,
    MaskRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead

from io import BytesIO
from urllib.request import urlopen, Request
from zipfile import ZipFile
from PIL import Image
from scipy import ndimage
from scipy.ndimage import binary_dilation

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

# NEW: MLflow logger. PIL.Image is already imported further up and is reused
# for the mask overlays, so no extra image import is needed here.
from lightning.pytorch.loggers import MLFlowLogger

# NEW: torchmetrics ships with lightning, no extra install needed
from torchmetrics.classification import BinaryJaccardIndex

import optuna
from optuna_integration import PyTorchLightningPruningCallback


# =========================================================
# Configuration
# =========================================================

DATA_DIR = "../data/crackForest/RoadCracks/Imgs"
CHECKPOINT_DIR = "../checkpoints"

# NEW: every SQLite database this project writes lives here. SQLite does NOT
# create missing directories - it just fails with "unable to open database
# file" - so the folder is created before any URI below is used.
SQL_DIR = "../sql"
os.makedirs(SQL_DIR, exist_ok=True)

# NEW: the checkpoint directory is shared with other models, so it must never
# be wiped. Everything this script writes carries this prefix, and cleanup
# only ever touches files that match it.
CHECKPOINT_PREFIX = "maskrcnn_crackForest_"

# CHANGED: moved out of the notebooks parent into SQL_DIR
STUDY_STORAGE = f"sqlite:///{SQL_DIR}/optuna_maskrcnn.db"
STUDY_NAME = "maskrcnn_crackforest_iou"

# NEW: MLflow tracking. Separate database from the Optuna storage - the two
# dashboards answer different questions: optuna-dashboard visualizes the
# SEARCH SPACE, MLflow visualizes individual RUNS over their epochs.
#
#   pip install mlflow --break-system-packages
#   mlflow ui --backend-store-uri sqlite:///../sql/mlflow.db --host 0.0.0.0 --port 5000
#
# Then reach it through an SSH tunnel:
#   ssh -L 5000:localhost:5000 user@sparky   ->   http://localhost:5000
USE_MLFLOW = True
# CHANGED: moved out of the notebooks parent into SQL_DIR
MLFLOW_URI = f"sqlite:///{SQL_DIR}/mlflow.db"
MLFLOW_EXPERIMENT = "maskrcnn_crackforest"

# NEW: how often a prediction overlay is uploaded as an artifact. This is the
# whole point of the browser UI - an IoU of 0.22 does not say WHETHER the model
# misses cracks, draws them too thin, or hallucinates them on asphalt texture.
# The images do. 0 disables image logging.
LOG_IMAGE_EVERY_N_EPOCHS = 5

SEED = 42
VAL_FRACTION = 0.2

# NEW: run mode switch.
#   "noise"  -> train the same configuration N times and report mean/std.
#               Do this FIRST. If the spread is larger than the gap between
#               your best trials, the search is a lottery and tuning is
#               pointless until the metric is stabilized.
#   "search" -> Optuna search followed by one final long training run.
RUN_MODE = "search"
NOISE_REPEATS = 5

# CHANGED: 50 epochs per trial was wasteful. Ranking configurations does not
# need full convergence; the final run gets the long budget instead.
SEARCH_EPOCHS = 25
# CHANGED: 80 was far too generous. The last run peaked at epoch 10 and
# EarlyStopping fired at 22 - the remaining 58 epochs only stretched the cosine
# curve, so search and final run were no longer training under comparable
# conditions. That alone may explain why the final run scored WORSE (0.2245)
# than the trial its hyperparameters came from (0.2576).
FINAL_EPOCHS = 30
N_TRIALS = 20

# CHANGED: torchvision rescales every input to min_size=800 by default.
# The source images are 320x480, so the default inflates them by ~6x in pixel
# count. Some upscaling helps 2px wide cracks survive the 28x28 RoIAlign mask
# head, but 800 is almost certainly overkill. Treat 480 vs 800 as an A/B test
# once the IoU metric is in place.
MODEL_MIN_SIZE = 480
MODEL_MAX_SIZE = 800

# NEW: how many of the five ResNet stages stay trainable. torchvision's default
# is 3. With 121 images and 45.7 M trainable parameters the last run overfitted
# hard (train_loss 0.308 vs val_loss 0.420, IoU falling after epoch 10), so
# freezing more of the pretrained backbone is the strongest available lever.
# Kept at the default for now so this is a deliberate A/B, not a silent change.
TRAINABLE_BACKBONE_LAYERS = 3

# NEW: replacing the anchor generator also discards the COCO-pretrained RPN
# head weights, which can hurt with only ~120 training images. Extreme aspect
# ratios fit thin diagonal cracks far better in theory - verify with the IoU
# before turning this on permanently.
USE_CRACK_ANCHORS = False

# NEW: StepLR needs step_size retuned whenever max_epochs changes, which makes
# search and final run inconsistent. Cosine scales itself to the budget and
# removes two dimensions from the search space.
SCHEDULER = "cosine"  # "cosine" or "step"

# CHANGED: bf16 instead of fp16 - Mask R-CNN box regression can produce NaN
# under fp16, bf16 has the exponent range of fp32 and is native on GB10.
# CHANGED: back to full fp32. The second crash landed in the cuDNN backward
# pass during training, not in the eval postprocessing, so reduced precision
# is no longer a sufficient explanation - but it stays a suspect until ruled
# out. Re-enable "bf16-mixed" only after the run is stable."32-true"
PRECISION = "bf16-mixed"

# NEW: cuDNN autotuning benchmarks several algorithms per layer shape and
# keeps the fastest. Some of those kernels are freshly written for Blackwell;
# turning the search off pins the library to the conservative default path.
torch.backends.cudnn.benchmark = False

# NEW: torchvision's own detection reference training clips gradients, and it
# matters more here. A single exploding step produces inf box regression
# deltas, which then crash NMS in the eval pass. Clipping is the cheapest
# insurance against that whole failure class.
GRAD_CLIP = 1.0

# NEW: confidence cutoff for counting a predicted instance. 0.5 is strict for
# an early-epoch model - a run reporting exactly 0.0 IoU may simply have no
# instance above the bar rather than nothing learned at all.
SCORE_THRESHOLD = 0.3

# NEW: minimum width and height in pixels for a box to be kept as an instance
MIN_BOX_EXTENT = 2

# CHANGED: workers switched off for the search. My earlier claim that
# persistent_workers caused the "can only test a child process" assertion was
# wrong - it kept appearing with False, and more often, because every epoch
# forks anew. The assertion comes from a forked child running __del__ on an
# inherited DataLoader iterator whose workers it does not own. With 121 images
# parallel loading buys nothing anyway: 25 epochs x 2 loaders x 4 workers is
# ~200 process starts per trial.
SEARCH_NUM_WORKERS = 0
SEARCH_PERSISTENT_WORKERS = False
# The final run forks once and keeps the workers alive, so it stays parallel.
FINAL_NUM_WORKERS = 4
FINAL_PERSISTENT_WORKERS = True

L.seed_everything(SEED, workers=True)


# =========================================================
# Download
# =========================================================

zipurl = (
    "https://raw.githubusercontent.com/abin24/"
    "Surface-Inspection-defect-detection-dataset/master/RoadCracks.zip"
)
MAX_RETRIES = 5

if not os.path.exists(DATA_DIR):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
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


# =========================================================
# Transforms
# =========================================================
# No Normalize: GeneralizedRCNNTransform inside the model already normalizes
# with the ImageNet statistics.

# NEW: photometric augmentation. These only touch the image, never the mask,
# so unlike the geometric operations they can live inside the Compose.
# With 121 training images this is the cheapest available regularization.
train_transforms = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.2),
        transforms.ToDtype(torch.float32, scale=True),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ]
)

CONNECTIVITY_8 = np.ones((3, 3), dtype=int)


# =========================================================
# Dataset
# =========================================================


class CrackForestDataset(Dataset):
    """Yields (image, target) pairs in torchvision detection format."""

    def __init__(self, image_dir, transform, augment=False):
        self.transform = transform
        self.augment = augment

        self.pairs = []
        for image_path in sorted(glob.glob(os.path.join(image_dir, "*.jpg"))):
            mask_path = os.path.splitext(image_path)[0] + ".png"
            if os.path.exists(mask_path):
                self.pairs.append((image_path, mask_path))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs[idx]
        image = Image.open(image_path).convert("RGB")
        foreground = np.array(Image.open(mask_path).convert("L")) > 0

        # -----------------------------------------------------
        # Geometric augmentation - image and mask must move together
        # -----------------------------------------------------
        if self.augment:
            if torch.rand(1).item() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                foreground = np.fliplr(foreground).copy()

            if torch.rand(1).item() < 0.5:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                foreground = np.flipud(foreground).copy()

            geometric_choice = torch.rand(1).item()

            if geometric_choice < 0.35:
                angle = float(torch.empty(1).uniform_(-8.0, 8.0).item())

                image = TF.rotate(
                    image,
                    angle,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0,
                )

                mask_tensor = torch.from_numpy(foreground.astype(np.uint8))
                mask_tensor = TF.rotate(
                    mask_tensor.unsqueeze(0),
                    angle,
                    interpolation=InterpolationMode.NEAREST,
                    fill=0,
                ).squeeze(0)
                foreground = mask_tensor.numpy().astype(bool)

            elif geometric_choice < 0.70:
                height, width = foreground.shape

                max_dx = int(width * 0.10)
                max_dy = int(height * 0.10)

                translate = [
                    int(torch.randint(-max_dx, max_dx + 1, (1,)).item()),
                    int(torch.randint(-max_dy, max_dy + 1, (1,)).item()),
                ]
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

        # -----------------------------------------------------
        # Split the binary mask into instances
        # -----------------------------------------------------
        # CHANGED: dilate before labelling so that fragments torn apart by
        # NEAREST interpolation get glued back into one component, then
        # multiply by the original mask to restore the true crack extent.
        # Without this the model learns to find crack pieces, not cracks.
        glued = binary_dilation(foreground, structure=CONNECTIVITY_8, iterations=2)
        labelled, n_components = ndimage.label(glued, structure=CONNECTIVITY_8)
        labelled = labelled * foreground

        masks, boxes = [], []
        for component_id in range(1, n_components + 1):
            instance = labelled == component_id
            ys, xs = np.where(instance)

            # NEW: after the multiplication above a component can be empty,
            # and xs.max() on an empty array raises. Guard first.
            if xs.size == 0:
                continue

            # CHANGED: was a check for zero extent only, which still let
            # 1 pixel wide boxes through. RoIAlign samples inside such a box
            # with a degenerate grid, and the dilation above can produce them.
            # Demanding at least MIN_BOX_EXTENT pixels in both directions
            # costs a few tiny fragments and removes a whole crash candidate.
            if (xs.max() - xs.min()) < MIN_BOX_EXTENT:
                continue
            if (ys.max() - ys.min()) < MIN_BOX_EXTENT:
                continue

            masks.append(instance)
            boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])

        if len(boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            masks_t = torch.zeros((0, *foreground.shape), dtype=torch.uint8)
            labels_t = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_t = torch.as_tensor(np.array(boxes), dtype=torch.float32)
            masks_t = torch.as_tensor(np.array(masks), dtype=torch.uint8)
            labels_t = torch.ones((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "masks": masks_t,
            "image_id": torch.tensor([idx]),
            "area": (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0]),
            "iscrowd": torch.zeros((len(boxes_t),), dtype=torch.int64),
        }

        return self.transform(image), target


def collate_fn(batch):
    """Instance counts differ per image, so targets stay a tuple of dicts."""
    return tuple(zip(*batch))


# Two dataset instances over the same files with identical split seeds produce
# identical index sets - the training half is augmented, validation is not.
data_augmented = CrackForestDataset(DATA_DIR, train_transforms, augment=True)
data_plain = CrackForestDataset(DATA_DIR, val_transforms, augment=False)

n_total = len(data_augmented)
n_val = int(n_total * VAL_FRACTION)
n_train = n_total - n_val

train_data, _ = random_split(
    data_augmented, [n_train, n_val], generator=torch.Generator().manual_seed(SEED)
)
_, val_data = random_split(
    data_plain, [n_train, n_val], generator=torch.Generator().manual_seed(SEED)
)

print(f"CrackForest: {n_total} pairs -> {n_train} train / {n_val} val")

# CHANGED: the module-level DataLoaders of the previous version were dead code -
# every training path builds its own. They are gone; make_loaders() replaces them.


def make_loaders(batch_size, num_workers, persistent_workers):
    """Single place that knows how a DataLoader for this project is built."""
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        # Unified memory - pinning buys nothing here
        pin_memory=False,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


# =========================================================
# Lightning module
# =========================================================


class CrackMaskRCNN(L.LightningModule):
    def __init__(
        self,
        num_classes=2,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005,
        step_size=7,
        gamma=0.1,
        max_epochs=SEARCH_EPOCHS,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = maskrcnn_resnet50_fpn_v2(
            weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
            # CHANGED: keep the internal rescaling close to the native 320x480
            min_size=MODEL_MIN_SIZE,
            max_size=MODEL_MAX_SIZE,
            # NEW: fewer trainable stages = less capacity to overfit 121 images
            trainable_backbone_layers=TRAINABLE_BACKBONE_LAYERS,
        )

        # Box head: 91 COCO classes -> background + crack
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Mask head: outputs a 28x28 image per RoI, not class scores
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, 256, num_classes
        )

        # NEW: default aspect ratios are (0.5, 1.0, 2.0), roughly square. The
        # bounding box of a 300px long, 3px wide diagonal crack is ~98%
        # background. Extreme ratios fit that shape much better - at the cost
        # of the pretrained RPN head, which has to be rebuilt from scratch.
        if USE_CRACK_ANCHORS:
            anchor_generator = AnchorGenerator(
                # one size tuple per FPN level (P2..P6)
                sizes=((16,), (32,), (64,), (128,), (256,)),
                aspect_ratios=((0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),) * 5,
            )
            self.model.rpn.anchor_generator = anchor_generator
            # the head must emit the new number of anchors per location;
            # fpn_v2 uses conv_depth=2 for its RPN head
            self.model.rpn.head = RPNHead(
                self.model.backbone.out_channels,
                anchor_generator.num_anchors_per_location()[0],
                conv_depth=2,
            )

        # NEW: dataset-wide pixel IoU on the merged crack map. Unlike the
        # multi-task loss this is deterministic, directly interpretable, and
        # comparable to published numbers (0.55-0.65 is solid on CrackForest).
        self.val_iou = BinaryJaccardIndex()

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch

        loss_dict = self.model(images, targets)
        loss = sum(loss_dict.values())

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

        # Loss branch: Mask R-CNN only returns losses in train mode. Safe here
        # because the backbone uses FrozenBatchNorm2d and there is no dropout.
        # Kept for monitoring only - note that RPN/RoI proposal sampling is
        # random, so this number is NOT reproducible run to run. That is
        # exactly why it must not be the optimization target.
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

        # NEW: IoU branch. eval mode returns predictions instead of losses.
        # Every predicted instance above the score threshold is OR-ed back
        # into a single binary crack map and compared against the merged
        # ground truth - which is the quantity we actually care about.
        #
        # CHANGED: forced back to fp32. Only the eval path runs box decoding
        # (exp() on the regression deltas) followed by NMS. bf16 has an 8 bit
        # mantissa, so a delta that is merely large in fp32 can decode to inf
        # here. An inf or NaN box coordinate makes the NMS CUDA kernel index
        # out of bounds, which surfaces as "illegal memory access" - usually
        # reported a few calls later, hence the confusing traceback.
        self.model.eval()
        with torch.autocast(device_type=self.device.type, enabled=False):
            with torch.no_grad():
                predictions = self.model(images)

        for prediction, target in zip(predictions, targets):
            height, width = target["masks"].shape[-2:]

            keep = prediction["scores"] > SCORE_THRESHOLD
            if keep.sum() == 0:
                predicted = torch.zeros((height, width), device=self.device)
            else:
                # masks arrive as (N, 1, H, W) float probabilities
                predicted = (prediction["masks"][keep, 0] > 0.5).any(dim=0).float()

            if target["masks"].numel():
                ground_truth = target["masks"].any(dim=0).long()
            else:
                ground_truth = torch.zeros(
                    (height, width), dtype=torch.long, device=self.device
                )

            self.val_iou.update(predicted, ground_truth)

        self.log("val_iou", self.val_iou, on_epoch=True, prog_bar=True)

        # NEW: upload one prediction overlay per interval so the result can be
        # inspected visually in the browser instead of guessed from a number.
        if (
            batch_idx == 0
            and LOG_IMAGE_EVERY_N_EPOCHS > 0
            and self.current_epoch % LOG_IMAGE_EVERY_N_EPOCHS == 0
        ):
            self._log_overlay(images[0], predictions[0], targets[0])

    def _log_overlay(self, image, prediction, target):
        """Red = prediction only, green = ground truth only, yellow = both."""
        # During the Optuna search logger=False, so trainer.logger is None and
        # this must be a no-op - without the guard every trial would crash here.
        if self.logger is None or not hasattr(self.logger, "run_id"):
            return

        # (C, H, W) float 0-1 -> (H, W, C) uint8. copy() because the channel
        # assignments below need a writable, contiguous array.
        rgb = (image.detach().cpu().numpy() * 255).astype(np.uint8)
        rgb = rgb.transpose(1, 2, 0).copy()

        keep = prediction["scores"] > SCORE_THRESHOLD
        if keep.sum():
            pred_mask = (prediction["masks"][keep, 0] > 0.5).any(dim=0).cpu().numpy()
        else:
            pred_mask = np.zeros(rgb.shape[:2], dtype=bool)

        if target["masks"].numel():
            gt_mask = target["masks"].any(dim=0).cpu().numpy().astype(bool)
        else:
            gt_mask = np.zeros_like(pred_mask)

        # saturating one channel each makes the overlap read as yellow
        rgb[..., 0] = np.where(pred_mask, 255, rgb[..., 0])
        rgb[..., 1] = np.where(gt_mask, 255, rgb[..., 1])

        self.logger.experiment.log_image(
            self.logger.run_id,
            Image.fromarray(rgb),
            f"overlay_epoch{self.current_epoch:03d}.png",
        )

    def configure_optimizers(self):
        optimizer = SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )

        # CHANGED: cosine annealing adapts to whatever epoch budget it is given,
        # so search and final run stay consistent without retuning step_size.
        if SCHEDULER == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs)
        else:
            scheduler = StepLR(
                optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma
            )

        return [optimizer], [scheduler]


def build_trainer(
    max_epochs, callbacks, enable_checkpointing=False, progress_bar=True, logger=False
):
    """All Trainer instances share these settings - logging and checkpointing
    stay off unless explicitly requested, so nothing writes files behind our back."""
    return L.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        accelerator="auto",
        devices="auto",
        precision=PRECISION,
        # CHANGED: clip gradients to keep box regression from exploding
        gradient_clip_val=GRAD_CLIP,
        # summary adds nothing once the architecture is known, and it warns
        # about bf16 on every single trial
        enable_model_summary=False,
        # CHANGED: was hardcoded False. The search still passes nothing, only
        # the runs worth keeping get a logger.
        logger=logger,
        enable_checkpointing=enable_checkpointing,
        enable_progress_bar=progress_bar,
    )


def make_logger(run_name, extra_tags=None):
    """NEW: one MLflow run per training run. Returns False when tracking is off,
    which is exactly what Trainer(logger=...) expects for "no logging"."""
    if not USE_MLFLOW:
        return False

    logger = MLFlowLogger(
        experiment_name=MLFLOW_EXPERIMENT,
        # without a run name MLflow generates something unmemorable
        run_name=run_name,
        tracking_uri=MLFLOW_URI,
    )

    # Settings that are not hyperparameters but change the result - without
    # these the run table cannot explain why two runs differ.
    tags = {
        "precision": PRECISION,
        "min_size": MODEL_MIN_SIZE,
        "max_size": MODEL_MAX_SIZE,
        "trainable_backbone_layers": TRAINABLE_BACKBONE_LAYERS,
        "scheduler": SCHEDULER,
        "score_threshold": SCORE_THRESHOLD,
        "crack_anchors": USE_CRACK_ANCHORS,
        "n_train": n_train,
        "n_val": n_val,
    }
    if extra_tags:
        tags.update(extra_tags)
    logger.log_hyperparams(tags)

    return logger


# =========================================================
# Noise floor measurement
# =========================================================
# NEW: run this before trusting any hyperparameter ranking. If the standard
# deviation across identical runs exceeds the gap between your best trials,
# the search was measuring randomness.


def measure_noise_floor(params, repeats=NOISE_REPEATS):
    results = []

    for run in range(repeats):
        L.seed_everything(SEED + run, workers=True)

        train_loader, val_loader = make_loaders(
            params["batch_size"], SEARCH_NUM_WORKERS, SEARCH_PERSISTENT_WORKERS
        )

        model = CrackMaskRCNN(max_epochs=SEARCH_EPOCHS, **params)

        # NEW: every repeat becomes its own MLflow run, so the five curves can
        # be overlaid in the browser. If they fan out widely, the spread IS the
        # answer - no hyperparameter difference below that band means anything.
        logger = make_logger(
            run_name=f"noise_seed{SEED + run}",
            extra_tags={"run_type": "noise_floor", "seed": SEED + run, **params},
        )

        trainer = build_trainer(
            SEARCH_EPOCHS, callbacks=[], progress_bar=False, logger=logger
        )
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        value = trainer.callback_metrics["val_iou"].item()
        results.append(value)
        print(f"[noise] run {run + 1}/{repeats}: val_iou = {value:.4f}")

    results = np.array(results)
    print("\n--- Noise floor ---")
    print(f"  mean : {results.mean():.4f}")
    print(f"  std  : {results.std():.4f}")
    print(f"  range: {results.min():.4f} .. {results.max():.4f}")
    print(
        "  Hyperparameter differences smaller than roughly 2x std "
        "are not distinguishable from noise."
    )
    return results


# =========================================================
# Optuna objective
# =========================================================


def objective(trial):
    # CHANGED: lower bound raised from 1e-4. Trial 1 at 1.4e-4 produced an IoU
    # of exactly 0.0 - a dead trial teaches TPE almost nothing, so the range
    # now starts where the previous runs actually showed life.
    lr = trial.suggest_float("lr", 1e-3, 3e-2, log=True)
    momentum = trial.suggest_float("momentum", 0.8, 0.95)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])

    # CHANGED: with cosine annealing these two dimensions disappear from the
    # search space entirely - 20 trials go much further over 4 dimensions
    # than over 6.
    if SCHEDULER == "step":
        step_size = trial.suggest_int("step_size", 3, 12)
        gamma = trial.suggest_float("gamma", 0.05, 0.5)
    else:
        step_size, gamma = 7, 0.1

    train_loader, val_loader = make_loaders(
        batch_size, SEARCH_NUM_WORKERS, SEARCH_PERSISTENT_WORKERS
    )

    model = CrackMaskRCNN(
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        step_size=step_size,
        gamma=gamma,
        max_epochs=SEARCH_EPOCHS,
    )

    trainer = build_trainer(
        SEARCH_EPOCHS,
        # CHANGED: pruning now watches the IoU
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_iou")],
        progress_bar=False,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return trainer.callback_metrics["val_iou"].item()


# =========================================================
# Main
# =========================================================

if RUN_MODE == "noise":
    # CHANGED: updated to the parameters the last search actually settled on
    measure_noise_floor(
        {
            "lr": 0.006962550818742679,
            "momentum": 0.8017918565617858,
            "weight_decay": 7.586227986640295e-05,
            "batch_size": 8,
        }
    )
    raise SystemExit(0)


# CHANGED: study persisted to SQLite with a seeded sampler. Three hours of
# search are no longer lost to a crash, and the run is reproducible.
study = optuna.create_study(
    direction="maximize",  # CHANGED: IoU is better when larger
    study_name=STUDY_NAME,
    storage=STUDY_STORAGE,
    load_if_exists=True,
    sampler=optuna.samplers.TPESampler(seed=SEED),
    # CHANGED: n_warmup_steps=5 pruned 14 of 20 trials around epoch 5-6, before
    # the learning rate schedule had done anything. Configurations that only
    # pay off late never got a chance.
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=15),
)
study.optimize(objective, n_trials=N_TRIALS)

print("Best hyperparameters found:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")
print(f"  -> val_iou: {study.best_value:.4f}")


# =========================================================
# Final training
# =========================================================

best = study.best_params

train_loader, val_loader = make_loaders(
    best["batch_size"], FINAL_NUM_WORKERS, FINAL_PERSISTENT_WORKERS
)

final_model = CrackMaskRCNN(
    lr=best["lr"],
    momentum=best["momentum"],
    weight_decay=best["weight_decay"],
    step_size=best.get("step_size", 7),
    gamma=best.get("gamma", 0.1),
    max_epochs=FINAL_EPOCHS,
)

# CHANGED: no rmtree. The directory holds checkpoints from other models, so it
# is only ever created if missing - never emptied.
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

checkpoint_callback = ModelCheckpoint(
    dirpath=CHECKPOINT_DIR,
    filename=CHECKPOINT_PREFIX
    + "epoch{epoch:02d}_valiou{val_iou:.4f}_valloss{val_loss:.4f}",
    # CHANGED: select on IoU, not on the noisy loss
    monitor="val_iou",
    mode="max",
    save_top_k=1,
    auto_insert_metric_name=False,
)

# CHANGED: patience raised - with only 30 validation images the metric still
# wobbles, and 5 epochs was tight enough to stop promising runs early.
early_stop = EarlyStopping(monitor="val_iou", patience=12, mode="max")

final_trainer = build_trainer(
    FINAL_EPOCHS,
    callbacks=[checkpoint_callback, early_stop],
    enable_checkpointing=True,
    # NEW: the run worth keeping gets full tracking including overlay images
    logger=make_logger(
        run_name=f"final_bb{TRAINABLE_BACKBONE_LAYERS}_ep{FINAL_EPOCHS}",
        extra_tags={"run_type": "final", "n_trials": N_TRIALS, **best},
    ),
)

final_trainer.fit(
    final_model, train_dataloaders=train_loader, val_dataloaders=val_loader
)

print(f"Best checkpoint saved at: {checkpoint_callback.best_model_path}")
print(f"Best val_iou: {checkpoint_callback.best_model_score:.4f}")


# =========================================================
# Checkpoint housekeeping
# =========================================================
# NEW: save first, then delete - and only within our own prefix. Files from
# other models, and any file whose score cannot be parsed, are left untouched.
# Across runs the single best val_iou survives; a worse new run deletes itself
# rather than the better older checkpoint.

# named group "score" pulls the IoU straight back out of the filename
SCORE_PATTERN = re.compile(r"_valiou(?P<score>\d+\.\d+)_")


def prune_checkpoints(directory, prefix, pattern):
    candidates = []

    for path in glob.glob(os.path.join(directory, prefix + "*.ckpt")):
        match = pattern.search(os.path.basename(path))
        if match is None:
            # e.g. checkpoints from the previous val_loss based revision
            print(f"Skipping (no parsable score): {os.path.basename(path)}")
            continue
        candidates.append((float(match.group("score")), path))

    if len(candidates) <= 1:
        return

    # highest IoU wins
    candidates.sort(key=lambda entry: entry[0], reverse=True)
    keeper_score, keeper_path = candidates[0]

    for score, path in candidates[1:]:
        os.remove(path)
        print(
            f"Removed older checkpoint (val_iou {score:.4f}): {os.path.basename(path)}"
        )

    print(f"Kept (val_iou {keeper_score:.4f}): {os.path.basename(keeper_path)}")


prune_checkpoints(CHECKPOINT_DIR, CHECKPOINT_PREFIX, SCORE_PATTERN)

# NEW: reminder of how to look at what just happened
print(
    "\nInspect in a browser (SSH tunnel: ssh -L 5000:localhost:5000 -L 8080:localhost:8080 ...):"
)
if USE_MLFLOW:
    print(f"  mlflow ui --backend-store-uri {MLFLOW_URI} --host 0.0.0.0 --port 5000")
print(f"  optuna-dashboard {STUDY_STORAGE} --host 0.0.0.0 --port 8080")
