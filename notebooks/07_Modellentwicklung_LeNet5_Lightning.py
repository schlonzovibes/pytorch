import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import random_split

# ---------------------------------------------------------
# Datenvorbereitung
# ---------------------------------------------------------
train_data_aug = CIFAR10(root="./train/", train=True, download=True)
train_data_plain = CIFAR10(root="./train/", train=True, download=True)
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

train_data_aug.transform = train_transforms
train_data_plain.transform = test_transforms
test_data.transform = test_transforms

# You cannot split and then set another transform to the val data set
# since both point to the same dataset
# solution: two train datasets, one augmented the other not
# both splitted with the same random seed
generator = torch.Generator().manual_seed(42)
train_set, _ = random_split(train_data_aug, [40000, 10000], generator=generator)
_, val_set = random_split(
    train_data_plain, [40000, 10000], generator=torch.Generator().manual_seed(42)
)

# pin_memory=True : reserves "pinned" RAM, faster Transfer CPU-GPU
# persistent_workers=True : keeps worker processes inbetween epochs alive, less overhead
trainloader = torch.utils.data.DataLoader(
    train_set,
    batch_size=128,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)
# validation loader must not use samples from test data
# or information from test set will leak into training.
# the test samples  must be kept separate!
valloader = torch.utils.data.DataLoader(
    val_set,
    batch_size=128,
    shuffle=False,
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


class LeNet5Lightning(pl.LightningModule):
    def __init__(self, lr=0.001, momentum=0.9):
        super().__init__()

        # from pl.LightningModule: saves lr, momentum in
        # "self.hparams" for later access, logging, checkpointing
        self.save_hyperparameters()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.criterion = nn.CrossEntropyLoss()

    # identical to the plain PyTorch version - Lightning doesn't change forward()
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # overwrites "training step" from pl.LightningModule
    # "trainer" calls this method once per batch
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    # overwrites "configure_optimizers" from pl.LightningModule
    def configure_optimizers(self):
        # self.parameters(): inherited from nn.Module, collects all trainable
        # weights from self.conv1, self.conv2, self.fc1... automatically
        return torch.optim.SGD(
            self.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum
        )


# Creates an object instance of the model
model = LeNet5Lightning(lr=0.001, momentum=0.9)
# Creates an object instance of the trainer
trainer = pl.Trainer(max_epochs=10, accelerator="auto")
# The training loop, using both objects for processing
trainer.fit(model, trainloader, valloader)
