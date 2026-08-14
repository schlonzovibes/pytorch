import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import optuna
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10
from torchvision import transforms


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

trainloader = torch.utils.data.DataLoader(
    train_set,
    batch_size=768,
    shuffle=True,
    num_workers=16,
    pin_memory=True,
    persistent_workers=True,
)
# validation loader must not use samples from test data
# or information from test set will leak into training.
# the test samples  must be kept separate!
valloader = torch.utils.data.DataLoader(
    val_set,
    batch_size=768,
    shuffle=False,
    num_workers=16,
    pin_memory=True,
    persistent_workers=True,
)
testloader = torch.utils.data.DataLoader(
    test_data,
    batch_size=768,
    shuffle=False,
    num_workers=16,
    pin_memory=True,
    persistent_workers=True,
)


# ---------------------------------------------------------
# Modell
# ---------------------------------------------------------
class LeNet5Lightning(pl.LightningModule):
    def __init__(self, lr=0.001, momentum=0.9, optimizer_name="SGD"):
        super().__init__()
        self.save_hyperparameters()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum
        )


# ---------------------------------------------------------
# Optuna: Hyperparameter-Suche
# ---------------------------------------------------------


#  "trial" holds the suggested hyperparameters for this run
#  and reports them back to Optuna
def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    momentum = trial.suggest_float("momentum", 0.5, 0.99)

    # creates a fresh LeNet5Lightning instance and a Trainer
    # using this trial's suggested hyperparameters
    model = LeNet5Lightning(lr=lr, momentum=momentum)
    trainer = pl.Trainer(
        max_epochs=25,
        accelerator="auto",
        enable_progress_bar=True,
        logger=False,
    )
    trainer.fit(model, trainloader, valloader)

    # returns a single float (this trial's final loss)
    # this is what Optuna uses to compare trials -
    # not train_loss, to avoid selecting for overfitting
    return trainer.callback_metrics["val_loss"].item()


if __name__ == "__main__":
    # creates an Optuna study;
    # direction="minimize" means lower returned values are better
    study = optuna.create_study(direction="minimize")

    # passes the objective function itself (not its result)
    # Optuna calls it once per trial, each time with a new trial object
    study.optimize(objective, n_trials=25)

    print("-------------------------")
    print("Beste Hyperparameter:", study.best_params)
    print("Bester Loss:", study.best_value)

    # Finales Modell mit den besten gefundenen Werten trainieren
    best_model = LeNet5Lightning(
        # study.best_params: dictionary of the hyperparameters
        # from the trial with the lowest loss
        lr=study.best_params["lr"],
        momentum=study.best_params["momentum"],
    )

    # retrains from scratch with the best hyperparameters found
    # this time without Optuna's constraints (progress bar, logging enabled again)
    final_trainer = pl.Trainer(max_epochs=45, accelerator="auto")
    final_trainer.fit(best_model, trainloader)
