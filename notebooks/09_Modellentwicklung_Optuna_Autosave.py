import os
import re

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import optuna
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10
from torchvision import transforms


# ---------------------------------------------------------
# Datenvorbereitung
# ---------------------------------------------------------
# datasets werden im gemeinsamen Parent-Ordner data/ abgelegt
# einmal herunterladen, danach nur noch referenzieren
DATA_DIR = "../data/"
CIFAR10(root=DATA_DIR, train=True, download=True)
train_data_aug = CIFAR10(root=DATA_DIR, train=True, download=False)
train_data_plain = CIFAR10(root=DATA_DIR, train=True, download=False)
test_data = CIFAR10(root=DATA_DIR, train=False, download=False)

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
# since both point to the same dataset.
# solution: two train datasets, one augmented the other not.
# the split indices are computed once and reused for both, so train
# and val are guaranteed to contain the same samples.
n = len(train_data_plain)
indices = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
train_set = Subset(train_data_aug, indices[:40000])
val_set = Subset(train_data_plain, indices[40000:])

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
# the test samples must be kept separate!
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
    def __init__(self, lr=0.001, momentum=0.9):
        super().__init__()
        self.save_hyperparameters()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.criterion = nn.CrossEntropyLoss()
        self._train_losses = []
        self._val_losses = []
        self._val_accs = []

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
        self._train_losses.append(loss.detach())
        return loss

    def on_train_epoch_end(self):
        self.last_train_loss = torch.stack(self._train_losses).mean().item()
        self._train_losses.clear()

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        self._val_losses.append(loss.detach())
        self._val_accs.append(acc.detach())

    def on_validation_epoch_end(self):
        self.last_val_loss = torch.stack(self._val_losses).mean().item()
        self.last_val_acc = torch.stack(self._val_accs).mean().item()
        self._val_losses.clear()
        self._val_accs.clear()

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", acc, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum
        )


# ---------------------------------------------------------
# Optuna: Hyperparameter-Suche
# ---------------------------------------------------------


# "trial" holds the suggested hyperparameters for this run
# and reports them back to Optuna
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
        enable_checkpointing=False,
    )
    trainer.fit(model, trainloader, valloader)

    trial.set_user_attr("train_loss", model.last_train_loss)
    trial.set_user_attr("val_acc", model.last_val_acc)

    # returns a single float (this trial's validation loss)
    # this is what Optuna uses to compare trials -
    # not train_loss, to avoid selecting for overfitting
    return model.last_val_loss


# ---------------------------------------------------------
# Autosave: keeps only the best checkpoint on disk
# ---------------------------------------------------------
def save_best_model(
    model,
    train_loss,
    val_loss,
    checkpoint_dir="../checkpoints",
    prefix="lenet5_cifar10",
):
    # Saves model weights with train_loss and val_loss encoded in the filename.
    # If a previous checkpoint with this prefix exists and has a worse (higher) val_loss,
    # the old file is deleted and replaced by the new one.
    os.makedirs(checkpoint_dir, exist_ok=True)

    # robust parsing: tolerates any prefix and optional +/- signs
    pattern = re.compile(
        rf"{re.escape(prefix)}_train(?P<train>[+-]?[0-9]+(?:\.[0-9]+)?)"
        rf"_val(?P<val>[+-]?[0-9]+(?:\.[0-9]+)?)\.pth$"
    )

    best_existing_val_loss = None
    best_existing_file = None

    for filename in os.listdir(checkpoint_dir):
        match = pattern.fullmatch(filename)
        if not match:
            continue
        filepath = os.path.join(checkpoint_dir, filename)
        existing_val_loss = float(match.group("val"))
        if (
            best_existing_val_loss is None
            or existing_val_loss < best_existing_val_loss
        ):
            best_existing_val_loss = existing_val_loss
            best_existing_file = filepath

    # skip saving if the new result isn't better than what's already on disk
    if best_existing_val_loss is not None and val_loss >= best_existing_val_loss:
        print(
            f"New val_loss ({val_loss:.4f}) is not better than existing best "
            f"({best_existing_val_loss:.4f}). Not saving."
        )
        return

    new_filename = f"{prefix}_train{train_loss:.4f}_val{val_loss:.4f}.pth"
    new_filepath = os.path.join(checkpoint_dir, new_filename)
    torch.save(model.state_dict(), new_filepath)
    print(f"Saved new best model: {new_filepath}")

    if best_existing_file is not None:
        os.remove(best_existing_file)
        print(f"Deleted old checkpoint: {best_existing_file}")


# ---------------------------------------------------------
# Ausführung: Suche starten, finales Modell trainieren, speichern
# ---------------------------------------------------------
if __name__ == "__main__":
    pl.seed_everything(42)

    # creates an Optuna study;
    # direction="minimize" means lower returned values are better
    study = optuna.create_study(
        direction="minimize",
        study_name="lenet5_cifar10",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # passes the objective function itself (not its result)
    # Optuna calls it once per trial, each time with a new trial object
    study.optimize(objective, n_trials=20)

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

    # retrains from scratch with the best hyperparameters found,
    # this time over more epochs and with validation enabled
    final_trainer = pl.Trainer(
        max_epochs=50,
        accelerator="auto",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )
    final_trainer.fit(best_model, trainloader, valloader)

    # das Modell gegen das echte Test-Set bewerten (kein Val-Leak!)
    test_results = final_trainer.test(best_model, testloader)[0]
    print(
        f"Test-Loss: {test_results['test_loss']:.4f} | "
        f"Test-Accuracy: {test_results['test_acc']:.4f}"
    )

    save_best_model(best_model, best_model.last_train_loss, best_model.last_val_loss)
