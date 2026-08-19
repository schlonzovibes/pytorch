"""
DCGAN on FashionMNIST - cleaned up reference implementation.
Book: "PyTorch Kompakt" (Joe Papa), GAN chapter.

Changes vs. the book version:
  - dead 'from turtle import forward' removed
  - fake_images.detach() instead of backward(retain_graph=True)
  - drop_last=True so every batch matches the fixed label tensors
  - Normalize((0.5,), (0.5,)) instead of manual '*2 - 1'
  - saves netG AND netD (the book saves an undefined 'model')
  - headless matplotlib backend (no GUI window over SSH)
"""

import os

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

import matplotlib

matplotlib.use("Agg")  # write files instead of opening a window
import matplotlib.pyplot as plt


# --------------------------------------------------------------- config
CODING_SIZE = 100  # length of the latent noise vector z
BATCH_SIZE = 64  # small batches = many optimizer steps, GANs need those
IMAGE_SIZE = 64  # DCGAN wants powers of two
N_EPOCHS = 150
LR_G = 2e-4
LR_D = 5e-5  # deliberately slower than G, otherwise D wins
BETAS = (0.5, 0.999)  # beta1=0.5 is the DCGAN recommendation
REAL_LABEL = 0.9  # one-sided label smoothing, keeps D from saturating
DROPOUT = 0.25  # applied inside D only
SAMPLE_EVERY = 5  # write a sample image every N epochs

# all paths are relative to notebooks/, where this script lives
DATA_DIR = "../data"
CKPT_DIR = "../checkpoints"
IMG_DIR = "../images/dcgan_fashionMnist"
RUN_NAME = f"dcgan_fashionMnist_b{BATCH_SIZE}_e{N_EPOCHS}"

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

torch.manual_seed(0)


# ----------------------------------------------------------------- data
transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),  # -> [0.0, 1.0]
        transforms.Normalize((0.5,), (0.5,)),  # -> [-1.0, 1.0], matches Tanh
    ]
)

dataset = datasets.FashionMNIST(
    root=DATA_DIR, train=True, download=True, transform=transform
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=16,
    pin_memory=True,
    persistent_workers=True,
)

print(f"batches per epoch: {len(dataloader)}")


# ------------------------------------------------------------ generator
class Generator(nn.Module):
    """Maps a (N, CODING_SIZE, 1, 1) noise tensor to a (N, 1, 64, 64) image."""

    def __init__(self, coding_sz: int):
        super().__init__()
        self.net = nn.Sequential(
            # ConvTranspose2d(in_channels, out_channels, kernel, stride, padding)
            # bias=False because the following BatchNorm has its own shift term
            # channel widths follow the original DCGAN paper (512 -> 64)
            nn.ConvTranspose2d(coding_sz, 512, 4, 1, 0, bias=False),  # 1 -> 4
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  # 4 -> 8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # 8 -> 16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),  # 16 -> 32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),  # 32 -> 64
            nn.Tanh(),  # -> [-1, 1]
        )

    def forward(self, x):
        return self.net(x)


# -------------------------------------------------------- discriminator
class Discriminator(nn.Module):
    """Maps a (N, 1, 64, 64) image to one raw logit per image."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # no BatchNorm on the first layer (DCGAN guideline),
            # therefore this layer keeps its own bias term
            nn.Conv2d(1, 64, 4, 2, 1),  # 64 -> 32
            nn.LeakyReLU(0.2, inplace=True),
            # Dropout2d drops whole feature maps, not single pixels.
            # It weakens D on purpose so G keeps getting useful gradients.
            nn.Dropout2d(DROPOUT),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # 32 -> 16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(DROPOUT),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # 16 -> 8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(DROPOUT),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # 8 -> 4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(DROPOUT),
            # no Sigmoid here - raw logits go into BCEWithLogitsLoss
            nn.Conv2d(512, 1, 4, 1, 0),  # 4 -> 1
        )

    def forward(self, x):
        return self.net(x)


def weights_init(m):
    """DCGAN weight init: normal distribution, mean 0, std 0.02."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


netG = Generator(CODING_SIZE).to(device)
netD = Discriminator().to(device)

netG.apply(weights_init)
netD.apply(weights_init)


# ------------------------------------------------------ loss & optimizer
criterion = nn.BCEWithLogitsLoss()

optimizerG = optim.Adam(netG.parameters(), lr=LR_G, betas=BETAS)
optimizerD = optim.Adam(netD.parameters(), lr=LR_D, betas=BETAS)

# labels are created per batch, because the last batch of an epoch
# is usually shorter than BATCH_SIZE

# one fixed noise vector, drawn from its own generator with a fixed seed.
# that way the same latent point is used in every run, even after the model
# code changed and shifted the global random stream
sample_gen = torch.Generator(device=device).manual_seed(1234)
fixed_noise = torch.randn(1, CODING_SIZE, 1, 1, device=device, generator=sample_gen)


# ------------------------------------------------------------- training
G_losses, D_losses, D_real, D_fake = [], [], [], []

for epoch in range(N_EPOCHS):
    netG.train()
    netD.train()

    # per-epoch accumulators, averaged at the end of the epoch
    epoch_D, epoch_G, epoch_Dx, epoch_DGz = [], [], [], []

    for i, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device, non_blocking=True)

        # the last batch of an epoch may be shorter than BATCH_SIZE,
        # so everything that has to line up with it is sized from the data
        current_batch_size = real_images.size(0)
        real_labels = torch.full((current_batch_size,), REAL_LABEL, device=device)
        fake_labels = torch.zeros(current_batch_size, device=device)

        # --- 1) discriminator on the real batch -------------------------
        netD.zero_grad(set_to_none=True)
        output = netD(real_images).view(-1)
        errD_real = criterion(output, real_labels)
        errD_real.backward()
        # output is a logit now, sigmoid turns it back into a probability
        D_x = torch.sigmoid(output).mean().item()  # should drift towards 1.0

        # --- 2) discriminator on the fake batch -------------------------
        noise = torch.randn(current_batch_size, CODING_SIZE, 1, 1, device=device)
        fake_images = netG(noise)
        # .detach() cuts the graph here: no gradient reaches the generator,
        # and the generator's own graph stays intact for step 3
        output = netD(fake_images.detach()).view(-1)
        errD_fake = criterion(output, fake_labels)
        errD_fake.backward()
        D_G_z1 = torch.sigmoid(output).mean().item()  # should drift towards 0.0
        errD = errD_real + errD_fake
        optimizerD.step()

        # --- 3) generator: make D classify the fakes as real ------------
        netG.zero_grad(set_to_none=True)
        output = netD(fake_images).view(-1)  # no detach -> gradient flows into G
        # G aims for 1.0, not the smoothed label - smoothing only helps D
        errG = criterion(output, torch.ones(current_batch_size, device=device))
        errG.backward()
        D_G_z2 = torch.sigmoid(output).mean().item()
        optimizerG.step()

        G_losses.append(errG.item())
        D_losses.append(errD.item())
        D_real.append(D_x)
        D_fake.append(D_G_z2)

        epoch_G.append(errG.item())
        epoch_D.append(errD.item())
        epoch_Dx.append(D_x)
        epoch_DGz.append(D_G_z2)

        if i % 100 == 0:
            print(
                f"epoch {epoch + 1}/{N_EPOCHS} "
                f"batch {i}/{len(dataloader)} "
                f"loss_D {errD.item():.4f} loss_G {errG.item():.4f} "
                f"D(x) {D_x:.4f} D(G(z)) {D_G_z1:.4f} -> {D_G_z2:.4f}"
            )

    # epoch averages - less noisy than the value of a single batch
    mean_D = sum(epoch_D) / len(epoch_D)
    mean_G = sum(epoch_G) / len(epoch_G)
    mean_Dx = sum(epoch_Dx) / len(epoch_Dx)
    mean_DGz = sum(epoch_DGz) / len(epoch_DGz)

    print(
        f"--- epoch {epoch + 1}/{N_EPOCHS} done: "
        f"lossD {mean_D:.4f} lossG {mean_G:.4f} "
        f"D(x) {mean_Dx:.4f} D(G(z)) {mean_DGz:.4f}"
    )

    # single sample image, written every SAMPLE_EVERY epochs and on the last one
    if (epoch + 1) % SAMPLE_EVERY == 0 or epoch + 1 == N_EPOCHS:
        netG.eval()
        with torch.no_grad():
            sample = netG(fixed_noise).cpu()
        save_image(
            (sample + 1.0) / 2.0,
            os.path.join(IMG_DIR, f"{RUN_NAME}_epoch{epoch + 1:03d}.png"),
        )


# ---------------------------------------------------- save & visualize
# filename follows the Lightning ModelCheckpoint scheme used in the other
# scripts: <model>_<dataset>_epoch<NN>_<metric><value>_...
# note: 'epoch' is the loop variable and therefore zero-based, exactly like
# Lightning's {epoch:02d} placeholder
CKPT_NAME = (
    f"dcgan_fashionMnist_epoch{epoch:02d}"
    f"_lossD{mean_D:.4f}_lossG{mean_G:.4f}"
    f"_Dx{mean_Dx:.4f}_DGz{mean_DGz:.4f}"
)

torch.save(netG.state_dict(), os.path.join(CKPT_DIR, f"{CKPT_NAME}_netG.pt"))
torch.save(netD.state_dict(), os.path.join(CKPT_DIR, f"{CKPT_NAME}_netD.pt"))
print("checkpoints saved")

# reload later with:
#   netG.load_state_dict(torch.load(path, weights_only=True))

# final sample, upscaled so it stays readable in a terminal viewer
final = (sample + 1.0) / 2.0
final = torch.nn.functional.interpolate(final, scale_factor=4, mode="nearest")
save_image(final, os.path.join(IMG_DIR, f"{RUN_NAME}_sample.png"))

plt.figure(figsize=(10, 5))
plt.title("Generator and discriminator loss during training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("loss")
plt.legend()
plt.savefig(os.path.join(IMG_DIR, f"{RUN_NAME}_loss.png"))
plt.close()

plt.figure(figsize=(10, 5))
plt.title("Discriminator output")
plt.plot(D_real, label="D(real)")
plt.plot(D_fake, label="D(fake)")
plt.xlabel("iterations")
plt.ylabel("mean probability 'real'")
plt.legend()
plt.savefig(os.path.join(IMG_DIR, f"{RUN_NAME}_discriminator.png"))
plt.close()

print("plots written to", IMG_DIR)
