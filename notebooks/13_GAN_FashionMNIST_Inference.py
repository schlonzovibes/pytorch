"""
Loads a trained DCGAN generator and writes several single images from
different random latent vectors. Used to check for mode collapse:
if every image shows the same object, the generator has collapsed.

usage:
    python3 _14_GAN_CheckSamples.py <checkpoint_filename>
    python3 _14_GAN_CheckSamples.py dcgan_fashionMnist_epoch99_..._netG.pt
"""

import os
import sys

import torch
import torch.nn as nn
from torchvision.utils import save_image

CODING_SIZE = 100
N_SAMPLES = 6
UPSCALE = 4              # nearest neighbour, only to help terminal viewers
SEED = 42                # fixed, so both checkpoints get the same latent points

CKPT_DIR = "../checkpoints"
IMG_DIR = "../images"

device = "cuda" if torch.cuda.is_available() else "cpu"


# generator definition must match the one used during training
class Generator(nn.Module):
    def __init__(self, coding_sz: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(coding_sz, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


if len(sys.argv) < 2:
    print("usage: python3 _14_GAN_CheckSamples.py <checkpoint_filename>")
    print("\navailable generator checkpoints:")
    for f in sorted(os.listdir(CKPT_DIR)):
        if f.endswith("_netG.pt"):
            print("   ", f)
    sys.exit(1)

ckpt_path = os.path.join(CKPT_DIR, sys.argv[1])
if not os.path.exists(ckpt_path):
    print(f"checkpoint not found: {ckpt_path}")
    sys.exit(1)

netG = Generator(CODING_SIZE).to(device)
netG.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
netG.eval()   # switches BatchNorm to its running statistics
print(f"loaded {sys.argv[1]}")

# same seed for every checkpoint -> the same latent points every time,
# so two checkpoints can be compared image by image
gen = torch.Generator(device=device).manual_seed(SEED)
noise = torch.randn(N_SAMPLES, CODING_SIZE, 1, 1, device=device, generator=gen)

with torch.no_grad():
    images = netG(noise).cpu()

images = (images + 1.0) / 2.0
images = torch.nn.functional.interpolate(images, scale_factor=UPSCALE,
                                         mode="nearest")

# strip the '_netG.pt' tail so the output names stay readable
stem = sys.argv[1].replace("_netG.pt", "")
os.makedirs(IMG_DIR, exist_ok=True)

for i in range(N_SAMPLES):
    out = os.path.join(IMG_DIR, f"check_{stem}_s{i:02d}.png")
    save_image(images[i], out)
    print("wrote", out)

print(f"\nview them with:  chafa {IMG_DIR}/check_{stem}_s00.png")
