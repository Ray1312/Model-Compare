"""Model architectures used in the Fashion-MNIST comparison.

Copied without modifying the class implementations from models/Comparison.ipynb.
Checkpoint keys and tensor shapes remain compatible with the completed run.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

class VanillaAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)

        return reconstructed.view(-1, 1, 28, 28)

class SparseAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)

        return reconstructed.view(-1, 1, 28, 28), z

class VAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        hidden = self.encoder(x)

        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)

        z = self.reparameterize(mu, logvar)

        reconstructed = self.decoder(z)
        reconstructed = reconstructed.view(-1, 1, 28, 28)

        return reconstructed, mu, logvar

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),

            nn.Linear(256, 512),
            nn.ReLU(),

            nn.Linear(512, 784),
            nn.Sigmoid()
        )

    def forward(self, z):
        generated = self.model(z)
        return generated.view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(),

            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.model(x)

class DCGenerator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()

        self.latent_dim = latent_dim

        self.model = nn.Sequential(
            nn.ConvTranspose2d(
                latent_dim, 128,
                kernel_size=7,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                128, 64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                64, 1,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.Sigmoid()
        )

    def forward(self, z):
        z = z.view(-1, self.latent_dim, 1, 1)
        return self.model(z)

class DCDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                1, 64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                64, 128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                128, 1,
                kernel_size=7,
                stride=1,
                padding=0
            )
        )

    def forward(self, x):
        output = self.model(x)
        return output.view(-1, 1)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2

        emb_factor = math.log(10000) / (half_dim - 1)

        emb = torch.exp(
            torch.arange(
                half_dim,
                device=t.device
            ) * -emb_factor
        )

        emb = t.float()[:, None] * emb[None, :]

        return torch.cat(
            [emb.sin(), emb.cos()],
            dim=1
        )

class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        time_emb_dim
    ):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                time_emb_dim,
                out_channels
            )
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.GroupNorm(8, out_channels)
        )

        self.residual_conv = (
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t_emb):
        h = self.block1(x)

        time_emb = self.time_mlp(t_emb)
        h = h + time_emb[:, :, None, None]

        h = self.block2(h)

        return F.silu(
            h + self.residual_conv(x)
        )

class DDPMUNet(nn.Module):
    def __init__(self, time_emb_dim=128):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.init_conv = nn.Conv2d(
            1, 32, kernel_size=3, padding=1
        )

        # Encoder

        self.down_block1 = ResBlock(
            32, 32, time_emb_dim
        )

        self.downsample1 = nn.Conv2d(
            32, 64,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.down_block2 = ResBlock(
            64, 64, time_emb_dim
        )

        self.downsample2 = nn.Conv2d(
            64, 128,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.mid_block = ResBlock(
            128, 128, time_emb_dim
        )

        # Decoder

        self.upsample1 = nn.ConvTranspose2d(
            128, 64,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.up_block1 = ResBlock(
            128, 64, time_emb_dim
        )

        self.upsample2 = nn.ConvTranspose2d(
            64, 32,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.up_block2 = ResBlock(
            64, 32, time_emb_dim
        )

        self.output_conv = nn.Conv2d(
            32, 1, kernel_size=1
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        x = self.init_conv(x)

        # Encode

        h1 = self.down_block1(x, t_emb)
        x = self.downsample1(h1)

        h2 = self.down_block2(x, t_emb)
        x = self.downsample2(h2)

        x = self.mid_block(x, t_emb)

        # Decode + Skip Connections

        x = self.upsample1(x)
        x = torch.cat([x, h2], dim=1)
        x = self.up_block1(x, t_emb)

        x = self.upsample2(x)
        x = torch.cat([x, h1], dim=1)
        x = self.up_block2(x, t_emb)

        return self.output_conv(x)

class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

        self.relu = nn.ReLU()

    def extract_features(self, x):
        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        z = self.relu(self.fc2(x))

        return z

    def forward(self, x):
        z = self.extract_features(x)
        return self.fc3(z)

class FrozenLatentDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def forward(self, z):
        reconstructed = self.decoder(z)

        return reconstructed.view(-1, 1, 28, 28)
