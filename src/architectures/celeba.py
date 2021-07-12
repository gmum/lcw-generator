import torch
import torch.nn as nn
from modules.view import View
from modules.map_tanh_zero_one import MapTanhZeroOne


class Encoder(nn.Module):
    def __init__(self, latent_size: int):
        super().__init__()
        self.__sequential_blocks = [
            nn.Conv2d(3, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.Flatten(start_dim=1),
            nn.Linear(1024*4*4, latent_size)
        ]
        self.main = nn.Sequential(*self.__sequential_blocks)

    def forward(self, input_images: torch.Tensor):
        assert input_images.size(1) == 3 and input_images.size(2) == 64 and input_images.size(3) == 64
        encoded_latent = self.main(input_images)
        return encoded_latent


class Decoder(nn.Module):
    def __init__(self, latent_size: int):
        super().__init__()
        self.__sequential_blocks = [
            nn.Linear(latent_size, 1024*8*8),
            View(-1, 1024, 8, 8),
            nn.ConvTranspose2d(1024, 512, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, 3, 1, 1),
            nn.Tanh(),
            MapTanhZeroOne()
        ]
        self.main = nn.Sequential(*self.__sequential_blocks)

    def forward(self, input_latent: torch.Tensor) -> torch.Tensor:
        decoded_images = self.main(input_latent)
        return decoded_images


class GeneratorBlock(nn.Module):

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.__sequential_blocks = [
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(True)
        ]

        self.main = nn.Sequential(*self.__sequential_blocks)

    def forward(self, input_data: torch.Tensor):
        return self.main(input_data)


class Generator(nn.Module):
    def __init__(self, noise_dim: int):
        super().__init__()
        self.__sequential_blocks = [
            GeneratorBlock(noise_dim, 512),
            GeneratorBlock(512, 512),
            GeneratorBlock(512, 512),
            GeneratorBlock(512, 512),
            nn.Linear(512, 1024*8*8),
            View(-1, 1024, 8, 8),
            nn.ConvTranspose2d(1024, 512, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, 3, 1, 1),
            nn.Tanh(),
            MapTanhZeroOne()
        ]
        self.main = nn.Sequential(*self.__sequential_blocks)

    def forward(self, input_noise: torch.Tensor) -> torch.Tensor:
        return self.main(input_noise)
