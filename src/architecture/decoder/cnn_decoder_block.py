import torch
import torch.nn as nn


class CnnDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.__sequential_blocks = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        ]
        self.main = nn.Sequential(*self.__sequential_blocks)

    def forward(self, input_latent: torch.Tensor) -> torch.Tensor:
        decoded_images = self.main(input_latent)
        return decoded_images
