
import torch
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from architecture.generator.linear_generator_block import LinearGeneratorBlock
from metrics.cw import cw_sampling_silverman


class SimpleCWGenerator(LightningModule):
    def __init__(self, latent_dim, noise_dim):
        super().__init__()
        self.__latent_dim = latent_dim
        self.__noise_dim = noise_dim
        self.generator = torch.nn.Sequential(
            LinearGeneratorBlock(self.noise_dim(), 128),
            LinearGeneratorBlock(128, 128),
            LinearGeneratorBlock(128, 128),
            LinearGeneratorBlock(128, 128),
            LinearGeneratorBlock(128, 128),
            nn.Linear(128, self.latent_dim())
        )

    def sample(self, count: int) -> torch.Tensor:
        with torch.no_grad():
            self.eval()
            noise = torch.randn((count, self.noise_dim()))
            decoded_points = self(noise)
            return decoded_points.detach().cpu()

    def noise_dim(self) -> int:
        return self.__noise_dim

    def latent_dim(self) -> int:
        return self.__latent_dim

    def forward(self, x):
        latent = self.generator(x)
        return latent

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def cost_function(self, x, y):
        cw_cost = cw_sampling_silverman(x, y)
        self.log('cw_cost', cw_cost, prog_bar=True)
        return cw_cost

    def training_step(self, batch, _):
        noise = torch.randn((batch.size(0), self.noise_dim())).cuda()
        y = self(noise)
        return self.cost_function(y, batch)
