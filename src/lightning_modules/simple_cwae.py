
import torch
import math as m
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from architecture.generator.linear_generator_block import LinearGeneratorBlock
from metrics.cw import cw_normality
from metrics.rec_err import mean_per_image_se
from common.math import pairwise_distances, euclidean_norm_squared


def __silverman_rule_of_thumb(N: int):
    return (4/(3*N))**0.4


def cw_1d(X: torch.Tensor, y: torch.Tensor = None):
    def N0(mean, variance):
        return 1.0/(torch.sqrt(2.0 * m.pi * variance)) * torch.exp((-(mean**2))/(2*variance))

    N = X.size(0)
    if y is None:
        y = torch.FloatTensor([__silverman_rule_of_thumb(N)]).cuda()

    A = X.unsqueeze(1) - X
    return (1.0/(N*N)) * N0(A, 2*y).sum() + N0(0.0, 2.0 + 2*y) - (2/N) * N0(X, 1.0 + 2*y).sum()


def cw_2d(X: torch.Tensor, y: float = None):
    def __phi(x):
        def __phi_f(s):
            t = s/7.5
            return torch.exp(-s/2) * (1 + 3.5156229*t**2 + 3.0899424*t**4 + 1.2067492*t**6 + 0.2659732*t**8
                                      + 0.0360768*t**10 + 0.0045813*t**12)

        def __phi_g(s):
            t = s/7.5
            return torch.sqrt(2/s) * (0.39894228 + 0.01328592*t**(-1) + 0.00225319*t**(-2) - 0.00157565*t**(-3)
                                      + 0.0091628*t**(-4) - 0.02057706*t**(-5) + 0.02635537*t**(-6) - 0.01647633*t**(-7)
                                      + 0.00392377*t**(-8))

        a = torch.FloatTensor([7.5]).cuda()
        return __phi_f(torch.minimum(x, a)) - __phi_f(a) + __phi_g(torch.maximum(x, a))

    N = X.size(0)
    if y is None:
        y = __silverman_rule_of_thumb(N)

    A = 1/(N*N*m.sqrt(y))
    B = 2.0/(N*m.sqrt(y+0.5))

    A1 = pairwise_distances(X)/(4*y)
    B1 = euclidean_norm_squared(X, axis=1)/(2+4*y)
    return 1/m.sqrt(1+y) + A*__phi(A1).sum() - B*__phi(B1).sum()


class SimpleCWAE(LightningModule):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.__latent_dim = latent_dim
        self.encoder = torch.nn.Sequential(
            LinearGeneratorBlock(2, 128),
            LinearGeneratorBlock(128, 128),
            LinearGeneratorBlock(128, 128),
            nn.Linear(128, self.latent_dim())
        )

        self.decoder = torch.nn.Sequential(
            LinearGeneratorBlock(self.latent_dim(), 128),
            LinearGeneratorBlock(128, 128),
            LinearGeneratorBlock(128, 128),
            nn.Linear(128, 2),
            nn.Tanh())

    def sample(self, count: int) -> torch.Tensor:
        with torch.no_grad():
            self.eval()
            noise = torch.randn((count, self.latent_dim()))
            decoded_points = self.decoder(noise)
            return decoded_points.detach().cpu()

    def latent_dim(self) -> int:
        return self.__latent_dim

    def forward(self, x):
        latent = self.encoder(x)
        y = self.decoder(latent)
        return latent, y

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def cost_function(self, x, latent, y):
        normality_index = torch.log(cw_normality(latent))
        rec_err = mean_per_image_se(x, y)
        lambda_val = 100

        self.log('rec_err', lambda_val*rec_err, prog_bar=True)
        self.log('normality', normality_index, prog_bar=True)

        return normality_index + lambda_val*rec_err

    def training_step(self, batch, _):
        latent, y = self(batch)
        return self.cost_function(batch, latent, y)
