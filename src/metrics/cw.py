from typing import Union
from common.math import euclidean_norm_squared, pairwise_distances
import torch
import math as m

# Based on https://github.com/gmum/cwae-pytorch/blob/master/src/metrics/cw.py


def cw_normality(X: torch.Tensor, y: Union[torch.Tensor, float] = None) -> torch.Tensor:
    assert len(X.size()) == 2

    N, D = X.size()

    if y is None:
        y = __silverman_rule_of_thumb_normal(N)

    K = 1.0/(2.0*D-3.0)

    A1 = pairwise_distances(X)
    A = (1/torch.sqrt(y + K*A1)).mean()

    B1 = euclidean_norm_squared(X, axis=1)
    B = 2*((1/torch.sqrt(y + 0.5 + K*B1))).mean()

    return (1/m.sqrt(1+y)) + A - B


def cw_sampling_silverman(first_sample: torch.Tensor, second_sample: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        stddev = second_sample.std()
        N = second_sample.size(0)
        gamma = silverman_rule_of_thumb(stddev, N)
    return cw_sampling(first_sample, second_sample, gamma)


def cw_sampling(first_sample: torch.Tensor, second_sample: torch.Tensor, y: Union[float, torch.Tensor]) -> torch.Tensor:
    first_sample = torch.flatten(first_sample, start_dim=1)
    second_sample = torch.flatten(second_sample, start_dim=1)

    assert len(first_sample.size()) == 2
    assert first_sample.size() == second_sample.size()

    _, D = first_sample.size()

    T = 1.0/(2.0*m.sqrt(m.pi*y))

    A0 = pairwise_distances(first_sample)
    A = (__phi_sampling(A0/(4*y), D)).mean()

    B0 = pairwise_distances(second_sample)
    B = (__phi_sampling(B0/(4*y), D)).mean()

    C0 = pairwise_distances(first_sample, second_sample)
    C = (__phi_sampling(C0/(4*y), D)).mean()

    return T*(A + B - 2*C)


def __phi_sampling(s: torch.Tensor, D: int) -> torch.Tensor:
    return (1.0 + 4.0*s/(2.0*D-3))**(-0.5)


def silverman_rule_of_thumb(stddev: torch.Tensor, N: int) -> torch.Tensor:
    return (1.06*stddev*N**(-0.2))**2


def __silverman_rule_of_thumb_normal(N: int) -> float:
    return (4/(3*N))**0.4
