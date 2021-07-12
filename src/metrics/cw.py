import numpy
import torch
import math as m

# Based on https://github.com/gmum/cwae-pytorch/blob/master/src/metrics/cw.py


def cw_normality(X: torch.Tensor, y: torch.Tensor = None):
    assert len(X.size()) == 2

    N, D = X.size()

    if y is None:
        y = __silverman_rule_of_thumb_normal(N)

    K = 1.0/(2.0*D-3.0)

    A1 = __pairwise_distances(X)
    A = (1/(N**2)) * (1/torch.sqrt(y + K*A1)).sum()

    B1 = __euclidean_norm_squared(X, axis=1)
    B = (2/N)*((1/torch.sqrt(y + 0.5 + K*B1))).sum()

    return (1/m.sqrt(1+y)) + A - B


def cw_sampling_silverman(first_sample: torch.Tensor, second_sample: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        gamma = __silverman_rule_of_thumb_sample(torch.cat((first_sample, second_sample), 0))
    return cw_sampling(first_sample, second_sample, gamma)


def cw_sampling(first_sample: torch.Tensor, second_sample: torch.Tensor, y: torch.Tensor):
    first_sample = torch.flatten(first_sample, start_dim=1)
    second_sample = torch.flatten(second_sample, start_dim=1)

    assert len(first_sample.size()) == 2
    assert first_sample.size() == second_sample.size()

    N, D = first_sample.size()

    T = 1.0/(2.0*N*N*m.sqrt(m.pi*y))

    A0 = __pairwise_distances(first_sample)
    A = (__phi_sampling(A0/(4*y), D)).sum()

    B0 = __pairwise_distances(second_sample)
    B = (__phi_sampling(B0/(4*y), D)).sum()

    C0 = __pairwise_distances(first_sample, second_sample)
    C = (__phi_sampling(C0/(4*y), D)).sum()

    return T*(A + B - 2*C)


def __phi_sampling(s, D):
    return (1.0 + 4.0*s/(2.0*D-3))**(-0.5)


def __silverman_rule_of_thumb_sample(combined_sample: torch.Tensor):
    N = combined_sample.size(0)//2
    stddev = combined_sample.std()
    return (1.06*stddev*N**(-0.2))**2


def __pairwise_distances(x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, numpy.inf)


def __silverman_rule_of_thumb_normal(N: int):
    return (4/(3*N))**0.4


def __euclidean_norm_squared(X: torch.Tensor, axis: int = None):
    return (X**2).sum(axis)
