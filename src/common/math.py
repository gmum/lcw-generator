import torch


def pairwise_distances(x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
    if y is None:
        y = x
    return torch.cdist(x, y)**2


def euclidean_norm_squared(X: torch.Tensor, axis: int) -> torch.Tensor:
    return torch.linalg.norm(X, 2, axis)**2
