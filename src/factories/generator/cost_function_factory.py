import torch
from metrics.cw import cw_sampling, cw_sampling_silverman
from metrics.swd import sliced_wasserstein_distance


def get_cost_function(model: str, gamma: float):
    if model == 'cwg_dynamic':
        return cw_sampling_silverman
    if model == 'cwg_fixed':
        return lambda x, y: cw_sampling(x, y, gamma)
    elif model == 'swg5000':
        return lambda x, y: sliced_wasserstein_distance(x, y, 5000)

    raise Exception('Invalid model')
