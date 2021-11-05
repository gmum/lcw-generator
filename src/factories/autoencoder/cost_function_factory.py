import torch
from metrics.cw import cw_normality, cw_sampling, cw_sampling_silverman
from metrics.rec_err import mean_per_image_se
from common.noise_creator import NoiseCreator

def get_cost_function(model: str, lambda_val: float, z_dim: int, noise_creator: NoiseCreator, gamma_val: float):

    cost_functions = {
        'ae': lambda x, _, y: mean_per_image_se(x, y),
        'cwae': lambda x, z, y: mean_per_image_se(x, y) + lambda_val*torch.log(cw_normality(z)),
        'cw2_fixed': lambda x, z, y: torch.log(cw_sampling(x, y, gamma_val)) + lambda_val*torch.log(cw_normality(z)),
        'cw2_dynamic': lambda x, z, y: torch.log(cw_sampling_silverman(x, y)) + lambda_val*torch.log(cw_normality(z)),
    }

    return cost_functions[model]
