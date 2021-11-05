from metrics.swd import sliced_wasserstein_distance, sliced_wasserstein_distance_normality
import torch
from metrics.cw import cw_normality, cw_sampling, cw_sampling_silverman
from metrics.rec_err import mean_per_image_se
from common.noise_creator import NoiseCreator
from metrics.mmd import mmd_penalty

def get_cost_function(model: str, lambda_val: float, z_dim: int, noise_creator: NoiseCreator, gamma_val: float):

    cost_functions = {
        'ae': lambda x, _, y: mean_per_image_se(x, y),
        'ae_cw_dynamic': lambda x, _, y: cw_sampling_silverman(x, y),
        'ae_cw_fixed': lambda x, _, y: cw_sampling(x, y, gamma_val),
        'cwae': lambda x, z, y: mean_per_image_se(x, y) + lambda_val*torch.log(cw_normality(z)),
        'swae': lambda x, z, y: mean_per_image_se(x, y) + lambda_val*sliced_wasserstein_distance_normality(z, noise_creator),
        'wae-mmd': lambda x, z, y: mean_per_image_se(x, y) + lambda_val*mmd_penalty(z, noise_creator),
        'cw2_fixed': lambda x, z, y: torch.log(cw_sampling(x, y, gamma_val)) + lambda_val*torch.log(cw_normality(z)),
        'cw2_dynamic': lambda x, z, y: torch.log(cw_sampling_silverman(x, y)) + lambda_val*torch.log(cw_normality(z)),
        'sw2': lambda x, z, y: sliced_wasserstein_distance(x, y, 5000) + lambda_val*sliced_wasserstein_distance_normality(z, noise_creator)
    }

    return cost_functions[model]
