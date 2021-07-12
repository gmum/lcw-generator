import torch
from metrics.cw import cw_normality, cw_sampling_silverman
from metrics.rec_err import mean_per_image_se
from metrics.swd import sliced_wasserstein_distance
from noise_creator import NoiseCreator


def get_cost_function(model: str, lambda_val: float, z_dim: int, noise_creator: NoiseCreator):

    def __recerr_plus_normality_index(normality_metric):
        return lambda x, z, y: mean_per_image_se(x, y) + lambda_val*normality_metric(z)

    def __recerr_plus_normality_index_log(normality_metric):
        return lambda x, z, y: mean_per_image_se(x, y) + lambda_val*torch.log(normality_metric(z) + 1e-7)

    cost_functions = {
        'ae': lambda x, _, y: mean_per_image_se(x, y),
        'ae_cw': lambda x, _, y: cw_sampling_silverman(x, y),
        'cwae': __recerr_plus_normality_index_log(cw_normality),
        'cw2': lambda x, z, y: torch.log(cw_normality(z)) + lambda_val*torch.log(cw_sampling_silverman(x, y)),
    }

    return cost_functions[model]
