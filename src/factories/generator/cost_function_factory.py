from metrics.cw import cw_sampling_silverman
from metrics.swd import sliced_wasserstein_distance


def get_cost_function(model: str):
    if model == 'cwg':
        return cw_sampling_silverman
    elif model == 'swg5000':
        return lambda x, y: sliced_wasserstein_distance(x, y, 5000)

    raise Exception('Invalid model')
