from metrics.cw import cw_sampling, cw_sampling_silverman


def get_cost_function(model: str, gamma_val: float):
    if model == 'cwg_dynamic':
        return cw_sampling_silverman
    elif model == 'cwg_fixed':
        return lambda x, y : cw_sampling(x, y, gamma_val)
    else:
        raise Exception('Unknown model')
