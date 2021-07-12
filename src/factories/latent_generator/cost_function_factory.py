from metrics.cw import cw_sampling_silverman


def get_cost_function(model: str):
    return cw_sampling_silverman
