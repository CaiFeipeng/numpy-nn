try:
    import cupy as np
except:
    import numpy as np
    
def linear_scheduler(start, end, timestemps):
    return np.linspace(start, end, timestemps)

def quaratic_scheduler(start, end, timesteps):
    return np.linspace(start ** 0.5, end ** 0.5, timesteps) ** 2

def expoential_scheduler(start, end, timesteps):
    return np.geomspace(start, end, timesteps)

def cosine_scheduler(start, end, timesteps):
    s = 8e-3
    x = np.linspace(0, timesteps, timesteps+1)
    
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    
    return np.clip(betas, 0.01, 0.99)

def get_scheduler(scheduler, start, end, timesteps):
    if scheduler == 'linear':
        return linear_scheduler(start, end, timesteps)
    elif scheduler == 'quadratic':
        return quaratic_scheduler(start, end, timesteps)
    elif scheduler == 'exponential':
        return expoential_scheduler(start, end, timesteps)
    elif scheduler == 'cosine':
        return cosine_scheduler(start, end, timesteps)
    else:
        raise ValueError(f'Unknown schedule: {scheduler}')