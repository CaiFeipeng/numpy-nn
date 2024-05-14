try:
    import cupy as np
except:
    import numpy as np
    
from src.utils.schedulers import get_scheduler
import tqdm

class Diffusion:
    def __init__(self, model, timesteps, beta_start, beta_end, loss_func, schedule="linear", objective='pred_noise') -> None:
        self.model = model
        self.layers = model.layers
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule = schedule
        self.timesteps = timesteps
        self.objective = objective
        self.loss_func = loss_func
        
        self.betas = get_scheduler(schedule, beta_start, beta_end, timesteps)
        self.sqrt_betas = np.sqrt(self.betas)
        
        self.alphas = 1 - self.betas
        self.inv_sqrt_alphas = 1 / np.sqrt(self.alphas)
        
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1-self.alphas_cumprod)
        
        self.scaled_alphas = (1 - self.alphas) / self.sqrt_one_minus_alphas_cumprod
        
    def q_sample(self, x_start, timestep, noise):
        return self.sqrt_alphas_cumprod[timestep, None, None, None] * x_start \
             + self.sqrt_one_minus_alphas_cumprod[timestep, None, None, None] * noise
             
    def forward(self, inputs, training=True):
        x_0 = inputs
        
        timestep_selected = np.random.randint(1, self.timesteps, (inputs.shape[0]))
        noise = np.random.normal(size=inputs.shape)
        
        # noise sample
        x_t = self.q_sample(x_start=x_0, timestep = timestep_selected, noise = noise)
        
        model_out = self.model.forward(x_t, timestep_selected / self.timesteps, training = True)
        
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_0

        return model_out, target
        
    def __call__(self, inputs, training=True, *args, **kwds):
        return self.forward(inputs, training, *args, **kwds)
    
    def backward(self, dvalues):
        dvalues = self.model.backward(dvalues)
        return dvalues
    
    def ddpm_p_sample(self, num_samples, image_size):
        
        x_t = np.random.normal(size=(num_samples, *image_size))
        
        x_ts = [x_t]
        
        for t in tqdm(reversed(range(self.timesteps)), total=self.timesteps):
            noise = np.random.normal(size=x_t.shape) if t > 0 else 0
            epsilon = self.model.forward(x_t, np.array(t)/self.timesteps, training=False)
            
            x_t = self.inv_sqrt_alphas[t] * (x_t - self.scaled_alphas * epsilon) + self.sqrt_betas[t] * noise
            
            x_ts.append(x_t)
        
        return x_ts
            
    def ddim_p_sample(self, num_samples, image_size):
        """TBD"""
        x_t = np.random.normal(size=(num_samples, *image_size))
        
        for t in tqdm(reversed(range(self.timesteps)), total=self.timesteps):
            noise = np.random.normal(size=x_t.shape) if t > 0 else 0
            epsilon = self.model.forward(x_t, np.array(t)/self.timesteps, training=False)
            
            