import torch

class BetaSchedule:
    def __init__(self, device, T, beta_start=1e-4, beta_end=0.02):
        """
        Initializes a linear schedule for beta values from beta_start to beta_end.
        T: Number of diffusion steps
        beta_start: Starting value for beta
        beta_end: Ending value for beta
        """
        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.T, dtype=torch.float32, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        # As square root appears too often, I thought it may be beneficial to embed it within the BetaSchedule class
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        #Same for 1-alpha
        self.one_minus_alpha_bars = torch.sqrt(1.0-self.alpha_bars)
        # Append 1.0 for the beginning as it was 1.0 for x_0.
        alpha_bars_prev = torch.cat([torch.tensor([1.0], device=device), self.alpha_bars[:-1]])
        #It's used fo 
        self.posterior_variance = self.betas * (1.0 - alpha_bars_prev) / (1.0 - self.alpha_bars)
        #Clipping the variance for potential NaN issues.
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)
 

    def get(self, t):
        """ Returns beta value at time step t. """
        return self.betas[t]

# Example test
if __name__ == '__main__':
    schedule = BetaSchedule(T=800)
    print(schedule.get(0))  
    print(schedule.get(799))  
    print(schedule.betas)  
