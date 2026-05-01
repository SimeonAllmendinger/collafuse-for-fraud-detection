import os
import json
import torch
import torch.nn as nn
from src.DDPM.noise_predictor import NoisePredictor
from src.DDPM.beta_schedule import BetaSchedule
from src.DDPM.time_embedding import get_sinusoidal_embedding

class DDPM(nn.Module):
    def __init__(self, device, input_dim, T=800, beta_start=1e-4, beta_end=0.02, time_embed_dim=128, num_classes = 2, class_emb_dim = 128):
        super().__init__()
        self.T = T
        self.device = device
        self.class_embedding = nn.Embedding(num_classes, class_emb_dim)
        self.beta_schedule = BetaSchedule(T=T, beta_start=beta_start, beta_end=beta_end, device=device)
        self.noise_predictor = NoisePredictor(input_dim, time_embed_dim, class_emb_dim=class_emb_dim)
        self.time_embed_dim = time_embed_dim

    def forward_diffusion(self, x_0):
        """
        Forward diffusion process: Adds noise to the input x_0 over T steps.
        """
        noise = torch.randn_like(x_0)
        x_t = x_0

        for t in range(self.T):
            beta_t = self.beta_schedule.get(t)
            noise_t = noise * torch.sqrt(beta_t)
            x_t = x_t * (1 - beta_t) + noise_t

        return x_t, noise

    def reverse_diffusion(self, x_t, t, t_emb, noise_pred=None):
        """
        Reverse diffusion process: Predicts noise to remove for denoising at step t.
        """
        if noise_pred is None:
            noise_pred = self.noise_predictor(x_t, t_emb)
        # These two are for pre-computing the values that are in the formula for reducing the workload.
        
        current_alpha_bars = self.beta_schedule.alpha_bars.to(t.device)
        current_sqrt_one_minus_alpha_bars = self.beta_schedule.one_minus_alpha_bars
        alpha_bar_t = current_alpha_bars[t].unsqueeze(1)
        sqrt_one_minus_alpha_bar_t = current_sqrt_one_minus_alpha_bars[t].unsqueeze(1)
        x_0_pred = (x_t - sqrt_one_minus_alpha_bar_t * noise_pred) / torch.sqrt(alpha_bar_t)
        return x_0_pred
        
    
    def p_sample(self, x_t, t, t_emb, cond=None):
        """ 
        It performs one step of reverse diff process. x_t -> x_{t-1}. Used internally by the generate function below.
        """
        noise_pred = self.noise_predictor(x_t, t_emb, cond)
        alphas = self.beta_schedule.alphas.to(x_t.device)
        betas = self.beta_schedule.betas.to(x_t.device)
        alpha_bars = self.beta_schedule.alpha_bars.to(x_t.device)
        posterior_variance = self.beta_schedule.posterior_variance.to(x_t.device)
        alpha_t = alphas[t].unsqueeze(1)
        beta_t = betas[t].unsqueeze(1)
        alpha_bar_t = alpha_bars[t].unsqueeze(1)
        x_0_pred = self.reverse_diffusion(x_t, t, t_emb, noise_pred=noise_pred)
        mean = (x_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * noise_pred) / torch.sqrt(alpha_t)
        noise = torch.randn_like(x_t) if t[0] > 0 else torch.zeros_like(x_t)
        variance = posterior_variance[t].unsqueeze(1)
        x_prev = mean + torch.sqrt(variance) * noise
        return x_prev
            
    @torch.no_grad()
    def generate(self, shape, device, class_label = None, num_of_inference_steps=None):
        """
        Generate new samples by reversing the diffusion process.
        """
        if num_of_inference_steps is None:
            num_of_inference_steps = self.T
        x_t = torch.randn(shape).to(device) #Starting denoising from pure noise.

        if class_label is not None:
            if isinstance(class_label, int):
                class_label = torch.full((shape[0],), int(class_label), device=device, dtype=torch.long)
            else:
                class_label = torch.as_tensor(class_label, device=device, dtype=torch.long)
                if class_label.ndim == 0:
                    class_label = class_label.repeat(shape[0])
                elif class_label.shape[0] != shape[0]:
                    raise ValueError("class_label must be a scalar or have one value per sample")
        
        for t_idx in reversed(range(num_of_inference_steps)):  # Reverse diffusion
            t = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)
            t_emb = get_sinusoidal_embedding(t,
    embedding_dim=self.time_embed_dim  
)
            
            if class_label is not None:
                c_emb = self.class_embedding(class_label)
                x_t = self.p_sample(x_t, t, t_emb, cond=c_emb)
            else:
                x_t = self.p_sample(x_t, t, t_emb)
        return x_t
    

    
    @torch.no_grad()
    def reconstruct_x_0(self, x_0_original, reconstruction_steps = 3):
        batch_size = x_0_original.size(0)
        device = x_0_original.device
        current_sqrt_alpha_bars = self.beta_schedule.sqrt_alpha_bars.to(device)
        current_sqrt_one_minus_alpha_bars = self.beta_schedule.one_minus_alpha_bars.to(device)
        total_reconstruction_error = torch.zeros(batch_size, device=device)
        for _ in range(reconstruction_steps):
            t = torch.randint(1, self.T, (batch_size,), device=device, dtype=torch.long)
            noise = torch.randn_like(x_0_original)
            sqrt_alpha_bar_t = current_sqrt_alpha_bars[t].unsqueeze(1)
            sqrt_one_minus_alpha_bar_t = current_sqrt_one_minus_alpha_bars[t].unsqueeze(1)
            x_t = sqrt_alpha_bar_t * x_0_original + sqrt_one_minus_alpha_bar_t * noise
            t_emb = get_sinusoidal_embedding(t, embedding_dim=self.time_embed_dim) 
            predicted_noise = self.noise_predictor(x_t, t_emb)
            estimated_x0 = (x_t - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t
            reconstruction_error_per_sample = torch.mean((x_0_original - estimated_x0)**2, dim=1)
            total_reconstruction_error += reconstruction_error_per_sample
        anomaly_scores = total_reconstruction_error / reconstruction_steps
        return anomaly_scores
            
    
    @classmethod
    def from_pretrained(cls, client_model_save_path, **kwargs):
        """
        Loads a DDPM model from a pretrained directory or path.
        """
        if not os.path.isdir(client_model_save_path):
            raise ValueError(f"'{client_model_save_path}' is not a directory. "
                             "Please provide a directory containing 'config.json' and 'pytorch_model.bin'.")

        # 1. Instantiate the model
        model = cls(input_dim=120, T=800, beta_start=1e-4, beta_end=0.02, time_embed_dim=128)

        # 2. Load model weights
        if not os.path.exists(client_model_save_path):
            raise FileNotFoundError(f"Model weights file not found at {client_model_save_path}")

        map_location = kwargs.get('map_location', None)
        state_dict = torch.load(client_model_save_path, map_location=map_location)
        model.load_state_dict(state_dict, strict=True)
        print(f"Model weights loaded from {client_model_save_path}")

        model.eval()
        return model
