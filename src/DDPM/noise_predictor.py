import torch
import torch.nn as nn


class NoisePredictor(nn.Module):
    def __init__(self, input_dim, time_embed_dim=128, class_emb_dim=None):
        super().__init__()
        self.use_class = class_emb_dim is not None
        self.time_embed_dim = time_embed_dim
        self.class_emb_dim = class_emb_dim
        conditioning_dim = class_emb_dim if self.use_class else 0

        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, 16),
            nn.LeakyReLU(0.2)
        )

        #    with 3 layers (512-512-512) stated in the paper but for time saving purposes I use 16-16-16
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + 16 + conditioning_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, input_dim)
        )

    def forward(self, x, t_emb, c_emb=None):
        t_proj = self.time_mlp(t_emb)

        if self.use_class:
            if c_emb is None:
                c_emb = torch.zeros((x.shape[0], self.class_emb_dim), device=x.device, dtype=x.dtype)
            else:
                c_emb = c_emb.to(device=x.device, dtype=x.dtype)
            h = torch.cat([x, t_proj, c_emb], dim=1)
        else:
            h = torch.cat([x, t_proj], dim=1)

        return self.mlp(h)


if __name__ == '__main__':
    dummy_x = torch.randn(16, 154)  # 117 input dims
    dummy_t = torch.randn(16, 128)  # sinusoidal embedded time

    model = NoisePredictor(input_dim=154)
    out = model(dummy_x, dummy_t)
    print(out.shape)
