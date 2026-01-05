import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Sinusoidal timestep embeddings. Timesteps expected shape: [batch].
    """
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, device=timesteps.device, dtype=torch.float32)
        * (-math.log(10000.0) / (half - 1))
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def make_beta_schedule(timesteps: int, schedule: str = "linear", beta_start: float = 1e-4, beta_end: float = 2e-2):
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, timesteps)
    if schedule == "cosine":
        steps = torch.arange(timesteps + 1, dtype=torch.float32)
        alphas_cumprod = torch.cos(((steps / timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(max=0.999)
    raise ValueError(f"Unknown beta schedule: {schedule}")


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = dim * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return self.net(x)


class DenoiseBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ff = FeedForward(dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cond):
        # Self attention over noisy sequence
        sa_out, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.dropout(sa_out)
        # Cross attention conditioned on MTCL features
        ca_out, _ = self.cross_attn(self.norm2(x), self.norm2(cond), self.norm2(cond))
        x = x + self.dropout(ca_out)
        # Feed-forward
        ff_out = self.ff(self.norm3(x))
        x = x + self.dropout(ff_out)
        return x


class ConditionalDiffusion(nn.Module):
    """
    Lightweight conditional denoiser used for DDPM-style training.
    """

    def __init__(
        self,
        num_nodes: int,
        cond_dim: int,
        model_dim: int = 64,
        num_heads: int = 4,
        depth: int = 3,
        timesteps: int = 100,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        cond_reconstruction: bool = True,
    ):
        super().__init__()
        self.timesteps = timesteps

        betas = make_beta_schedule(timesteps, beta_schedule, beta_start=beta_start, beta_end=beta_end)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))

        self.input_proj = nn.Linear(num_nodes, model_dim)
        self.cond_proj = nn.Linear(cond_dim, model_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

        self.blocks = nn.ModuleList([DenoiseBlock(model_dim, num_heads) for _ in range(depth)])
        self.out_proj = nn.Linear(model_dim, num_nodes)

        self.enable_cond_reconstruction = cond_reconstruction
        if cond_reconstruction:
            self.cond_recon = nn.Sequential(
                nn.Linear(cond_dim, model_dim),
                nn.GELU(),
                nn.Linear(model_dim, num_nodes),
            )

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        z_t: [B, L, N]
        t: [B] integer timesteps
        cond: [B, L, cond_dim]
        """
        time_emb = timestep_embedding(t, self.input_proj.out_features)
        time_emb = self.time_mlp(time_emb).unsqueeze(1)  # [B, 1, D]

        x = self.input_proj(z_t) + time_emb
        cond_emb = self.cond_proj(cond)

        for block in self.blocks:
            x = block(x, cond_emb)

        eps_pred = self.out_proj(x)
        return eps_pred

    def reconstruct_from_cond(self, cond: torch.Tensor) -> torch.Tensor:
        """
        Lightweight projection from condition features back to input space.
        cond: [B, L, cond_dim]
        """
        if not self.enable_cond_reconstruction:
            raise RuntimeError("Conditional reconstruction head is disabled.")
        return self.cond_recon(cond)
