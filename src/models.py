import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalDenoiser(nn.Module):
    """
    Контекстно-зависимый корректор активаций.
    """
    def __init__(self, d_model: int, hidden_dim: int = 1024):
        super().__init__()
        # Вход: [h_norm (d), delta_norm (d), rel_strength (1), alpha (1)]
        input_dim = 2 * d_model + 2

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_model),
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        self.alpha_scale = nn.Parameter(torch.tensor(-2.0))

    def forward(self, h_steered: torch.Tensor, delta: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        h_steered: [batch, seq_len, d_model]
        delta: [batch, seq_len, d_model]
        alpha: float
        """
        h_norm = F.layer_norm(h_steered, h_steered.shape[-1:])
        delta_norm = F.layer_norm(delta, delta.shape[-1:])

        h_scale = h_steered.norm(dim=-1, keepdim=True)
        delta_scale = delta.norm(dim=-1, keepdim=True)
        rel_strength = delta_scale / (h_scale + 1e-6)

        alpha_tensor = torch.full_like(rel_strength, alpha)

        x = torch.cat([h_norm, delta_norm, rel_strength, alpha_tensor], dim=-1)

        correction = self.net(x)

        gate = torch.sigmoid(self.alpha_scale) * rel_strength

        return h_steered + gate * correction
