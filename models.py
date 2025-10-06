import math
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


def smooth_bump(phi: torch.Tensor) -> torch.Tensor:
    """Smooth bump that peaks at phi=0 and vanishes at phi=\pm1."""
    safe_denom = torch.clamp(1.0 - phi ** 2, min=1e-6)
    bump_core = torch.exp(-1.0 / safe_denom) * math.e
    return torch.where(phi.abs() < 1.0, bump_core, torch.zeros_like(phi))


def positional_encoding(x: torch.Tensor, num_freqs: int) -> torch.Tensor:
    """Applies NeRF-style positional encoding without the base coordinates."""
    if num_freqs <= 0:
        return x
    device = x.device
    freq_bands = (2.0 ** torch.arange(num_freqs, device=device)) * math.pi
    xb = x.unsqueeze(-1) * freq_bands  # (..., 3, L)
    enc = torch.cat([torch.sin(xb), torch.cos(xb)], dim=-1)
    return enc.view(*x.shape[:-1], -1)


class PhaseFieldModel(nn.Module):
    """Neural phase-field NeRF variant with shared trunk and dual heads."""

    def __init__(self,
                 pos_freqs: int = 10,
                 dir_freqs: int = 4,
                 hidden_dim: int = 256,
                 num_trunk_layers: int = 8,
                 skip_layer: int = 4):
        super().__init__()
        self.pos_freqs = pos_freqs
        self.dir_freqs = dir_freqs
        pos_dim = 6 * pos_freqs
        dir_dim = 6 * dir_freqs

        layers = []
        for layer_idx in range(num_trunk_layers):
            if layer_idx == 0:
                in_dim = pos_dim
            elif layer_idx == skip_layer:
                in_dim = hidden_dim + pos_dim
            else:
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim))
        self.trunk = nn.ModuleList(layers)
        self.skip_layer = skip_layer
        self.act = nn.ReLU(inplace=True)

        self.phi_head = nn.Linear(hidden_dim, 1)
        self.color_head = nn.Sequential(
            nn.Linear(hidden_dim + dir_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluates phase field and color along positions and directions.

        Args:
            x: Sample positions, shape [N, 3].
            d: View directions, shape [N, 3]. Should already be normalized.

        Returns:
            phi: Phase field values in [-1, 1], shape [N, 1].
            rgb: Radiance colors in [0, 1], shape [N, 3].
            trunk_feat: Shared trunk features, shape [N, hidden_dim].
        """
        pos_enc = positional_encoding(x, self.pos_freqs)
        h = pos_enc
        for idx, layer in enumerate(self.trunk):
            if idx == self.skip_layer:
                h = torch.cat([h, pos_enc], dim=-1)
            h = self.act(layer(h))
        trunk_feat = h

        phi = torch.tanh(self.phi_head(trunk_feat))
        dir_enc = positional_encoding(d, self.dir_freqs)
        color_input = torch.cat([trunk_feat, dir_enc], dim=-1)
        rgb = torch.sigmoid(self.color_head(color_input))
        return phi, rgb, trunk_feat


def integrate_rays(model: PhaseFieldModel,
                   ray_origins: torch.Tensor,
                   ray_dirs: torch.Tensor,
                   num_samples: int,
                   near: float,
                   far: float,
                   eps_rend: float,
                   kappa: float,
                   create_graph: bool = False) -> dict:
    """Marches rays through the phase-field network and composites radiance."""
    device = ray_origins.device
    num_rays = ray_origins.shape[0]
    t_vals = torch.linspace(near, far, num_samples, device=device)
    delta_t = (far - near) / num_samples

    sample_positions = ray_origins.unsqueeze(1) + ray_dirs.unsqueeze(1) * t_vals.view(1, num_samples, 1)
    sample_dirs = ray_dirs.unsqueeze(1).expand(-1, num_samples, -1)

    flat_positions = sample_positions.reshape(-1, 3)
    flat_dirs = sample_dirs.reshape(-1, 3)
    flat_positions.requires_grad_(True)

    phi, rgb, _ = model(flat_positions, flat_dirs)
    phi = phi.view(num_rays, num_samples)
    rgb = rgb.view(num_rays, num_samples, 3)

    grad_outputs = torch.ones_like(phi)
    grads = torch.autograd.grad(
        outputs=phi,
        inputs=flat_positions,
        grad_outputs=grad_outputs,
        create_graph=create_graph,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grads = grads.view(num_rays, num_samples, 3)

    bump = smooth_bump(phi)
    weights_s = kappa * bump
    alpha = 1.0 - torch.exp(-weights_s * delta_t)

    trans = torch.cumprod(torch.cat([torch.ones((num_rays, 1), device=device), 1.0 - alpha + 1e-10], dim=1), dim=1)
    trans = trans[:, :-1]
    weights = trans * alpha

    rgb_map = torch.sum(weights.unsqueeze(-1) * rgb, dim=1)
    acc_map = weights.sum(dim=1)
    depth_map = torch.sum(weights * t_vals.view(1, num_samples), dim=1)

    return {
        "rgb": rgb_map,
        "phi": phi,
        "grads": grads,
        "weights": weights,
        "alpha": alpha,
        "acc": acc_map,
        "depth": depth_map,
        "delta_t": delta_t,
    }
