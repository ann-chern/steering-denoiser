import torch
from typing import Callable


def additive_steering_hook(alpha: float, vector: torch.Tensor) -> Callable:
    """Возвращает хук для обычного additive steering."""
    def hook(resid_post, hook):
        delta = (alpha * vector).view(1, 1, -1).expand_as(resid_post)
        return resid_post + delta
    return hook


def denoiser_hook(denoiser: torch.nn.Module, vector: torch.Tensor, alpha: float) -> Callable:
    """Возвращает хук для применения Denoiser."""
    def hook(resid_post, hook):
        batch_size, seq_len, d_model = resid_post.shape
        delta = (alpha * vector).view(1, 1, d_model).expand(batch_size,
                                                            seq_len,
                                                            d_model
                                                            ).to(resid_post.device)
        h_steered = resid_post + delta
        h_denoised = denoiser(h_steered, delta, alpha)
        return h_denoised
    return hook


def norm_preserving_steering_hook(resid_post, hook, alpha, steering_vector):
    """
    COAST-style norm-preserving steering.
    Сохраняет норму исходной активации.
    """
    h_clean = resid_post
    h_steered = h_clean + alpha * steering_vector

    norm_ratio = h_clean.norm(dim=-1, keepdim=True) / (h_steered.norm(dim=-1, keepdim=True) + 1e-6)
    h_corrected = h_steered * norm_ratio

    return h_corrected


def adaptive_norm_preserving_steering_hook(resid_post, hook, alpha, steering_vector):
    """
    Adaptive COAST: norm preservation только при больших alpha.
    """
    h_clean = resid_post
    h_steered = h_clean + alpha * steering_vector

    if alpha > 20:
        norm_ratio = h_clean.norm(dim=-1, keepdim=True) / (h_steered.norm(dim=-1, keepdim=True) + 1e-6)
        return h_steered * norm_ratio
    else:
        return h_steered


def slerp_steering_hook(resid_post, hook, alpha, steering_vector, t=0.5):
    """
    SLERP (Spherical Linear intERPolation) version.
    Более геометрически корректная интерполяция на сфере.
    """
    h_clean = resid_post
    v = steering_vector

    h_hat = h_clean / (h_clean.norm(dim=-1, keepdim=True) + 1e-6)
    v_hat = v / (v.norm() + 1e-6)

    # Угол между векторами
    cos_theta = torch.sum(h_hat * v_hat, dim=-1, keepdim=True)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)

    # SLERP
    sin_theta = torch.sin(theta)
    a = torch.sin((1 - t) * theta) / sin_theta
    b = torch.sin(t * theta) / sin_theta

    h_slerp = a * h_hat + b * v_hat.expand_as(h_hat)
    h_slerp = h_slerp / (h_slerp.norm(dim=-1, keepdim=True) + 1e-6)

    h_final = h_clean.norm(dim=-1, keepdim=True) * h_slerp + alpha * v

    norm_ratio = h_clean.norm(dim=-1, keepdim=True) / (h_final.norm(dim=-1, keepdim=True) + 1e-6)
    return h_final * norm_ratio
