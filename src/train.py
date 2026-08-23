import os
import logging
import hydra
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformer_lens import HookedTransformer
from sae_lens import SAE
from src.models import ConditionalDenoiser
from src.data import c4_dataloader
from src.utils import set_seeds, get_hydra_output_dir


log = logging.getLogger(__name__)


def get_alpha_safe(alpha_high):
    """Adaptive alpha scaling"""
    if alpha_high < 15:
        return alpha_high
    elif alpha_high < 30:
        return alpha_high * 0.85
    else:
        return alpha_high * 0.70


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s][%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler('train.log'),
            logging.StreamHandler()
        ]
    )

    set_seeds(cfg.seed)
    out_dir = get_hydra_output_dir()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(cfg.model.name, device=device)
    model.eval()

    sae_tuple = SAE.from_pretrained(release=cfg.sae.release, sae_id=cfg.sae.id, device=device)
    sae = sae_tuple[0] if isinstance(sae_tuple, tuple) else sae_tuple

    feature_id = cfg.sae.target_feature_id
    steering_vector = sae.W_dec[feature_id].detach().clone()
    steering_vector = steering_vector / steering_vector.norm()

    hook_name = f"blocks.{cfg.model.intervention_layer}.hook_resid_post"

    d_model = model.cfg.d_model
    denoiser = ConditionalDenoiser(d_model=d_model).to(device)

    optimizer_fn = hydra.utils.instantiate(cfg.optimizer)
    optimizer = optimizer_fn(params=denoiser.parameters())

    scheduler_fn = hydra.utils.instantiate(cfg.scheduler)
    scheduler = scheduler_fn(optimizer=optimizer)

    w_clean = cfg.train.loss_weights.clean
    w_id = cfg.train.loss_weights.identity

    dataloader = c4_dataloader(model, cfg.data)

    alpha_high_values = torch.tensor(cfg.train.alpha_range, device=device)

    print("Starting training...")
    global_step = 0
    pbar = tqdm(total=cfg.train.max_steps)

    for tokens in dataloader:
        if global_step >= cfg.train.max_steps:
            break

        tokens = tokens.to(device)

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == hook_name)
            h_clean = cache[hook_name]

        alpha_high = alpha_high_values[torch.randint(0, len(alpha_high_values), (1,))].item()

        # alpha_safe = 15.0
        # alpha_safe = max(15.0, alpha_high * 0.8)
        alpha_safe = get_alpha_safe(alpha_high)

        delta_high = (alpha_high * steering_vector).view(1, 1, d_model).expand_as(h_clean)
        delta_safe = (alpha_safe * steering_vector).view(1, 1, d_model).expand_as(h_clean)

        noise_scale = torch.empty(1, device=device).uniform_(0.0, 0.3).item()
        noise = torch.randn_like(h_clean) * noise_scale * (h_clean.norm(dim=-1, keepdim=True) / (d_model ** 0.5))

        h_input = h_clean + delta_high + noise
        h_target = h_clean + delta_safe

        h_denoised = denoiser(h_input, delta_high, alpha_high)

        loss_recon = F.mse_loss(h_denoised, h_target)

        h_identity = denoiser(h_clean, torch.zeros_like(delta_high), 0.0)
        loss_identity = F.mse_loss(h_identity, h_clean)

        # Итоговый loss
        loss = w_clean * loss_recon + w_id * loss_identity

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(denoiser.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if global_step % 50 == 0:
            log.info(f"Step {global_step}: "
                     f"loss={loss.item():.4f}, "
                     f"recon={loss_recon.item():.4f}, "
                     f"identity={loss_identity.item():.4f}, "
                     f"alpha_high={alpha_high:.1f} -> alpha_safe={alpha_safe:.1f}")

        global_step += 1
        pbar.update(1)

    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "denoiser.pt")
    torch.save(denoiser.state_dict(), ckpt_path)
    print(f"Training finished. Model saved to {ckpt_path}")


if __name__ == "__main__":
    main()
