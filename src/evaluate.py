import os
import json
import hydra
from omegaconf import DictConfig
import torch
import numpy as np
from tqdm import tqdm
from transformer_lens import HookedTransformer
from sae_lens import SAE
from src.models import ConditionalDenoiser
from src.metrics import (continuation_ppl,
                         repetition_rate,
                         calculate_logit_margin
                         )
from src.steering import (additive_steering_hook,
                          denoiser_hook,
                          adaptive_norm_preserving_steering_hook,
                          slerp_steering_hook
                          )
from src.utils import set_seeds, get_hydra_output_dir
from src.plots import plot_pareto_comparison


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
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

    denoiser = None
    if cfg.evaluate.mode in ["denoiser", "both", "all"]:
        denoiser = ConditionalDenoiser(d_model=model.cfg.d_model).to(device)
        denoiser.load_state_dict(torch.load(cfg.evaluate.checkpoint_path, map_location=device))
        denoiser.eval()

    concept_prompts = cfg.evaluate.concept_prompts
    fluency_prompts = cfg.evaluate.fluency_prompts

    # sampling decoding
    num_samples = cfg.evaluate.get("num_samples", 3)
    temperature = cfg.evaluate.get("temperature", 0.7)
    top_p = cfg.evaluate.get("top_p", 0.9)

    if cfg.evaluate.mode == "baseline":
        modes = ["baseline"]
    elif cfg.evaluate.mode == "denoiser":
        modes = ["baseline", "denoiser"]
    elif cfg.evaluate.mode == "coast":
        modes = ["baseline", "coast"]
    # elif cfg.evaluate.mode == "slerp":
    #     modes = ["baseline", "slerp"]
    elif cfg.evaluate.mode == "all":
        modes = ["baseline", "denoiser", "coast"]
    else:
        modes = ["baseline", "denoiser"]

    results = {mode: [] for mode in modes}

    for alpha in tqdm(cfg.evaluate.alpha_range, desc="Evaluating alphas"):
        for mode in modes:
            if mode == "denoiser" and denoiser is None:
                continue

            if mode == "baseline":
                hook_fn = additive_steering_hook(alpha, steering_vector)
            elif mode == "denoiser":
                hook_fn = denoiser_hook(denoiser, steering_vector, alpha)
            elif mode == "coast":
                hook_fn = lambda resid_post, hook: \
                    adaptive_norm_preserving_steering_hook(resid_post, hook, alpha, steering_vector)
            elif mode == "slerp":
                hook_fn = lambda resid_post, hook: \
                    slerp_steering_hook(resid_post, hook, alpha, steering_vector, t=0.5)

            ppls_per_prompt = []
            rep_rates_per_prompt = []
            margins_per_prompt = []

            # 1. Fluency & Repetition
            for prompt in fluency_prompts:
                prompt_tokens = model.to_tokens([prompt], prepend_bos=True).to(device)
                prompt_len = prompt_tokens.shape[1]

                sample_ppls = []
                sample_rep_rates = []

                for sample_idx in range(num_samples):
                    with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
                        generated = model.generate(
                            [prompt],
                            max_new_tokens=cfg.evaluate.max_new_tokens,
                            verbose=False,
                            do_sample=True,
                            temperature=temperature,
                            top_p=top_p,
                            return_type="tokens"
                        )

                    # Sample PPL
                    ppl = continuation_ppl(model, generated, prompt_len)
                    sample_ppls.append(ppl)

                    # Sample epetition rate
                    cont_tokens = generated[:, prompt_len:]
                    cont_text = model.tokenizer.decode(cont_tokens[0], skip_special_tokens=True)
                    rep_rate = repetition_rate(cont_text, n=3)
                    sample_rep_rates.append(rep_rate)

                median_ppl = np.median(sample_ppls)
                mean_rep_rate = np.mean(sample_rep_rates)

                ppls_per_prompt.append(median_ppl)
                rep_rates_per_prompt.append(mean_rep_rate)

            overall_median_ppl = np.median(ppls_per_prompt)

            # 2. Concept Score
            for p_dict in concept_prompts:
                prompt_tokens = model.to_tokens([p_dict.prompt], prepend_bos=True).to(device)
                with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
                    margin = calculate_logit_margin(model, prompt_tokens, p_dict.target, p_dict.control)
                    margins_per_prompt.append(margin)

            results[mode].append({
                "alpha": alpha,
                "fluency": 1.0 / overall_median_ppl,
                "concept": sum(margins_per_prompt)/len(margins_per_prompt) if margins_per_prompt else 0.0,
                "ppl": overall_median_ppl,
                "rep": sum(rep_rates_per_prompt)/len(rep_rates_per_prompt),
                "ppls_per_prompt": ppls_per_prompt,
                "margins_per_prompt": margins_per_prompt,
                "num_samples": num_samples,
            })

    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    plot_pareto_comparison(results, out_dir)

    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    for mode in modes:
        print(f"\n{mode.upper()}:")
        print(f"{'Alpha':<6} | {'PPL':<8} | {'Fluency':<10} | {'Concept':<10} | {'Rep':<8}")
        print("-"*80)
        for r in results[mode]:
            print(f"{r['alpha']:<6.1f} | {r['ppl']:<8.2f} | {r['fluency']:<10.4f} | {r['concept']:<10.4f} | {r['rep']:<8.4f}")


if __name__ == "__main__":
    main()
