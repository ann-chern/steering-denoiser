# src/screen_features.py
import hydra
from omegaconf import DictConfig
import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
from sklearn.metrics import roc_auc_score


FEATURE_PROMPTS = {
    15500: {  # second-third (ординарные числительные)
        "pos": ["The second", "The third", "The next", "The following"],
        "neg": ["The two", "The three", "The one", "The four"],
        "target_pos": [" second", " third"],
        "target_neg": [" two", " three"]
    },
    15900: {  # defense (защита/оборона)
        "pos": ["The defense", "The defenders", "The defending", "In defense of"],
        "neg": ["The prosecution", "The attackers", "The offense", "In attack of"],
        "target_pos": [" defense", " defenders"],
        "target_neg": [" prosecution", " attackers"]
    },
    16000: {  # kind (вид/добрый)
        "pos": ["This kind of", "That kind of", "In-kind", "What kind"],
        "neg": ["This type of", "That type of", "Cash", "What type"],
        "target_pos": [" kind"],
        "target_neg": [" type", " cruel"]
    },
    16100: {  # role (роль)
        "pos": ["play a role", "play an important role", "play a vital role", "played a key role"],
        "neg": ["play a game", "play a sport", "play music", "played well"],
        "target_pos": [" role"],
        "target_neg": [" game", " sport"]
    },
    16400: {  # traditional (традиционный)
        "pos": ["traditional marriage", "traditional medicine", "traditional roles", "traditional values"],
        "neg": ["modern marriage", "modern medicine", "modern roles", "modern values"],
        "target_pos": [" traditional"],
        "target_neg": [" modern", " conventional"]
    },
    16500: {  # knight (рыцарь - игровые контексты)
        "pos": ["Dragon Knight", "Silver Knight", "Sentry Knight", "Dark Knight"],
        "neg": ["Dragon Quest", "Silver Creek", "Sentry Tower", "Dark Forest"],
        "target_pos": [" Knight"],
        "target_neg": [" Quest", " Creek"]
    },
    16600: {  # medal (медаль/награда)
        "pos": ["gold medal", "silver medal", "bronze medal", "Medal of Honor"],
        "neg": ["gold standard", "silver lining", "bronze statue", "Honor Roll"],
        "target_pos": [" medal"],
        "target_neg": [" standard", " prize"]
    },
    16700: {  # plasma (плазма - научные/медицинские контексты)
        "pos": ["plasma levels", "plasma clearance", "blood plasma", "plasma membrane"],
        "neg": ["serum levels", "serum clearance", "blood serum", "cell membrane"],
        "target_pos": [" plasma"],
        "target_neg": [" serum", " liquid"]
    },
    17000: {  # as mentioned (как упомянуто)
        "pos": ["as mentioned", "as described", "as argued", "as noted"],
        "neg": ["as we know", "as it is", "as follows", "as shown"],
        "target_pos": [" mentioned", " described"],
        "target_neg": [" know", " follows"]
    },
    18500: {  # grade (оценка/класс/качество)
        "pos": ["high grade", "passing grade", "first grade", "grade V"],
        "neg": ["high quality", "passing score", "first class", "level V"],
        "target_pos": [" grade"],
        "target_neg": [" quality", " score"]
    },
    18530: {  # are/were (грамматическое число - множественное)
        "pos": ["The dogs", "The children", "The results", "The people", "The factors"],
        "neg": ["The dog", "The child", "The result", "The person", "The factor"],
        "target_pos": [" are", " were"],
        "target_neg": [" is", " was"]
    }
}


def compute_cohens_d(z_pos, z_neg, eps=1e-6):
    """Вычисляет Cohen's d effect size"""
    mu_pos = z_pos.mean()
    mu_neg = z_neg.mean()
    var_pos = z_pos.var(unbiased=False)
    var_neg = z_neg.var(unbiased=False)
    pooled_std = torch.sqrt(0.5 * (var_pos + var_neg) + eps)
    return ((mu_pos - mu_neg) / pooled_std).item()


def compute_auc(z_pos, z_neg):
    """Вычисляет AUROC для классификации positive/negative"""
    values = torch.cat([z_pos, z_neg]).cpu().detach().numpy()
    labels = [1] * len(z_pos) + [0] * len(z_neg)
    try:
        return roc_auc_score(labels, values)
    except ValueError:
        return 0.5


def compute_selectivity(z_pos, z_neg, eps=1e-6):
    """Вычисляет selectivity: насколько feature предпочитает positive"""
    mu_pos = z_pos.mean()
    mu_neg = z_neg.mean()
    return (mu_pos / (mu_pos + mu_neg + eps)).item()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(cfg.model_name, device=device)
    model.eval()

    sae_tuple = SAE.from_pretrained(
        release=cfg.sae_release,
        sae_id=cfg.sae_id,
        device=device
    )
    sae = sae_tuple[0] if isinstance(sae_tuple, tuple) else sae_tuple
    hook_name = f"blocks.{cfg.intervention_layer}.hook_resid_post"

    print("="*100)
    print("FEATURE SCREENING RESULTS")
    print("="*100)
    print(f"{'Feature':<10} | {'Name':<15} | {'Cohen d':<10} | {'AUROC':<8} | {'Select':<8} | {'Causal Δ':<10}")
    print("-"*100)

    results = []

    for feature_id, prompts in FEATURE_PROMPTS.items():
        def get_last_activations(texts):
            tokens = model.to_tokens(texts, prepend_bos=True).to(device)
            with torch.no_grad():
                cache = model.run_with_cache(tokens, names_filter=lambda n: n == hook_name)[1]
            h = cache[hook_name][:, -1, :]
            z = sae.encode(h)
            return z[:, feature_id]

        z_pos = get_last_activations(prompts["pos"])
        z_neg = get_last_activations(prompts["neg"])

        cohens_d = compute_cohens_d(z_pos, z_neg)
        auc = compute_auc(z_pos, z_neg)
        selectivity = compute_selectivity(z_pos, z_neg)

        v = sae.W_dec[feature_id].detach()
        v = v / v.norm()

        test_tokens = model.to_tokens([prompts["pos"][0]], prepend_bos=True).to(device)

        with torch.no_grad():
            base_logits = model(test_tokens)[0, -1]

        target_pos_ids = [model.to_single_token(t) for t in prompts["target_pos"]]
        target_neg_ids = [model.to_single_token(t) for t in prompts["target_neg"]]

        base_margin = torch.logsumexp(base_logits[target_pos_ids], dim=0) - \
                      torch.logsumexp(base_logits[target_neg_ids], dim=0)

        hook_fn = lambda resid, hook: resid + 10.0 * v
        with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
            with torch.no_grad():
                steered_logits = model(test_tokens)[0, -1]

        steered_margin = torch.logsumexp(steered_logits[target_pos_ids], dim=0) - \
                         torch.logsumexp(steered_logits[target_neg_ids], dim=0)

        causal_delta = (steered_margin - base_margin).item()

        results.append({
            "feature_id": feature_id,
            "name": prompts["target_pos"][0].strip(),
            "cohens_d": cohens_d,
            "auc": auc,
            "selectivity": selectivity,
            "causal_delta": causal_delta
        })

        print(f"{feature_id:<10} | {prompts['target_pos'][0].strip():<15} | {cohens_d:<+10.3f} | {auc:<8.3f} | {selectivity:<8.3f} | {causal_delta:<+10.3f}")

    print("="*100)

    results.sort(key=lambda x: abs(x["cohens_d"]) * x["auc"] * abs(x["causal_delta"]), reverse=True)

    print("\nTOP 5 FEATURES (by combined score):")
    print("-"*100)
    for i, r in enumerate(results[:5]):
        print(f"{i+1}. Feature {r['feature_id']} ({r['name']}): d={r['cohens_d']:+.3f}, AUROC={r['auc']:.3f}, Δ={r['causal_delta']:+.3f}")

    print("\n" + "="*100)
    print("RECOMMENDATIONS:")
    print("="*100)

    best_feature = results[0]
    print(f"\nBest candidate for steering: Feature {best_feature['feature_id']} ({best_feature['name']})")
    print(f"  - High contrastive selectivity (|d|={abs(best_feature['cohens_d']):.3f})")
    print(f"  - Good generalization (AUROC={best_feature['auc']:.3f})")
    print(f"  - Strong causal effect (Δ={best_feature['causal_delta']:+.3f})")

    negative_causal = [r for r in results if r["causal_delta"] < 0]
    if negative_causal:
        print(f"\n⚠️  Warning: {len(negative_causal)} features have negative causal delta.")


if __name__ == "__main__":
    main()
