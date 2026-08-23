import torch
from typing import List
from transformer_lens import HookedTransformer


def continuation_ppl(model: HookedTransformer,
                     generated_tokens: torch.Tensor,
                     prompt_len: int) -> float:
    """
    Считает Perplexity на сгенерированном продолжении.
    """
    if generated_tokens.shape[1] <= prompt_len:
        return float('nan')

    with torch.no_grad():
        logits = model(generated_tokens)

        shift_logits = logits[:, prompt_len - 1:-1, :]
        shift_targets = generated_tokens[:, prompt_len:]

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

        loss = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_targets.reshape(-1)
        )
        return torch.exp(loss.mean()).item()


def calculate_logit_margin(model: HookedTransformer,
                           tokens: torch.Tensor,
                           target_token: str,
                           control_token: str) -> float:
    """Считает logit(target) - logit(control) на последнем токене промпта."""
    with torch.no_grad():
        logits = model(tokens)[0, -1]
        target_id = model.to_single_token(target_token)
        control_id = model.to_single_token(control_token)
        return (logits[target_id] - logits[control_id]).item()


def repetition_rate(text: str, n: int = 3) -> float:
    """Доля повторяющихся n-грамм."""
    words = text.lower().split()
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    return 1.0 - len(set(ngrams)) / len(ngrams)


def calculate_dist_n(generated_texts: List[str], n: int = 2) -> float:
    """Dist-n — метрика разнообразия."""
    total_ngrams = 0
    unique_ngrams = set()
    for text in generated_texts:
        words = text.lower().split()
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i:i+n])
            unique_ngrams.add(ngram)
            total_ngrams += 1
    if total_ngrams == 0:
        return 0.0
    return len(unique_ngrams) / total_ngrams
