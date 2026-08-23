import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer
from typing import Iterator
from omegaconf import DictConfig


def c4_dataloader(
    model: HookedTransformer,
    data_cfg: DictConfig
) -> Iterator[torch.Tensor]:
    """
    Загружает датасет из конфига и возвращает батчи токенов.

    Args:
        model: HookedTransformer модель для токенизации
        data_cfg: конфигурация данных из configs/data/c4.yaml

    Yields:
        tokens: тензор [batch_size, seq_len]
    """
    dataset = load_dataset(
        data_cfg.dataset_name,
        data_cfg.config_name,
        split=data_cfg.split,
        streaming=data_cfg.streaming
    )

    batch_texts = []

    for sample in dataset:
        text = sample["text"].strip()

        if len(text) < data_cfg.min_text_length:
            continue

        batch_texts.append(text)

        if len(batch_texts) == data_cfg.batch_size:
            tokens = model.to_tokens(batch_texts, prepend_bos=True)

            if tokens.shape[1] > data_cfg.seq_len:
                tokens = tokens[:, :data_cfg.seq_len]
            else:
                pad_len = data_cfg.seq_len - tokens.shape[1]
                if pad_len > 0:
                    pad_id = model.tokenizer.pad_token_id or model.tokenizer.eos_token_id
                    padding = torch.full(
                        (tokens.shape[0], pad_len),
                        pad_id,
                        dtype=tokens.dtype,
                        device=tokens.device
                    )
                    tokens = torch.cat([tokens, padding], dim=1)

            yield tokens
            batch_texts = []
