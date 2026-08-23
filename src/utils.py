import os
import random
import numpy as np
import torch


def set_seeds(seed: int = 42):
    """Фиксирует все случайные сиды для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_hydra_output_dir() -> str:
    """
    Возвращает путь к папке, созданной Hydra для текущего запуска.
    """
    return os.getcwd()
