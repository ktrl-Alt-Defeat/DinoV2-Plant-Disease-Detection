"""Reproducibility control.

Seeding every random number generator the project relies on, plus the cuDNN
algorithm-selection switches, is what makes two runs of the same configuration
comparable.
"""

import random
from typing import Final

import numpy as np
import torch

from src.logger import get_logger

#: NumPy's legacy global generator only accepts seeds in ``[0, 2**32)``.
MAX_SEED: Final[int] = 2**32 - 1

_LOGGER: Final = get_logger("seed")


def set_seed(seed: int, *, deterministic: bool = True, benchmark: bool = False) -> None:
    """Seed Python, NumPy and PyTorch, and set the cuDNN determinism switches.

    Args:
        seed: Seed shared by every generator. Must be in ``[0, MAX_SEED]``.
        deterministic: Value assigned to ``torch.backends.cudnn.deterministic``.
            ``True`` restricts cuDNN to reproducible algorithms.
        benchmark: Value assigned to ``torch.backends.cudnn.benchmark``. ``True``
            lets cuDNN autotune per input shape, which is faster but not
            reproducible.

    Raises:
        TypeError: If ``seed`` is not an integer.
        ValueError: If ``seed`` falls outside the supported range.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"Seed must be an integer, got {type(seed).__name__}.")
    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"Seed must be within [0, {MAX_SEED}], got {seed}.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark

    _LOGGER.debug(
        "Seed set to %d (cudnn.deterministic=%s, cudnn.benchmark=%s).",
        seed,
        deterministic,
        benchmark,
    )
