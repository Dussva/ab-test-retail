import numpy as np
from scipy import stats


def random_split(store_ids, seed=None):
    """Простая рандомизация: половина магазинов в тест."""
    rng = np.random.default_rng(seed)
    ids = np.asarray(store_ids)
    return rng.choice(ids, size=len(ids) // 2, replace=False)


def paired_split(store_ids, pre_metric, seed=None):
    """Парная рандомизация: сортируем по метрике пред-периода, в каждой паре один магазин в тест."""
    rng = np.random.default_rng(seed)
    ids = np.asarray(store_ids)[np.argsort(pre_metric)]
    n_pairs = len(ids) // 2
    pairs = ids[: 2 * n_pairs].reshape(n_pairs, 2)
    pick = rng.integers(0, 2, size=n_pairs)
    return pairs[np.arange(n_pairs), pick]


def mde(std, n_per_group, alpha=0.05, power=0.8, variance_reduction=0.0):
    """Минимальный детектируемый эффект в абсолютных единицах метрики."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    sd = std * np.sqrt(1 - variance_reduction)
    return z * sd * np.sqrt(2 / n_per_group)


def sample_size(std, effect, alpha=0.05, power=0.8, variance_reduction=0.0):
    """Сколько объектов нужно в каждую группу, чтобы поймать effect."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    sd = std * np.sqrt(1 - variance_reduction)
    return int(np.ceil(2 * (z * sd / effect) ** 2))
