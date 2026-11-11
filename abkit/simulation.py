import numpy as np
import pandas as pd

from abkit.data import generate_stores, apply_effect, store_level
from abkit.design import random_split, paired_split
from abkit.methods import ttest, cuped, did, delta_ratio


def _store_week_ttest(df, treated_ids):
    post = df[(df["period"] == "post") & (df["checks"] > 0)]
    ratio = post["revenue"] / post["checks"]
    return ttest(ratio, post["store_id"].isin(treated_ids))


def run_simulations(n_sims=1000, lift=0.0, sampler=None, seed=0):
    """Повторяет эксперимент n_sims раз и возвращает p-value каждого метода.

    sampler(seed) должен возвращать данные в формате generate_stores;
    по умолчанию используются синтетические данные.
    """
    sampler = sampler or (lambda s: generate_stores(seed=s))
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_sims):
        s = int(rng.integers(1 << 31))
        df = sampler(s)
        pre = df[df["period"] == "pre"].groupby("store_id")["revenue"].mean()
        ids = pre.index.values

        for design, treated_ids in (
            ("random", random_split(ids, seed=s + 1)),
            ("paired", paired_split(ids, pre.values, seed=s + 1)),
        ):
            d = apply_effect(df, treated_ids, lift)
            st = store_level(d, treated_ids)
            y, x, t = st["revenue_post"], st["revenue_pre"], st["treated"]
            rows.append({
                "design": design,
                "ttest": ttest(y, t)["pvalue"],
                "did": did(y, x, t)["pvalue"],
                "cuped": cuped(y, x, t)["pvalue"],
                "check_delta": delta_ratio(st["revenue_post"], st["checks_post"], t)["pvalue"],
                "check_naive": _store_week_ttest(d, treated_ids)["pvalue"],
            })
    return pd.DataFrame(rows)
