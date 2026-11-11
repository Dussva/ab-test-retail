import numpy as np

from abkit import (
    generate_stores, apply_effect, store_level, random_split, paired_split,
    mde, sample_size, ttest, cuped, did, delta_ratio, bootstrap_ratio,
)


def make(lift=0.0, seed=0):
    df = generate_stores(seed=seed)
    ids = np.arange(df["store_id"].nunique())
    treated = random_split(ids, seed=seed + 1)
    return store_level(apply_effect(df, treated, lift), treated)


def test_split_sizes():
    ids = np.arange(100)
    assert len(random_split(ids, seed=0)) == 50
    pre = np.random.default_rng(0).normal(size=100)
    assert len(set(paired_split(ids, pre, seed=0))) == 50


def test_mde_and_sample_size_agree():
    n = sample_size(std=1.0, effect=0.2)
    assert mde(std=1.0, n_per_group=n) <= 0.2


def test_cuped_reduces_se():
    st = make()
    y, x, t = st["revenue_post"], st["revenue_pre"], st["treated"]
    assert cuped(y, x, t)["se"] < 0.5 * ttest(y, t)["se"]
    assert did(y, x, t)["se"] < 0.5 * ttest(y, t)["se"]


def test_cuped_finds_effect():
    st = make(lift=0.05)
    res = cuped(st["revenue_post"], st["revenue_pre"], st["treated"])
    assert res["pvalue"] < 0.05
    assert res["lift_ci_low"] < 0.05 < res["lift_ci_high"] + 0.02


def test_delta_matches_bootstrap():
    st = make()
    args = st["revenue_post"], st["checks_post"], st["treated"]
    d, b = delta_ratio(*args), bootstrap_ratio(*args, seed=0)
    assert abs(d["se"] - b["se"]) / d["se"] < 0.15


def test_aa_false_positive_rate():
    pvals = [cuped(s["revenue_post"], s["revenue_pre"], s["treated"])["pvalue"]
             for s in (make(seed=i) for i in range(200))]
    assert 0.01 < np.mean(np.array(pvals) < 0.05) < 0.10


def test_rossmann_window():
    import os
    import pytest
    if not os.path.exists("data/train.csv"):
        pytest.skip("нет данных Rossmann")
    from abkit import load_rossmann, sample_window
    w = sample_window(load_rossmann("data"), n_stores=50, seed=0)
    assert w["store_id"].nunique() == 50
    assert set(w["period"]) == {"pre", "post"}
    assert w["week"].max() == 11
