import numpy as np
import pandas as pd


def generate_stores(n_stores=200, n_pre=8, n_post=4, seed=None):
    """Синтетические недельные данные по магазинам: выручка и число чеков."""
    rng = np.random.default_rng(seed)
    n_weeks = n_pre + n_post

    base = rng.lognormal(mean=np.log(3_000_000), sigma=0.5, size=n_stores)
    avg_check = rng.normal(700, 80, size=n_stores)
    trend = rng.normal(0, 0.03, size=n_stores)
    season = 1 + 0.05 * np.sin(np.arange(n_weeks) / n_weeks * 2 * np.pi)

    weeks = np.arange(n_weeks)
    drift = 1 + np.outer(trend, weeks / n_weeks)
    noise = rng.normal(1, 0.08, size=(n_stores, n_weeks))
    revenue = base[:, None] * season[None, :] * drift * noise

    check_noise = rng.normal(1, 0.03, size=(n_stores, n_weeks))
    checks = np.round(revenue / (avg_check[:, None] * check_noise))

    df = pd.DataFrame({
        "store_id": np.repeat(np.arange(n_stores), n_weeks),
        "week": np.tile(weeks, n_stores),
        "revenue": revenue.ravel(),
        "checks": checks.ravel(),
    })
    df["period"] = np.where(df["week"] < n_pre, "pre", "post")
    return df


def apply_effect(df, treated_ids, lift):
    """Увеличивает выручку тестовых магазинов в пост-периоде на lift (доля)."""
    out = df.copy()
    out["revenue"] = out["revenue"].astype(float)
    mask = out["store_id"].isin(treated_ids) & (out["period"] == "post")
    out.loc[mask, "revenue"] *= 1 + lift
    return out


def store_level(df, treated_ids):
    """Агрегация до уровня магазина: средние за пред- и пост-период."""
    agg = (
        df.pivot_table(index="store_id", columns="period",
                       values=["revenue", "checks"], aggfunc="mean")
    )
    agg.columns = [f"{m}_{p}" for m, p in agg.columns]
    agg = agg.reset_index()
    agg["treated"] = agg["store_id"].isin(treated_ids).astype(int)
    return agg


def load_rossmann(data_dir="data"):
    """Недельные продажи магазинов Rossmann (Kaggle): только полные недели и магазины без пропусков."""
    daily = pd.read_csv(f"{data_dir}/train.csv", low_memory=False, parse_dates=["Date"])
    days = daily.groupby("Store").size()
    daily = daily[daily["Store"].isin(days[days >= days.max() - 1].index)]
    daily["week_start"] = daily["Date"].dt.to_period("W-SUN").dt.start_time
    weekly = (
        daily.groupby(["Store", "week_start"])
        .agg(revenue=("Sales", "sum"), checks=("Customers", "sum"), days=("Date", "size"))
        .reset_index()
    )
    weekly = weekly[weekly["days"] == 7].drop(columns="days")
    weekly = weekly.rename(columns={"Store": "store_id"})
    weeks = np.sort(weekly["week_start"].unique())
    weekly["week"] = np.searchsorted(weeks, weekly["week_start"])
    return weekly.sort_values(["store_id", "week"]).reset_index(drop=True)


def sample_window(weekly, n_stores=200, n_pre=8, n_post=4, seed=None):
    """Случайное окно из реальных данных: случайные магазины и случайная дата старта теста."""
    rng = np.random.default_rng(seed)
    stores = rng.choice(weekly["store_id"].unique(), size=n_stores, replace=False)
    start = rng.integers(0, weekly["week"].max() - n_pre - n_post + 2)
    df = weekly[
        weekly["store_id"].isin(stores)
        & weekly["week"].between(start, start + n_pre + n_post - 1)
    ].copy()
    df["week"] -= start
    df["period"] = np.where(df["week"] < n_pre, "pre", "post")
    ids = {s: i for i, s in enumerate(np.sort(stores))}
    df["store_id"] = df["store_id"].map(ids)
    return df.reset_index(drop=True)
