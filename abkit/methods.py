import numpy as np
from scipy import stats


def _result(effect, se, df, control_mean, alpha):
    t = effect / se
    p = 2 * stats.t.sf(abs(t), df)
    q = stats.t.ppf(1 - alpha / 2, df)
    return {
        "effect": effect,
        "ci_low": effect - q * se,
        "ci_high": effect + q * se,
        "lift": effect / control_mean,
        "lift_ci_low": (effect - q * se) / control_mean,
        "lift_ci_high": (effect + q * se) / control_mean,
        "pvalue": p,
        "se": se,
    }


def ttest(y, treated, alpha=0.05):
    """t-тест Уэлча на метрике пост-периода."""
    y, treated = np.asarray(y, float), np.asarray(treated, bool)
    a, b = y[treated], y[~treated]
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    return _result(a.mean() - b.mean(), se, df, b.mean(), alpha)


def cuped(y, x, treated, alpha=0.05):
    """CUPED: вычитаем из метрики ее предсказуемую по пред-периоду часть."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    theta = np.cov(y, x, ddof=1)[0, 1] / x.var(ddof=1)
    y_adj = y - theta * (x - x.mean())
    res = ttest(y_adj, treated, alpha)
    control_mean = y[~np.asarray(treated, bool)].mean()
    res = _result(res["effect"], res["se"], len(y) - 3, control_mean, alpha)
    res["theta"] = theta
    return res


def did(y, x, treated, alpha=0.05):
    """Difference-in-differences: t-тест на приросте post - pre."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    res = ttest(y - x, treated, alpha)
    control_mean = y[~np.asarray(treated, bool)].mean()
    return _result(res["effect"], res["se"], len(y) - 2, control_mean, alpha)


def _ratio_var(num, den):
    n = len(num)
    mx, my = num.mean(), den.mean()
    vx, vy = num.var(ddof=1), den.var(ddof=1)
    cxy = np.cov(num, den, ddof=1)[0, 1]
    var = (vx / my**2 - 2 * mx * cxy / my**3 + mx**2 * vy / my**4) / n
    return mx / my, var


def delta_ratio(num, den, treated, alpha=0.05):
    """Дельта-метод для ratio-метрики (например, средний чек = выручка / чеки)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    treated = np.asarray(treated, bool)
    rt, vt = _ratio_var(num[treated], den[treated])
    rc, vc = _ratio_var(num[~treated], den[~treated])
    se = np.sqrt(vt + vc)
    return _result(rt - rc, se, np.inf, rc, alpha)


def bootstrap_ratio(num, den, treated, n_boot=2000, alpha=0.05, seed=None):
    """Бутстрап по магазинам для разницы ratio-метрик."""
    rng = np.random.default_rng(seed)
    num, den = np.asarray(num, float), np.asarray(den, float)
    treated = np.asarray(treated, bool)
    it, ic = np.where(treated)[0], np.where(~treated)[0]
    bt = rng.choice(it, size=(n_boot, len(it)))
    bc = rng.choice(ic, size=(n_boot, len(ic)))
    diffs = num[bt].sum(1) / den[bt].sum(1) - num[bc].sum(1) / den[bc].sum(1)
    point = num[it].sum() / den[it].sum() - num[ic].sum() / den[ic].sum()
    low, high = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {"effect": point, "ci_low": low, "ci_high": high, "se": diffs.std(ddof=1)}
