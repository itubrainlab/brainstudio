"""Group comparisons of per-epoch values (within-recording, epoch-level statistics)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return np.nan
    j = 1 - 3 / (4 * (na + nb) - 9)
    return float((a.mean() - b.mean()) / sp * j)


def describe(values: dict[str, np.ndarray]) -> dict[str, dict]:
    out = {}
    for g, v in values.items():
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        n = len(v)
        out[g] = {"n": n, "mean": float(v.mean()) if n else np.nan,
                  "sd": float(v.std(ddof=1)) if n > 1 else np.nan,
                  "sem": float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan}
    return out


def compare(values: dict[str, np.ndarray]) -> dict:
    """Two groups: Welch t-test, Mann-Whitney U and Hedges' g (first − second).
    More groups: one-way ANOVA and Kruskal-Wallis."""
    vals = {g: np.asarray(v, float)[np.isfinite(np.asarray(v, float))]
            for g, v in values.items()}
    vals = {g: v for g, v in vals.items() if len(v) >= 2}
    if len(vals) < 2:
        return {"test": "", "stat": np.nan, "p": np.nan, "effect": np.nan, "p_np": np.nan}
    if len(vals) == 2:
        a, b = vals.values()
        t, p = st.ttest_ind(a, b, equal_var=False)
        u = st.mannwhitneyu(a, b, alternative="two-sided")
        return {"test": "Welch t", "stat": float(t), "p": float(p),
                "effect": hedges_g(a, b), "p_np": float(u.pvalue)}
    f, p = st.f_oneway(*vals.values())
    h = st.kruskal(*vals.values())
    return {"test": "ANOVA F", "stat": float(f), "p": float(p), "effect": np.nan,
            "p_np": float(h.pvalue)}


def fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values (NaNs kept)."""
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        from mne.stats import fdr_correction
        q[ok] = fdr_correction(p[ok])[1]
    return q


def comparison_table(per_epoch: pd.DataFrame, measures: list[str], groups: list[str],
                     group_col: str = "group") -> pd.DataFrame:
    """One row per measure: mean ± SEM per group, test, p, FDR q, effect size."""
    rows = []
    for m in measures:
        vals = {g: per_epoch.loc[per_epoch[group_col] == g, m].to_numpy(float)
                for g in groups}
        d = describe(vals)
        c = compare(vals)
        row = {"measure": m}
        for g in groups:
            row[f"{g} mean"] = d[g]["mean"]
            row[f"{g} sem"] = d[g]["sem"]
            row[f"{g} n"] = d[g]["n"]
        row.update({"test": c["test"], "stat": c["stat"], "p": c["p"],
                    "p (non-param.)": c["p_np"], "hedges g": c["effect"]})
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df.insert(df.columns.get_loc("p") + 1, "q (FDR)", fdr(df["p"].to_numpy()))
    return df
