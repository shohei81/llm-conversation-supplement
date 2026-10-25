"""Statistical helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def icc1(values: pd.Series, groups: pd.Series) -> float:
    """ICC(1) from a one-way ANOVA: the share of variance between groups (speakers)."""
    df = pd.DataFrame({"y": values, "g": groups}).dropna()
    gb = df.groupby("g")["y"]
    sizes = gb.size()
    means = gb.mean()
    grand = df["y"].mean()
    n_groups = len(sizes)
    n_total = int(sizes.sum())
    ss_between = float((sizes * (means - grand) ** 2).sum())
    ss_within = float(((df["y"] - gb.transform("mean")) ** 2).sum())
    ms_between = ss_between / (n_groups - 1)
    ms_within = ss_within / (n_total - n_groups)
    k_bar = (n_total - (sizes**2).sum() / n_total) / (n_groups - 1)
    return float((ms_between - ms_within) / (ms_between + (k_bar - 1) * ms_within))


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Cohen's d, pooled SD."""
    na, nb = len(a), len(b)
    var = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    return float((a.mean() - b.mean()) / np.sqrt(var))
