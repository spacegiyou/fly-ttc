"""Deterministic balanced sampling and leakage-free clip splits."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sample_subset(df: pd.DataFrame, n_pos: int, n_neg: int, seed: int) -> pd.DataFrame:
    """Sample labels equally as requested, cycling scene/light strata for diversity.

    The resulting subset is deliberately diverse, not prevalence representative.
    IDs and strata are sorted before randomness, so input row order has no effect.
    """
    if n_pos < 0 or n_neg < 0 or n_pos + n_neg == 0:
        raise ValueError("Subset counts must be nonnegative and total at least one")
    rng = np.random.default_rng(seed)
    selected = []
    for label, count in ((1, n_pos), (0, n_neg)):
        pool = df.loc[df["label"].eq(label)].copy()
        if label == 1:
            pool = pool.loc[pool["time_of_event"].notna()]
        pool = pool.sort_values("video_id").reset_index(drop=True)
        if count > len(pool):
            raise ValueError(f"Requested {count} label={label} clips, only {len(pool)} eligible")
        columns = [c for c in ("scene", "light_conditions") if c in pool and pool[c].notna().any()]
        groups = []
        if columns:
            keys = pool[columns].fillna("Unknown").astype(str).agg("|".join, axis=1)
            for key in sorted(keys.unique()):
                groups.append(list(rng.permutation(pool.index[keys.eq(key)])))
        else:
            groups = [list(rng.permutation(pool.index))]
        # Shuffle stratum order; cycle through available groups to include rare conditions.
        groups = [groups[i] for i in rng.permutation(len(groups))]
        chosen = []
        while len(chosen) < count:
            for group in groups:
                if group and len(chosen) < count:
                    chosen.append(group.pop())
        selected.append(pool.loc[chosen])
    return pd.concat(selected, ignore_index=True).sort_values("video_id").reset_index(drop=True)


def stratified_split(df: pd.DataFrame, val_frac: float, seed: int) -> tuple[list[str], list[str]]:
    """Return evaluation IDs, validation IDs, with both classes in both partitions."""
    if not 0 < val_frac < 1:
        raise ValueError("val_frac must be between 0 and 1")
    if df["video_id"].duplicated().any():
        raise ValueError("Duplicate video IDs cannot be split safely")
    rng = np.random.default_rng(seed)
    validation, evaluation = [], []
    for label in (0, 1):
        ids = sorted(df.loc[df["label"].eq(label), "video_id"].astype(str))
        if len(ids) < 2:
            raise ValueError(f"At least two clips of label={label} are required for validation/evaluation")
        ids = list(rng.permutation(ids))
        n_val = min(len(ids) - 1, max(1, int(round(len(ids) * val_frac))))
        validation.extend(ids[:n_val])
        evaluation.extend(ids[n_val:])
    return sorted(evaluation), sorted(validation)
