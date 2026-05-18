"""Statistical utilities — kept numpy-only so they import without torch/skimage."""
import numpy as np


def bootstrap_ci(values, ci=0.95, n_boot=10000, seed=42):
    """Bootstrap percentile confidence interval for the mean.

    Args:
        values: 1-D array-like of per-image metric values.
        ci: Confidence level (default 0.95 → 95% CI).
        n_boot: Number of bootstrap resamples (default 10,000).
        seed: RNG seed for reproducibility.

    Returns:
        dict with keys:
            mean   — point estimate (sample mean)
            std    — sample std-dev (ddof=1)
            ci     — confidence level requested
            ci_lo  — lower bound of bootstrap percentile CI
            ci_hi  — upper bound of bootstrap percentile CI
            n      — number of input values
            n_boot — number of bootstrap resamples

    Method: percentile bootstrap. For each of n_boot iterations, resample
    `values` with replacement (size = len(values)) and record the mean.
    The (1-ci)/2 and 1-(1-ci)/2 quantiles of the resulting bootstrap means
    form the CI.
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci": ci,
                "ci_lo": float("nan"), "ci_hi": float("nan"),
                "n": 0, "n_boot": n_boot}

    # Vectorized bootstrap: draw an (n_boot × n) matrix of resampled indices
    # and reduce along axis 1. Faster than a Python loop.
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)

    alpha = (1.0 - ci) / 2.0
    return {
        "mean":   float(values.mean()),
        "std":    float(values.std(ddof=1)) if n > 1 else 0.0,
        "ci":     ci,
        "ci_lo":  float(np.quantile(boot_means, alpha)),
        "ci_hi":  float(np.quantile(boot_means, 1.0 - alpha)),
        "n":      int(n),
        "n_boot": int(n_boot),
    }


def format_metric(name, ci_dict, precision=3):
    """One-line human-readable formatter: 'PSNR: 25.32 [95% CI 25.13, 25.51]'."""
    return (f"{name}: {ci_dict['mean']:.{precision}f} "
            f"[{int(ci_dict['ci']*100)}% CI {ci_dict['ci_lo']:.{precision}f}, "
            f"{ci_dict['ci_hi']:.{precision}f}]")
