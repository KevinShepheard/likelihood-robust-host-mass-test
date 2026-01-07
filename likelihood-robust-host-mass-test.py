"""
Likelihood-Robust Host-Mass Step Test for Type Ia Supernovae (Pantheon+ / SH0ES)

This script robustly fits a Student-t likelihood model to Type Ia supernova distance-modulus residuals,
testing for an environment-dependent zero-point offset (the host-galaxy mass step) using Pantheon+ or SH0ES data.
It distinguishes between *raw* and *post-corrected* magnitude logic, and models intrinsic scatter explicitly.

All magnitude offsets ΔM are reported in magnitudes.
No cosmological parameters enter the likelihood.

Raw vs post-corrected magnitude logic: "raw" uses the original apparent magnitude (e.g., mB),
allowing detection of a mass step present in the uncorrected data. "postcorr" uses a corrected magnitude
(e.g., m_b_corr), typically after upstream corrections; this tests for any *residual* step after corrections.

Reproducibility:
  - Required input: Pantheon_SH0ES.dat (whitespace-delimited table from Pantheon+ or SH0ES releases).
  - Deterministic run: RNG seed settable via --seed.
  - Emitted artifacts: nu_sweep.csv, nu_sweep_summary.json, deltaM_vs_nu.png, summary.json (all in --outdir).
"""

# Version corresponding to arXiv/GitHub paper release
__version__ = "1.0.0"

import numpy as np
import pandas as pd
from scipy.stats import t
from scipy.optimize import minimize
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import json

def load_shoes(path: str) -> pd.DataFrame:
    """Load Pantheon_SH0ES.dat or equivalent table."""
    df = pd.read_csv(path, sep=r"\s+", comment="#", header=0)
    print(f"Loaded data with columns: {list(df.columns)}")
    return df

def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    """Raise if any required columns are missing."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        avail = ", ".join(map(str, df.columns))
        raise KeyError(f"Missing required columns: {missing}. Available columns: [{avail}]")


def build_y_w(
    df: pd.DataFrame,
    *,
    y_mode: str,
    mu_col: str,
    mag_col: str,
    magerr_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct y (residual-like) and weights w from DataFrame and column choices."""
    if y_mode == "postcorr":
        mu_col_eff = mu_col
        mag_col_eff = mag_col
    elif y_mode == "raw":
        mu_col_eff = mu_col
        mag_col_eff = mag_col
    else:
        raise ValueError(f"Unknown --y-mode: {y_mode}")

    _require_cols(df, [mu_col_eff, mag_col_eff, magerr_col])

    mu = df[mu_col_eff].to_numpy(dtype=float)
    mag = df[mag_col_eff].to_numpy(dtype=float)
    magerr = df[magerr_col].to_numpy(dtype=float)
    # y = mu - mag is a relative magnitude residual; it carries no cosmological dependence.
    y = mu - mag
    w = 1.0 / (magerr ** 2)
    return y, w

def student_t_step_loglike(
    params: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    is_high_mass: np.ndarray,
    nu: float,
    *,
    sigma_int: float,
) -> float:
    """
    Log-likelihood for the Student-t host-mass step model.
    y: residuals (array)
    w: weights (1/error^2)
    is_high_mass: indicator array (0/1)
    nu: degrees of freedom
    sigma_int: intrinsic scatter (added in quadrature)
    params: [mu, delta_m]
    """
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    is_high_mass = np.asarray(is_high_mass, dtype=float)
    # delta_m is an additive zero-point shift, not a population split.
    mu, delta_m = float(params[0]), float(params[1])
    mean = mu + delta_m * is_high_mass
    sigma_meas = np.sqrt(1.0 / w)
    scale = np.sqrt(sigma_meas ** 2 + float(sigma_int) ** 2)
    logpdfs = t.logpdf(y, df=nu, loc=mean, scale=scale)
    return float(np.sum(logpdfs))

def fit_student_t_step(
    y: np.ndarray,
    w: np.ndarray,
    is_high_mass: np.ndarray,
    nu: float,
    *,
    sigma_int_fixed: float | None,
) -> tuple[np.ndarray, float]:
    """
    Fit the Student-t host-mass step model.
    Returns (params, sigma_int), where params = [mu, delta_m].
    If sigma_int_fixed is None, fit sigma_int > 0 by MLE.
    """
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    is_high_mass = np.asarray(is_high_mass, dtype=float)

    mu0 = float(np.sum(y * w) / np.sum(w))

    if sigma_int_fixed is not None:
        sigma_int0 = float(sigma_int_fixed)
        def neg_loglike(params: np.ndarray) -> float:
            return -student_t_step_loglike(params, y, w, is_high_mass, nu, sigma_int=sigma_int0)
        res = minimize(neg_loglike, x0=np.array([mu0, 0.0], dtype=float), method="L-BFGS-B")
        if not res.success:
            raise RuntimeError("Optimization failed: " + str(res.message))
        return np.asarray(res.x, dtype=float), sigma_int0

    # Fit sigma_int (positive) via log-parameterization.
    log_s0 = np.log(0.1)
    def neg_loglike_full(theta: np.ndarray) -> float:
        mu = float(theta[0])
        dm = float(theta[1])
        log_s = float(theta[2])
        s_int = float(np.exp(log_s))
        return -student_t_step_loglike(np.array([mu, dm], dtype=float), y, w, is_high_mass, nu, sigma_int=s_int)
    bounds = [
        (None, None),  # mu
        (None, None),  # delta_m
        (np.log(1e-6), np.log(1.0)),  # log sigma_int
    ]
    res = minimize(
        neg_loglike_full,
        x0=np.array([mu0, 0.0, log_s0], dtype=float),
        method="L-BFGS-B",
        bounds=bounds,
    )
    if not res.success:
        raise RuntimeError("Optimization failed: " + str(res.message))
    mu_hat, dm_hat, log_s_hat = map(float, res.x)
    return np.array([mu_hat, dm_hat], dtype=float), float(np.exp(log_s_hat))

def permutation_test_step(
    y: np.ndarray,
    w: np.ndarray,
    is_high_mass: np.ndarray,
    nu: float,
    n_perm: int,
    *,
    sigma_int_fixed: float | None,
) -> tuple[float, float, float]:
    """
    Permutation test for the host-mass step Δ_M.
    Returns (delta_m_hat, p_perm, sigma_int_hat)
    """
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    is_high_mass = np.asarray(is_high_mass, dtype=float)

    (mu_hat, delta_m_hat), sigma_int_hat = fit_student_t_step(
        y, w, is_high_mass, nu, sigma_int_fixed=sigma_int_fixed
    )

    count = 0
    n_eff = 0
    for _ in range(int(n_perm)):
        permuted = np.random.permutation(is_high_mass)
        try:
            (_, delta_m_perm), _ = fit_student_t_step(y, w, permuted, nu, sigma_int_fixed=sigma_int_fixed)
        except RuntimeError:
            continue
        n_eff += 1
        if abs(delta_m_perm) >= abs(delta_m_hat):
            count += 1
    denom = (n_eff + 1) if n_eff > 0 else (n_perm + 1)
    p_value = (count + 1) / denom
    return float(delta_m_hat), float(p_value), float(sigma_int_hat)

def profile_likelihood(
    y: np.ndarray,
    w: np.ndarray,
    is_high_mass: np.ndarray,
    nu: float,
    grid: np.ndarray,
    *,
    sigma_int_fixed: float | None,
) -> np.ndarray:
    """
    Profile likelihood for delta_m over a grid, profiling over mu and (optionally) sigma_int.
    Returns array of log-likelihood values.
    """
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    is_high_mass = np.asarray(is_high_mass, dtype=float)
    mu0 = float(np.sum(y * w) / np.sum(w))
    prof_likes: list[float] = []
    for delta_m in grid:
        dm = float(delta_m)
        if sigma_int_fixed is not None:
            s0 = float(sigma_int_fixed)
            def neg_loglike_mu(mu_arr: np.ndarray) -> float:
                mu = float(mu_arr[0])
                return -student_t_step_loglike(np.array([mu, dm], dtype=float), y, w, is_high_mass, nu, sigma_int=s0)
            res = minimize(neg_loglike_mu, x0=np.array([mu0], dtype=float), method="L-BFGS-B")
            prof_likes.append(float(-res.fun) if res.success else float("nan"))
            continue
        def neg_loglike_mu_logsig(theta: np.ndarray) -> float:
            mu = float(theta[0])
            log_s = float(theta[1])
            s_int = float(np.exp(log_s))
            return -student_t_step_loglike(np.array([mu, dm], dtype=float), y, w, is_high_mass, nu, sigma_int=s_int)
        bounds = [(None, None), (np.log(1e-6), np.log(1.0))]
        res = minimize(
            neg_loglike_mu_logsig,
            x0=np.array([mu0, np.log(0.1)], dtype=float),
            method="L-BFGS-B",
            bounds=bounds,
        )
        prof_likes.append(float(-res.fun) if res.success else float("nan"))
    return np.asarray(prof_likes, dtype=float)

def sweep_nu(
    y: np.ndarray,
    w: np.ndarray,
    is_high_mass: np.ndarray,
    nu_grid: list[float],
    n_perm: int,
    *,
    sigma_int_fixed: float | None,
) -> pd.DataFrame:
    """
    Sweep over nu (degrees of freedom) and report fit results, permutation p, profile likelihood interval.
    Returns DataFrame.
    """
    records: list[dict[str, float]] = []
    for nu in nu_grid:
        (mu_hat, delta_hat), sigma_int_hat = fit_student_t_step(y, w, is_high_mass, nu, sigma_int_fixed=sigma_int_fixed)
        delta_hat, p_perm, sigma_int_hat2 = permutation_test_step(
            y, w, is_high_mass, nu, n_perm, sigma_int_fixed=sigma_int_fixed
        )
        sigma_used = float(sigma_int_hat if sigma_int_fixed is not None else sigma_int_hat2)
        grid = np.linspace(delta_hat - 0.05, delta_hat + 0.05, 80)
        prof = profile_likelihood(y, w, is_high_mass, nu, grid, sigma_int_fixed=sigma_int_fixed)
        rel = prof - np.nanmax(prof)
        ok = np.where(rel >= -0.5)[0]
        if len(ok) > 1:
            lo = float(grid[ok[0]])
            hi = float(grid[ok[-1]])
        else:
            lo = float("nan")
            hi = float("nan")
        records.append({
            "nu": float(nu),
            "delta_m": float(delta_hat),
            "delta_lo": lo,
            "delta_hi": hi,
            "p_perm": float(p_perm),
            "sigma_int": float(sigma_used),
        })
    return pd.DataFrame.from_records(records)

def main():
    parser = argparse.ArgumentParser(description="Fit heavy-tailed population + host-mass zero-point step model.")
    parser.add_argument('--shoes-dat', required=True, help='Path to Pantheon_SH0ES.dat file')
    parser.add_argument('--mass-threshold', type=float, required=True, help='Host galaxy mass threshold for step')
    parser.add_argument('--nu', type=float, help='Degrees of freedom for Student-t distribution')
    parser.add_argument('--n-perm', type=int, default=1000, help='Number of permutations for permutation test')
    parser.add_argument('--outdir', type=Path, required=True, help='Output directory')
    parser.add_argument('--sweep', action='store_true', help='Run full degrees of freedom sweep ignoring --nu')
    parser.add_argument('--nu-grid', type=str, default='2,3,4,5,7,10,15,30,100,inf',
                        help='Comma-separated list of degrees of freedom values for sweep')
    parser.add_argument('--y-mode', choices=['postcorr', 'raw'], default='postcorr',
                        help='How to construct y. postcorr uses MU_SH0ES - m_b_corr; raw uses MU_SH0ES - mB.')
    parser.add_argument('--mu-col', default='MU_SH0ES', help='Column name for distance modulus (default: MU_SH0ES)')
    parser.add_argument('--mag-col-postcorr', default='m_b_corr', help='Magnitude-like column for y-mode=postcorr')
    parser.add_argument('--mag-col-raw', default='mB', help='Magnitude-like column for y-mode=raw')
    parser.add_argument('--magerr-col', default='MU_SH0ES_ERR_DIAG', help='Error column used for weights (default: MU_SH0ES_ERR_DIAG)')
    parser.add_argument('--sigma-int', type=float, default=None,
                        help='Fix intrinsic scatter sigma_int (same units as y). If omitted, fit sigma_int by MLE.')
    parser.add_argument('--seed', type=int, default=1337, help='RNG seed for reproducibility')
    args = parser.parse_args()

    np.random.seed(int(args.seed))
    seed_value = int(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    df = load_shoes(args.shoes_dat)

    if args.y_mode == 'postcorr':
        mag_col = args.mag_col_postcorr
        if mag_col == 'm_b_corr':
            print('NOTE: y-mode=postcorr with mag_col=m_b_corr may already include an upstream mass-step correction; this tests for any residual step beyond that correction.')
    else:
        mag_col = args.mag_col_raw

    y, w = build_y_w(df, y_mode=args.y_mode, mu_col=args.mu_col, mag_col=mag_col, magerr_col=args.magerr_col)
    mass = df['HOST_LOGMASS'].to_numpy(dtype=float)
    is_high_mass = (mass >= args.mass_threshold).astype(int)

    # Print run configuration summary for reproducibility
    config_line = (
        f"Run config: y_mode={args.y_mode}, mass_threshold={args.mass_threshold}, "
        f"{'sweep' if args.sweep else 'nu='+str(args.nu)}, "
        f"sigma_int={'fixed' if args.sigma_int is not None else 'fitted'}"
    )
    print(config_line)

    if args.sweep:
        nu_strs = [s.strip() for s in args.nu_grid.split(',')]
        nu_grid = []
        for s in nu_strs:
            if s.lower() == 'inf':
                nu_grid.append(np.inf)
            else:
                nu_grid.append(float(s))
        nu_eval = [1e6 if np.isinf(nu) else nu for nu in nu_grid]
        df_sweep = sweep_nu(y, w, is_high_mass, nu_eval, args.n_perm, sigma_int_fixed=args.sigma_int)
        df_sweep["nu_label"] = nu_grid
        df_sweep.to_csv(args.outdir / "nu_sweep.csv", index=False)
        plt.figure(figsize=(6,4))
        plt.errorbar(
            df_sweep["nu_label"],
            df_sweep["delta_m"],
            yerr=[
                df_sweep["delta_m"] - df_sweep["delta_lo"],
                df_sweep["delta_hi"] - df_sweep["delta_m"]
            ],
            fmt="o-"
        )
        plt.axhline(0.0, color="k", linestyle="--", alpha=0.6)
        plt.xscale("symlog")
        plt.xlabel("Student-t degrees of freedom ν")
        plt.ylabel("Host-mass step Δ_M")
        title_extra = f"sigma_int={'fixed' if args.sigma_int is not None else 'fitted'}"
        plt.title(title_extra)
        plt.tight_layout()
        plt.savefig(args.outdir / "deltaM_vs_nu.png")
        plt.close()
        run_cfg = {
            'shoes_dat': str(args.shoes_dat),
            'y_mode': args.y_mode,
            'mu_col': args.mu_col,
            'mag_col': mag_col,
            'magerr_col': args.magerr_col,
            'mass_threshold': float(args.mass_threshold),
            'sigma_int_fixed': (float(args.sigma_int) if args.sigma_int is not None else None),
            'nu_grid': [float(nu) if not np.isinf(nu) else 'inf' for nu in nu_grid],
            'n_perm': int(args.n_perm),
            'seed': int(seed_value),
        }
        with open(args.outdir / 'run_config.json', 'w') as f:
            json.dump(run_cfg, f, indent=2)
        summary = {
            "mass_threshold": float(args.mass_threshold),
            "sigma_int_fixed": (float(args.sigma_int) if args.sigma_int is not None else None),
            "nu_grid": [float(nu) if not np.isinf(nu) else "inf" for nu in nu_grid],
            "n_perm": int(args.n_perm),
            "seed": int(seed_value),
            "results": [],
        }
        for _, row in df_sweep.iterrows():
            lo = float(row["delta_lo"]) if np.isfinite(row["delta_lo"]) else float("nan")
            hi = float(row["delta_hi"]) if np.isfinite(row["delta_hi"]) else float("nan")
            zero_in_interval = bool((lo <= 0.0) and (hi >= 0.0)) if (np.isfinite(lo) and np.isfinite(hi)) else False
            summary["results"].append({
                "nu": float(row["nu"]),
                "delta_m": float(row["delta_m"]),
                "sigma_int": float(row["sigma_int"]),
                "p_perm": float(row["p_perm"]),
                "delta_lo": lo,
                "delta_hi": hi,
                "zero_in_1sigma_interval": bool(zero_in_interval),
            })
        with open(args.outdir / "nu_sweep_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
    else:
        if args.nu is None:
            raise ValueError("--nu must be specified if --sweep is not set")
        nu = 1e6 if np.isinf(args.nu) else args.nu
        (mu_hat, delta_hat), sigma_int_hat = fit_student_t_step(y, w, is_high_mass, nu, sigma_int_fixed=args.sigma_int)
        delta_hat, p_perm, sigma_int_hat2 = permutation_test_step(y, w, is_high_mass, nu, args.n_perm, sigma_int_fixed=args.sigma_int)
        sigma_used = float(sigma_int_hat if args.sigma_int is not None else sigma_int_hat2)
        grid = np.linspace(delta_hat - 0.05, delta_hat + 0.05, 80)
        prof = profile_likelihood(y, w, is_high_mass, nu, grid, sigma_int_fixed=args.sigma_int)
        rel = prof - np.nanmax(prof)
        ok = np.where(rel >= -0.5)[0]
        if len(ok) > 1:
            lo = float(grid[ok[0]])
            hi = float(grid[ok[-1]])
        else:
            lo = float("nan")
            hi = float("nan")
        df_out = pd.DataFrame([{
            "nu": float(args.nu),
            "delta_m": float(delta_hat),
            "delta_lo": float(lo),
            "delta_hi": float(hi),
            "p_perm": float(p_perm),
            "sigma_int": float(sigma_used),
        }])
        df_out.to_csv(args.outdir / "nu_sweep.csv", index=False)
        plt.figure(figsize=(6,4))
        plt.errorbar(
            [args.nu],
            [delta_hat],
            yerr=[[delta_hat - lo], [hi - delta_hat]],
            fmt="o"
        )
        plt.axhline(0.0, color="k", linestyle="--", alpha=0.6)
        plt.xscale("symlog")
        plt.xlabel("Student-t degrees of freedom ν")
        plt.ylabel("Host-mass step Δ_M")
        title_extra = f"sigma_int={'fixed' if args.sigma_int is not None else 'fitted'}"
        plt.title(title_extra)
        plt.tight_layout()
        plt.savefig(args.outdir / "deltaM_vs_nu.png")
        plt.close()
        summary = {
            "shoes_dat": str(args.shoes_dat),
            "y_mode": args.y_mode,
            "mu_col": args.mu_col,
            "mag_col": mag_col,
            "magerr_col": args.magerr_col,
            "mass_threshold": float(args.mass_threshold),
            "nu": float(args.nu),
            "sigma_int_fixed": (float(args.sigma_int) if args.sigma_int is not None else None),
            "sigma_int_used": float(sigma_used),
            "n_perm": int(args.n_perm),
            "seed": int(seed_value),
        }
        with open(args.outdir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
