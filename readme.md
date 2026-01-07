# Likelihood-Robust Isolation of the Host-Galaxy Mass Step in Type Ia Supernovae

This repository contains the reproducibility code for the paper:
Likelihood-Robust Isolation of the Host-Galaxy Mass Step in Type Ia Supernovae
Kevin Shepheard (Independent Researcher)

The analysis tests whether Type Ia supernova residual-like quantities are best described by:
a single heavy-tailed population with an environment-dependent zero-point shift in high-mass hosts,
rather than distinct sub-populations (“twins”), dust, or modified-gravity screening.

---

# What this code does

- Fits a Student-t likelihood model with a host-mass step parameter ΔM (in magnitudes)
- Models intrinsic scatter σ_int explicitly in the likelihood (in quadrature with measurement error)
- Runs permutation tests for ΔM
- Computes a profile-likelihood 1σ interval for ΔM
- Sweeps ν (degrees of freedom) to test robustness to likelihood choice
- Runs both “raw” and “postcorr” residual constructions to show what is or is not already corrected upstream
- Uses no cosmological parameters (H0 does not enter)

---

# Data requirements

You must provide Pantheon_SH0ES.dat (whitespace-delimited table). This repository does not redistribute the dataset.

---

# Installation

Use a Python environment with:
numpy, pandas, scipy, matplotlib

---

# Main script

FutureBreaker/src/whats_next.py

---

# Reproducibility runs

- Raw magnitude sweep (detect the mass step in uncorrected magnitudes)
python3 FutureBreaker/src/whats_next.py \
  --shoes-dat data/pantheonplus/distance_moduli/Pantheon_SH0ES.dat \
  --mass-threshold 10.0 \
  --y-mode raw \
  --sweep \
  --n-perm 2000 \
  --outdir FutureBreaker/out/nu_sweep_raw

- Post-corrected sweep (sanity check: residual step after upstream corrections)
python3 FutureBreaker/src/whats_next.py \
  --shoes-dat data/pantheonplus/distance_moduli/Pantheon_SH0ES.dat \
  --mass-threshold 10.0 \
  --y-mode postcorr \
  --sweep \
  --n-perm 2000 \
  --outdir FutureBreaker/out/nu_sweep_postcorr

- Gaussian limit (ν → ∞) in raw mode
python3 FutureBreaker/src/whats_next.py \
  --shoes-dat data/pantheonplus/distance_moduli/Pantheon_SH0ES.dat \
  --mass-threshold 10.0 \
  --y-mode raw \
  --nu inf \
  --n-perm 2000 \
  --outdir FutureBreaker/out/nu_inf_raw

---

# Outputs

Each run writes to --outdir:
- nu_sweep.csv
- nu_sweep_summary.json
- deltaM_vs_nu.png
- run_config.json (for sweep runs) or summary.json (for single-ν runs)

All ΔM values are in magnitudes.

---

# Notes on interpretation

- y-mode raw uses MU_SH0ES - mB and should recover a nonzero ΔM
- y-mode postcorr uses MU_SH0ES - m_b_corr and should yield ΔM consistent with zero if the mass step was already applied upstream
- ν-sweep tests robustness to heavy tails; the inference should not depend materially on ν

---

# Versioning

The paper-release script version is:
__version__ = 1.0.0

---

# License

MIT

---

# Citation

If you use this code, please cite the accompanying paper and software release:

Kevin Shepheard, *Likelihood-Robust Isolation of the Host-Galaxy Mass Step in Type Ia Supernovae*, Zenodo, https://doi.org/10.5281/zenodo.18169453