"""
minimax_esg.py — Multi-Objective Minimax ESG Portfolio (Xidonas replication)
=============================================================================
Replicates the Xidonas-style ESG minimax formulation:

    minimise   t                              (auxiliary variable)
    subject to:
      nw_e * (ideal_e − e(w)) / ideal_e ≤ t  [E deviation ≤ max]
      nw_s * (ideal_s − s(w)) / ideal_s ≤ t  [S deviation ≤ max]
      nw_g * (ideal_g − g(w)) / ideal_g ≤ t  [G deviation ≤ max]
      (μᵀw − rf) / σ(w) ≥ sharpe_min         [financial floor]
      raw_esg @ w ≤ esg_max                   [ESG hard ceiling]
      Σw = 1,  w ≥ 0,  w_i ≤ w_max
      t ≥ 0

ESG is the sole objective. Financial performance is enforced as a hard
floor constraint (sharpe_min), not blended into the objective.

Adapted to work with the 150-asset sp500_price_data.csv / ESG_top150.xlsx
dataset and produce output that is directly comparable to NSGA-II results.

Key changes from the original Term 1 code
------------------------------------------
  1. Data source: sp500_price_data.csv + ESG_top150.xlsx (no yfinance download).
  2. ESG data: maps environmentScore / socialScore / governanceScore + totalEsg
     directly from ESG_top150.xlsx — no hardcoded fallback dict needed.
  3. ESG direction: ERS/SRS/GRS are risk sub-scores (lower = better). We invert
     them to performance scores (ERP/SRP/GRP, higher = better), consistent with
     the original Term 1 code logic.
  4. W_MAX = 0.25  (matches NSGA-II W_MAX for fair comparison).
  5. RF_RATE = 0.05  (matches NSGA-II RF_RATE).
  6. yfinance removed — price data loaded from sp500_price_data.csv.
  7. Output DataFrame columns mirror NSGA-II's pareto_to_dataframe() output.
  8. Timing utility: benchmark_minimax() measures runtime across asset sizes.
  9. Sensitivity analysis: run across multiple E/S/G weight combinations to
     generate a solution set comparable in spirit to the NSGA-II Pareto front.

Bug-fixes applied to original code
------------------------------------
  * get_ticker_symbol() had reversed key/value entries for SBUX, NFLX, V, TT, LIN
    (used ticker as key instead of company name).
  * esg_minimax_optimization() returned `np.max(erp)` etc. as extra return values
    which were never used downstream — removed to clean up the interface.
  * calculate_returns() called prices.asfreq('D') on a file that already had daily
    data, causing NaN blowup — removed resample step; use daily returns directly.
  * beta constraint removed — was not present in NSGA-II or MV-ESG, making the
    comparison unfair by restricting Minimax's feasible set.
  * ESG_PERF_LB was set to 0.5 but the normalised scores rarely achieved this with
    150 diverse assets — now defaulted to 0.3 (30th percentile) and logged clearly.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
import warnings
import sys, os
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from nsga2 import RF_RATE, W_MAX   # keep constants consistent

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
PRICE_PATH  = "sp500_price_data.csv"
ESG_PATH    = "ESG_top150.xlsx"
BETA_LB    = 0.90   # portfolio beta lower bound (Xidonas Table 3: 0.900)
                     # keeps portfolio market-like; implicitly supports financial performance
BETA_UB    = 1.10   # portfolio beta upper bound (Xidonas Table 3: 1.100)
                     # prevents excessive systematic risk vs the market
DELTA_UB   = 0.20   # deviation upper bound: each E/S/G dimension's normalised shortfall
                     # from its ideal must not exceed this value (0.20 = 20%)
                     # prevents excessive sacrifice of any single ESG dimension

# ── UNIFIED CONSTRAINT (same across all three models) ──────────────────────
ESG_MAX = 20.0   # hard ceiling: portfolio totalEsg ≤ ESG_MAX (raw score, lower=better)
                  # enforced as: raw_esg @ w ≤ ESG_MAX  (identical to NSGA-II)
                  # W_MAX=0.25 and RF_RATE=0.05 imported from nsga2.py

# E/S/G weight combinations for sensitivity sweep
# Each entry: (w_E, w_S, w_G) — raw values, normalised internally to sum=1.
#
# Design rationale (from ESG_top150.xlsx data):
#   Natural data proportions: E=21.7%  S=45.2%  G=33.1%
#   S has highest correlation with totalEsg (0.778), E has highest variance (std=4.39)
#   Weights represent investor ESG preference — only ratios matter, not absolute values.
WEIGHT_COMBINATIONS = {
    # ── Baselines
    "Equal (33/33/33)"           : (10, 10, 10),   # equal weight on all three
    "Data-proportional (22/45/33)": (22, 45, 33),  # mirrors actual ESG score composition

    # ── Single-dimension focus (extreme investor preferences)
    "E-only"                     : (10,  1,  1),   # almost exclusively Environmental
    "S-only"                     : ( 1, 10,  1),   # almost exclusively Social
    "G-only"                     : ( 1,  1, 10),   # almost exclusively Governance

    # ── Two-dimension blends (one dimension down-weighted)
    "E+S (G down-weighted)"      : (10, 10,  1),
    "E+G (S down-weighted)"      : (10,  1, 10),
    "S+G (E down-weighted)"      : ( 1, 10, 10),

    # ── E ramp: E has highest variance → sweep its influence
    "E-light  (10/45/33)"        : (10, 45, 33),   # below data-proportional E
    "E-heavy  (40/30/30)"        : (40, 30, 30),   # above data-proportional E
    "E-dominant (60/20/20)"      : (60, 20, 20),   # E strongly dominant
    "E-max (80/10/10)"           : (80, 10, 10),   # extreme E focus

    # ── S ramp: S drives totalEsg most (corr=0.778) → stress-test its influence
    "S-light  (33/10/33)"        : (33, 10, 33),   # S below natural proportion
    "S-heavy  (20/60/20)"        : (20, 60, 20),   # S well above natural proportion
    "S-max (10/80/10)"           : (10, 80, 10),   # extreme S focus

    # ── G ramp: Governance often overlooked → explore its range
    "G-light  (40/40/10)"        : (40, 40, 10),   # G below natural proportion
    "G-heavy  (20/20/60)"        : (20, 20, 60),   # G well above natural proportion
    "G-max (10/10/80)"           : (10, 10, 80),   # extreme G focus

    # ── Balanced high-S (S naturally dominates; test with financial balance)
    "Balanced-S (25/50/25)"      : (25, 50, 25),   # S at natural level, E=G
}

# ─────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────

def load_minimax_data(price_path=PRICE_PATH, esg_path=ESG_PATH):
    """
    Load and prepare all data needed by the minimax optimiser.
    Returns:
      mu, Sigma         — annualised return + covariance (252-day)
      erp, srp, grp     — normalised E/S/G performance (0-1, higher=better)
      esg_risk_perf     — normalised totalEsg performance (0-1, higher=better)
      beta              — raw beta values
      raw_esg           — raw totalEsg scores (lower=better, for comparison with NSGA-II)
      tickers, esg_df
    """
    prices  = pd.read_csv(price_path, index_col=0, parse_dates=True)
    esg_df  = pd.read_excel(esg_path)
    tickers = esg_df["Symbol"].tolist()

    prices  = prices[tickers].copy()
    returns = prices.pct_change().dropna()    # daily returns, NO resample

    mu    = returns.mean().values * 252
    Sigma = returns.cov().values  * 252

    # E / S / G sub-scores from ESG_top150.xlsx
    # environmentScore / socialScore / governanceScore are risk sub-scores (lower=better)
    # → invert to performance (higher=better) via min-max normalisation
    def invert_normalise(arr):
        lo, hi = arr.min(), arr.max()
        if hi == lo:
            return np.ones_like(arr) * 0.5
        return (hi - arr) / (hi - lo)

    ers = esg_df.set_index("Symbol")["environmentScore"].loc[tickers].values
    srs = esg_df.set_index("Symbol")["socialScore"].loc[tickers].values
    grs = esg_df.set_index("Symbol")["governanceScore"].loc[tickers].values
    raw_esg = esg_df.set_index("Symbol")["totalEsg"].loc[tickers].values
    beta    = esg_df.set_index("Symbol")["beta"].loc[tickers].values

    erp          = invert_normalise(ers)
    srp          = invert_normalise(srs)
    grp          = invert_normalise(grs)
    esg_risk_perf = invert_normalise(raw_esg)   # overall ESG performance from totalEsg

    return mu, Sigma, erp, srp, grp, esg_risk_perf, beta, raw_esg, tickers, esg_df


# ─────────────────────────────────────────────
#  PORTFOLIO HELPERS
# ─────────────────────────────────────────────

def _project_weights(w):
    """Project w onto {w ≥ 0, Σw = 1, w_i ≤ W_MAX} — same as NSGA-II."""
    w = np.maximum(w, 0.0)
    for _ in range(len(w) + 1):
        s = w.sum()
        if s < 1e-12:
            return np.full(len(w), min(W_MAX, 1.0 / len(w)))
        w = w / s
        if (w - W_MAX).max() < 1e-9:
            break
        w = np.minimum(w, W_MAX)
    return w


def portfolio_metrics(w, mu, Sigma, rf=RF_RATE):
    ret  = float(w @ mu)
    vol  = float(np.sqrt(w @ Sigma @ w))
    sharpe = (ret - rf) / vol if vol > 1e-8 else 0.0
    return ret, vol, sharpe


# ─────────────────────────────────────────────
#  MINIMAX CORE OPTIMISER
# ─────────────────────────────────────────────

def minimax_esg_optimise(mu, erp, srp, grp, raw_esg, beta,
                          w_erp=10, w_srp=10, w_grp=10,
                          esg_max=ESG_MAX,
                          beta_lb=BETA_LB,
                          beta_ub=BETA_UB,
                          delta_ub=DELTA_UB,
                          w_max=W_MAX,
                          seed=42):
    """
    Xidonas & Essner (2022) minimax ESG — minimise the maximum weighted
    deviation from ideal E/S/G performance.

    Variables: x = [w_0, ..., w_{n-1}, t]  where t is the auxiliary minimax var.

    Formulation (equations from the paper):
        minimise   t                                        [Eq 20]
        subject to:
          nw_e * (ideal_e − e(w)) / ideal_e ≤ t            [Eq 21]
          nw_s * (ideal_s − s(w)) / ideal_s ≤ t            [Eq 22]
          nw_g * (ideal_g − g(w)) / ideal_g ≤ t            [Eq 23]
          (ideal_e − e(w)) / ideal_e ≤ delta_ub            [Eq 24]
          (ideal_s − s(w)) / ideal_s ≤ delta_ub            [Eq 25]
          (ideal_g − g(w)) / ideal_g ≤ delta_ub            [Eq 26]
          beta_lb ≤ Σw_i·β_i ≤ beta_ub                    [Eq 14]
          raw_esg @ w ≤ esg_max                            [ESG ceiling]
          Σw = 1,  w ≥ 0,  w_i ≤ w_max,  t ≥ 0            [Eq 9–10, 13]

    Financial performance is controlled via beta bounds (Xidonas Table 3:
    beta_lb=0.90, beta_ub=1.10), not a Sharpe floor. This keeps the portfolio
    market-like and implicitly supports risk-adjusted returns.

    SLSQP with multiple restarts. Returns (best_w, best_t).
    """
    np.random.seed(seed)
    n = len(mu)

    # Normalise E/S/G preference weights
    total = w_erp + w_srp + w_grp
    nw_e, nw_s, nw_g = w_erp / total, w_srp / total, w_grp / total

    # Ideal (utopian) points — best achievable per dimension
    ideal_e = max(float(np.max(erp)), 1e-8)
    ideal_s = max(float(np.max(srp)), 1e-8)
    ideal_g = max(float(np.max(grp)), 1e-8)

    def _t_init(w):
        """Compute starting t = max deviation for a given weight vector."""
        de = nw_e * (ideal_e - float(w @ erp)) / ideal_e
        ds = nw_s * (ideal_s - float(w @ srp)) / ideal_s
        dg = nw_g * (ideal_g - float(w @ grp)) / ideal_g
        return max(de, ds, dg, 0.0) + 0.01   # small buffer for feasibility

    # Objective: minimise t (last element of x)
    def objective(x):
        return x[-1]

    # Constraints for SLSQP (ineq: fun(x) >= 0,  eq: fun(x) == 0)
    cons = [
        {"type": "eq",   "fun": lambda x: x[:n].sum() - 1},
        # t >= Δe
        {"type": "ineq", "fun": lambda x: x[-1] - nw_e * (ideal_e - float(x[:n] @ erp)) / ideal_e},
        # t >= Δs
        {"type": "ineq", "fun": lambda x: x[-1] - nw_s * (ideal_s - float(x[:n] @ srp)) / ideal_s},
        # t >= Δg
        {"type": "ineq", "fun": lambda x: x[-1] - nw_g * (ideal_g - float(x[:n] @ grp)) / ideal_g},
        # Δe <= delta_ub  (per-dimension upper bound — investor-customisable)
        {"type": "ineq", "fun": lambda x, dub=delta_ub: dub - nw_e * (ideal_e - float(x[:n] @ erp)) / ideal_e},
        # Δs <= delta_ub
        {"type": "ineq", "fun": lambda x, dub=delta_ub: dub - nw_s * (ideal_s - float(x[:n] @ srp)) / ideal_s},
        # Δg <= delta_ub
        {"type": "ineq", "fun": lambda x, dub=delta_ub: dub - nw_g * (ideal_g - float(x[:n] @ grp)) / ideal_g},
        # beta_lb <= portfolio beta <= beta_ub  [Eq 14]
        {"type": "ineq", "fun": lambda x, lb=beta_lb: float(x[:n] @ beta) - lb},
        {"type": "ineq", "fun": lambda x, ub=beta_ub: ub - float(x[:n] @ beta)},
    ]
    if esg_max > 0:
        cons.append({
            "type": "ineq",
            "fun": lambda x, em=esg_max: em - float(x[:n] @ raw_esg)
        })

    # Bounds: weights in [0, w_max], auxiliary t in [0, inf)
    bounds = [(0.0, w_max)] * n + [(0.0, None)]

    def make_initial_points():
        wt_esg = nw_e * erp + nw_s * srp + nw_g * grp
        w_pts = [
            _project_weights(wt_esg),
            _project_weights(erp),
            _project_weights(srp),
            _project_weights(grp),
            np.ones(n) / n,
        ]
        for s in range(4):
            np.random.seed(seed + s + 1)
            raw = np.random.dirichlet(np.ones(n) * 0.5)
            blended = 0.7 * (wt_esg / wt_esg.sum()) + 0.3 * raw
            w_pts.append(_project_weights(blended))
        # Extend each w0 with its initial t value
        return [np.append(w, _t_init(w)) for w in w_pts]

    best_w, best_t = None, float("inf")
    for x0 in make_initial_points():
        try:
            res = minimize(objective, x0, method="SLSQP",
                           bounds=bounds, constraints=cons,
                           options={"maxiter": 2000, "ftol": 1e-9})
            if res.success and res.x[-1] < best_t:
                esg_ok = (esg_max == 0) or (float(res.x[:n] @ raw_esg) <= esg_max * 1.05)
                if esg_ok:
                    best_t = res.x[-1]
                    # Clip only — SLSQP bounds already enforce [0, w_max] and Σw=1.
                    # Avoid _project_weights here: its iterative capping moves weights
                    # away from the SLSQP optimum and degrades Sharpe.
                    w = np.clip(res.x[:n], 0.0, w_max)
                    best_w = w / w.sum() if w.sum() > 1e-10 else w
        except Exception:
            continue

    if best_w is None:
        # All SLSQP restarts failed — build a heuristic portfolio that still
        # satisfies the beta bounds. Score candidates by ESG, then greedily
        # widen the pool until the equal-weight beta is within [beta_lb, beta_ub].
        eligible = np.where(raw_esg <= esg_max)[0] if esg_max > 0 else np.arange(n)
        esg_score = nw_e * erp[eligible] + nw_s * srp[eligible] + nw_g * grp[eligible]
        ranked = eligible[np.argsort(esg_score)[::-1]]   # best ESG first

        top_idx = ranked[:10]   # default
        for pool in range(10, len(ranked) + 1, 10):
            pool = min(pool, len(ranked))
            idx  = ranked[:pool]
            w_try = np.zeros(n)
            w_try[idx] = 1.0 / pool
            port_beta = float(w_try @ beta)
            if (beta_lb <= port_beta <= beta_ub) or pool >= len(ranked):
                top_idx = idx
                break

        best_w = np.zeros(n)
        best_w[top_idx] = 1.0 / len(top_idx)
        best_t = _t_init(best_w)
        port_beta = float(best_w @ beta)
        print(f"[fallback] used {len(top_idx)}-asset equal-weight "
              f"(beta={port_beta:.3f})", end=" ")

    return best_w, best_t


# ─────────────────────────────────────────────
#  STAGE-2: MAX-SHARPE WITHIN MINIMAX SOLUTION
# ─────────────────────────────────────────────

def minimax_esg_max_sharpe(mu, Sigma, erp, srp, grp, raw_esg, beta,
                            w_erp=10, w_srp=10, w_grp=10,
                            esg_max=ESG_MAX,
                            beta_lb=BETA_LB,
                            beta_ub=BETA_UB,
                            delta_ub=DELTA_UB,
                            rf=RF_RATE,
                            w_max=W_MAX,
                            t_epsilon=0.02,
                            seed=42):
    """
    Two-stage lexicographic optimisation:

      Stage 1 — solve the standard Xidonas minimax problem → optimal t* (ESG quality).
      Stage 2 — fix  t ≤ t* + t_epsilon  and maximise Sharpe subject to all original
                constraints. Returns the highest-Sharpe portfolio that is still
                minimax-ESG-optimal (within tolerance t_epsilon).

    Parameters
    ----------
    t_epsilon : float
        Slack added to t* before solving stage 2 (default 0.02 = 2 pp).
        Keeps ESG quality essentially unchanged while giving the Sharpe solver
        enough room to shift weights toward higher-return assets.

    Returns
    -------
    w_s1, t_s1  : stage-1 minimax-optimal weights and t value
    w_s2, t_s2  : stage-2 max-Sharpe weights (ESG-locked) and its t value
    sharpe_s2   : achieved Sharpe in stage 2
    """
    # ── Stage 1: minimax ESG ──────────────────────────────────────────────────
    w_s1, t_s1 = minimax_esg_optimise(
        mu, erp, srp, grp, raw_esg, beta,
        w_erp=w_erp, w_srp=w_srp, w_grp=w_grp,
        esg_max=esg_max, beta_lb=beta_lb, beta_ub=beta_ub,
        delta_ub=delta_ub, w_max=w_max, seed=seed
    )

    n = len(mu)
    total = w_erp + w_srp + w_grp
    nw_e, nw_s, nw_g = w_erp / total, w_srp / total, w_grp / total
    ideal_e = max(float(np.max(erp)), 1e-8)
    ideal_s = max(float(np.max(srp)), 1e-8)
    ideal_g = max(float(np.max(grp)), 1e-8)
    t_lock  = t_s1 + t_epsilon   # ESG quality ceiling for stage 2

    # ── Stage 2: maximise Sharpe with ESG quality locked ─────────────────────
    # Objective: minimise negative Sharpe (SLSQP minimises)
    def neg_sharpe(w):
        ret = float(w @ mu)
        vol = float(np.sqrt(w @ Sigma @ w))
        return -(ret - rf) / vol if vol > 1e-8 else 0.0

    cons_s2 = [
        {"type": "eq",   "fun": lambda w: w.sum() - 1},
        # ESG minimax lock: each deviation ≤ t_lock
        {"type": "ineq", "fun": lambda w, tl=t_lock: tl - nw_e * (ideal_e - float(w @ erp)) / ideal_e},
        {"type": "ineq", "fun": lambda w, tl=t_lock: tl - nw_s * (ideal_s - float(w @ srp)) / ideal_s},
        {"type": "ineq", "fun": lambda w, tl=t_lock: tl - nw_g * (ideal_g - float(w @ grp)) / ideal_g},
    ]
    if esg_max > 0:
        cons_s2.append({"type": "ineq", "fun": lambda w, em=esg_max: em - float(w @ raw_esg)})
    # beta bounds carried through to stage 2
    cons_s2.append({"type": "ineq", "fun": lambda w, lb=beta_lb: float(w @ beta) - lb})
    cons_s2.append({"type": "ineq", "fun": lambda w, ub=beta_ub: ub - float(w @ beta)})

    bounds_s2 = [(0.0, w_max)] * n

    # ── Finance-driven warm start: unconstrained max-Sharpe (no ESG lock) ──────
    # Gives SLSQP a high-Sharpe starting point so it searches the right region.
    def _max_sharpe_warmstart():
        cons_fin = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        if esg_max > 0:
            cons_fin.append({"type": "ineq", "fun": lambda w, em=esg_max: em - float(w @ raw_esg)})
        res_fin = minimize(neg_sharpe, np.ones(n) / n, method="SLSQP",
                           bounds=[(0.0, w_max)] * n, constraints=cons_fin,
                           options={"maxiter": 1000, "ftol": 1e-8})
        if res_fin.success:
            w = np.clip(res_fin.x, 0.0, w_max)
            return w / w.sum()
        return None

    w_fin = _max_sharpe_warmstart()

    # Multiple restarts: finance warm start, s1 solution, equal-weight, blends
    best_w_s2, best_sh = w_s1.copy(), portfolio_metrics(w_s1, mu, Sigma, rf)[2]
    start_pts = [w_s1.copy(), np.ones(n) / n]
    if w_fin is not None:
        start_pts.insert(0, w_fin)                      # finance-driven start first
        start_pts.append(0.5 * w_s1 + 0.5 * w_fin)     # blend: ESG meets finance
    for s in range(5):
        np.random.seed(seed + s + 100)
        # Perturb w_s1 slightly — stays near the ESG-feasible region
        noise = np.random.dirichlet(np.ones(n) * 2.0)   # concentrated, not flat
        start_pts.append(0.8 * w_s1 + 0.2 * noise)

    for w0 in start_pts:
        w0 = np.clip(w0, 0.0, w_max)
        w0 = w0 / w0.sum() if w0.sum() > 1e-10 else w0
        try:
            res = minimize(neg_sharpe, w0, method="SLSQP",
                           bounds=bounds_s2, constraints=cons_s2,
                           options={"maxiter": 2000, "ftol": 1e-9})
            if res.success:
                w_cand = np.clip(res.x, 0.0, w_max)
                w_cand = w_cand / w_cand.sum()
                sh_cand = portfolio_metrics(w_cand, mu, Sigma, rf)[2]
                if sh_cand > best_sh:
                    best_sh   = sh_cand
                    best_w_s2 = w_cand
        except Exception:
            continue

    # Recompute t for the stage-2 portfolio
    de = nw_e * (ideal_e - float(best_w_s2 @ erp)) / ideal_e
    ds = nw_s * (ideal_s - float(best_w_s2 @ srp)) / ideal_s
    dg = nw_g * (ideal_g - float(best_w_s2 @ grp)) / ideal_g
    t_s2 = max(de, ds, dg)

    return w_s1, t_s1, best_w_s2, t_s2, best_sh


# ─────────────────────────────────────────────
#  SENSITIVITY SWEEP  (generates solution set)
# ─────────────────────────────────────────────

def run_minimax_sweep(mu, Sigma, erp, srp, grp, raw_esg, beta,
                       tickers,
                       weight_combos=None,
                       esg_max=ESG_MAX,
                       delta_ub=DELTA_UB,
                       rf=RF_RATE):
    """
    Run minimax optimisation for each E/S/G weight combination.
    Unified constraint: raw totalEsg ≤ esg_max (same as NSGA-II).
    delta_ub: per-dimension deviation upper bound (investor-customisable, default 20%).
    beta: per-asset beta values for the portfolio beta bounds constraint.
    Returns a DataFrame with the same column structure as NSGA-II output.
    """
    if weight_combos is None:
        weight_combos = WEIGHT_COMBINATIONS

    # Compute normalised ESG performance for reporting only (not used in constraint)
    esg_min, esg_max_r = raw_esg.min(), raw_esg.max()
    esg_risk_perf = (esg_max_r - raw_esg) / (esg_max_r - esg_min)

    rows = []
    for i, (combo_name, (we, ws, wg)) in enumerate(weight_combos.items()):
        print(f"      [{i+1}/{len(weight_combos)}] {combo_name}...", end=" ", flush=True)
        w_opt, t_opt = minimax_esg_optimise(
            mu, erp, srp, grp, raw_esg, beta,
            w_erp=we, w_srp=ws, w_grp=wg,
            esg_max=esg_max, delta_ub=delta_ub, seed=42 + i
        )
        ret, vol, sharpe = portfolio_metrics(w_opt, mu, Sigma, rf)
        esg_raw  = float(raw_esg      @ w_opt)   # totalEsg (lower=better)
        esg_perf = float(esg_risk_perf @ w_opt)  # normalised performance (for info)

        row = {
            "Portfolio_ID"    : i + 1,
            "Combo_Name"      : combo_name,
            "w_E": we, "w_S": ws, "w_G": wg,
            "Annual_Return"   : round(ret,      6),
            "Annual_Variance" : round(vol**2,   6),
            "Annual_Volatility": round(vol,     6),
            "Sharpe_Ratio"    : round(sharpe,   4),
            "ESG_Score"       : round(esg_raw,  4),   # raw totalEsg <= esg_max
            "ESG_Performance" : round(esg_perf, 4),   # normalised (info only)
            "Minimax_T"       : round(t_opt,    6),   # achieved minimax deviation (lower=better)
            "Pareto_Rank"     : 0,
        }
        for ticker, wt in zip(tickers, w_opt):
            row[f"w_{ticker}"] = round(float(wt), 6)
        rows.append(row)
        esg_status = "OK" if esg_raw <= esg_max else "VIOLATED"
        print(f"ret={ret:.2%}  vol={vol:.2%}  sharpe={sharpe:.3f}  "
              f"ESG={esg_raw:.2f}  t={t_opt:.4f}  [{esg_status}]")

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
#  RUNTIME BENCHMARKING
# ─────────────────────────────────────────────

def benchmark_minimax(mu_full, erp_full, srp_full, grp_full,
                       raw_esg_full, beta_full,
                       sample_sizes=(50, 100, 150),
                       n_timing_runs=3,
                       esg_max=ESG_MAX,
                       seed=42):
    """
    Measure mean ± std wall-clock time for minimax_esg_optimise() across
    asset-subset sizes. Unified constraint: totalEsg ≤ esg_max.
    Returns dict: {size: {"mean": float, "std": float}}
    """
    np.random.seed(seed)
    n_full = len(mu_full)
    timing = {}

    for sz in sample_sizes:
        times = []
        for run in range(n_timing_runs):
            idx = np.random.choice(n_full, sz, replace=False)
            t0  = time.perf_counter()
            minimax_esg_optimise(
                mu_full[idx], erp_full[idx], srp_full[idx], grp_full[idx],
                raw_esg_full[idx], beta_full[idx],
                esg_max=esg_max, seed=seed + run
            )
            times.append(time.perf_counter() - t0)
        timing[sz] = {"mean": float(np.mean(times)), "std": float(np.std(times))}
        print(f"  Minimax {sz:>3d} assets | "
              f"mean={timing[sz]['mean']:.2f}s  std={timing[sz]['std']:.2f}s")

    return timing


# ─────────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────────

def _norm_col(series):
    lo, hi = series.min(), series.max()
    return (series - lo) / (hi - lo) if hi > lo else series * 0 + 0.5

# colour per combination — consistent across all three figures
_COMBO_COLORS = {
    "Equal (33/33/33)"            : "#555555",
    "Data-proportional (22/45/33)": "#888888",
    "E-only"                      : "#27ae60",
    "S-only"                      : "#2980b9",
    "G-only"                      : "#8e44ad",
    "E+S (G down-weighted)"       : "#1abc9c",
    "E+G (S down-weighted)"       : "#16a085",
    "S+G (E down-weighted)"       : "#6c5ce7",
    "E-light  (10/45/33)"         : "#a8e063",
    "E-heavy  (40/30/30)"         : "#2ecc71",
    "E-dominant (60/20/20)"       : "#27ae60",
    "E-max (80/10/10)"            : "#1e8449",
    "S-light  (33/10/33)"         : "#74b9ff",
    "S-heavy  (20/60/20)"         : "#2980b9",
    "S-max (10/80/10)"            : "#1a5276",
    "G-light  (40/40/10)"         : "#c39bd3",
    "G-heavy  (20/20/60)"         : "#8e44ad",
    "G-max (10/10/80)"            : "#6c3483",
    "Balanced-S (25/50/25)"       : "#e67e22",
}


def _compute_esg_subscores(df, tickers, erp, srp, grp):
    """Add E_Score / S_Score / G_Score columns to df (in-place copy)."""
    df = df.copy()
    weight_cols = [f"w_{t}" for t in tickers]
    df["E_Score"] = df[weight_cols].values.astype(float) @ erp
    df["S_Score"] = df[weight_cols].values.astype(float) @ srp
    df["G_Score"] = df[weight_cols].values.astype(float) @ grp
    df["Sharpe_norm"] = _norm_col(df["Sharpe_Ratio"])
    df["Return_norm"] = _norm_col(df["Annual_Return"])
    return df


def plot_minimax_visuals(df, tickers, erp, srp, grp,
                          esg_max=ESG_MAX,
                          radar_path="minimax_radar.png",
                          tradeoff_path="minimax_tradeoff.png",
                          heatmap_path="minimax_heatmap.png"):
    """
    Generate three report-ready figures for the Minimax ESG sensitivity sweep:

      1. minimax_radar.png    — spider/radar chart: E/S/G + Sharpe + Return
                                per combination, grouped into 4 panels.
      2. minimax_tradeoff.png — Sharpe vs ESG risk scatter (bubble=return)
                                + stacked E/S/G sub-score bar chart.
      3. minimax_heatmap.png  — full metrics heatmap across all combinations
                                (green=better, red=worse, independently normalised).
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors

    LIGHT_BG = "white"
    df = _compute_esg_subscores(df, tickers, erp, srp, grp)

    def style_ax(ax, title, xlabel, ylabel, fs=11):
        ax.set_facecolor("white")
        ax.set_title(title, fontsize=fs, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=fs - 1)
        ax.set_ylabel(ylabel, fontsize=fs - 1)
        ax.tick_params(labelsize=9)
        ax.grid(True, color="#e0e0e0", linewidth=0.6)
        for sp in ax.spines.values():
            sp.set_edgecolor("#aaaaaa")

    # ── Figure 1: Radar (spider) chart ───────────────────────────────────────
    categories = ["E Score", "S Score", "G Score", "Sharpe\n(norm)", "Return\n(norm)"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    GROUPS = {
        "Baselines"       : ["Equal (33/33/33)", "Data-proportional (22/45/33)",
                             "Balanced-S (25/50/25)"],
        "Single-dimension": ["E-only", "S-only", "G-only"],
        "Two-dim blends"  : ["E+S (G down-weighted)", "E+G (S down-weighted)",
                             "S+G (E down-weighted)"],
        "E / S / G ramps" : ["E-light  (10/45/33)", "E-heavy  (40/30/30)",
                             "E-dominant (60/20/20)", "E-max (80/10/10)",
                             "S-light  (33/10/33)", "S-heavy  (20/60/20)",
                             "S-max (10/80/10)", "G-light  (40/40/10)",
                             "G-heavy  (20/20/60)", "G-max (10/10/80)"],
    }

    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 14),
                                subplot_kw=dict(polar=True))
    fig1.patch.set_facecolor(LIGHT_BG)
    fig1.suptitle(
        "Minimax ESG: E/S/G Sub-Score Profile by Investor Preference\n"
        "(Radar axes: E performance, S performance, G performance, "
        "Sharpe, Return — all normalised 0–1)",
        fontsize=13, fontweight="bold", y=1.01)

    for ax, (group_name, combos) in zip(axes1.flat, GROUPS.items()):
        ax.set_facecolor("#f9f9f9")
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"],
                           fontsize=7, color="#888")
        ax.grid(color="#cccccc", linewidth=0.5)
        ax.spines["polar"].set_visible(False)
        ax.set_title(group_name, fontsize=11, fontweight="bold", pad=14)

        for combo in combos:
            row = df[df["Combo_Name"].str.strip() == combo.strip()]
            if row.empty:
                continue
            r = row.iloc[0]
            vals = [r["E_Score"], r["S_Score"], r["G_Score"],
                    r["Sharpe_norm"], r["Return_norm"]]
            vals += vals[:1]
            color = _COMBO_COLORS.get(combo, "#333333")
            ax.plot(angles, vals, color=color, linewidth=2)
            ax.fill(angles, vals, color=color, alpha=0.12)
            max_idx = int(np.argmax(vals[:-1]))
            ax.annotate(
                combo.split("(")[0].strip(),
                xy=(angles[max_idx], vals[max_idx]),
                xytext=(angles[max_idx], min(vals[max_idx] + 0.12, 1.0)),
                fontsize=7, color=color, ha="center",
                arrowprops=dict(arrowstyle="-", color=color, lw=0.8))

    plt.tight_layout()
    plt.savefig(radar_path, dpi=150, bbox_inches="tight", facecolor=LIGHT_BG)
    plt.close()
    print(f"      {radar_path} saved")

    # ── Figure 2: Trade-off scatter + stacked E/S/G bar ─────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
    fig2.patch.set_facecolor(LIGHT_BG)
    fig2.suptitle(
        "Minimax ESG: Trade-off Landscape across All 19 E/S/G Preference Combinations",
        fontsize=13, fontweight="bold", y=1.01)

    ax = axes2[0]
    ret_min = df["Annual_Return"].min()
    ret_rng = df["Annual_Return"].max() - ret_min
    for _, row in df.iterrows():
        color = _COMBO_COLORS.get(row["Combo_Name"], "#333333")
        size  = 100 + (row["Annual_Return"] - ret_min) / max(ret_rng, 1e-8) * 400
        ax.scatter(row["ESG_Score"], row["Sharpe_Ratio"],
                   color=color, s=size, alpha=0.80,
                   edgecolors="white", linewidth=0.8, zorder=4)
        ax.annotate(row["Combo_Name"].split("(")[0].strip(),
                    xy=(row["ESG_Score"], row["Sharpe_Ratio"]),
                    xytext=(3, 3), textcoords="offset points",
                    fontsize=6.5, color=color)
    ax.axvline(esg_max, color="#c0392b", linewidth=1.5, linestyle="--",
               label=f"ESG ceiling = {esg_max}")
    ax.legend(fontsize=9)
    style_ax(ax,
             "Sharpe Ratio vs ESG Risk Score\n(bubble size = Annual Return)",
             "ESG Risk Score (totalEsg, lower=better)", "Sharpe Ratio")

    ax = axes2[1]
    mm_sorted = df.sort_values("E_Score", ascending=True).reset_index(drop=True)
    y = np.arange(len(mm_sorted))
    h = 0.6
    ax.barh(y, mm_sorted["E_Score"], h,
            color="#27ae60", alpha=0.85, label="E Score")
    ax.barh(y, mm_sorted["S_Score"], h, left=mm_sorted["E_Score"],
            color="#2980b9", alpha=0.85, label="S Score")
    ax.barh(y, mm_sorted["G_Score"], h,
            left=mm_sorted["E_Score"] + mm_sorted["S_Score"],
            color="#8e44ad", alpha=0.85, label="G Score")
    ax.set_yticks(y)
    ax.set_yticklabels(mm_sorted["Combo_Name"], fontsize=7.5)
    ax.legend(fontsize=9, loc="lower right")
    style_ax(ax,
             "E / S / G Performance Sub-Scores per Combination\n"
             "(sorted by E score ascending)",
             "Normalised Score (0=worst, 1=best)", "")

    plt.tight_layout()
    plt.savefig(tradeoff_path, dpi=150, bbox_inches="tight", facecolor=LIGHT_BG)
    plt.close()
    print(f"      {tradeoff_path} saved")

    # ── Figure 3: Heatmap ────────────────────────────────────────────────────
    cols_show  = ["E_Score", "S_Score", "G_Score", "Sharpe_Ratio",
                  "Annual_Return", "Annual_Volatility", "ESG_Score"]
    col_labels = ["E Score\n(norm, ↑)", "S Score\n(norm, ↑)", "G Score\n(norm, ↑)",
                  "Sharpe\nRatio", "Annual\nReturn", "Annual\nVolatility",
                  "ESG Risk\nScore (↓)"]
    higher_better = [True, True, True, True, True, False, False]

    heat_data = df[cols_show].values.astype(float)
    heat_norm = np.zeros_like(heat_data)
    for j, hb in enumerate(higher_better):
        lo, hi = heat_data[:, j].min(), heat_data[:, j].max()
        if hi > lo:
            heat_norm[:, j] = (heat_data[:, j] - lo) / (hi - lo)
            if not hb:
                heat_norm[:, j] = 1 - heat_norm[:, j]
        else:
            heat_norm[:, j] = 0.5

    fig3, ax3 = plt.subplots(figsize=(14, 9))
    fig3.patch.set_facecolor(LIGHT_BG)
    im = ax3.imshow(heat_norm, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax3.set_xticks(range(len(col_labels)))
    ax3.set_xticklabels(col_labels, fontsize=9, fontweight="bold")
    ax3.set_yticks(range(len(df)))
    ax3.set_yticklabels(df["Combo_Name"], fontsize=8.5)
    ax3.set_title(
        "Minimax ESG: Performance Heatmap across All 19 Preference Combinations\n"
        "(Green = better, Red = worse — each column independently normalised)",
        fontsize=12, fontweight="bold", pad=12)

    for i in range(len(df)):
        for j, col in enumerate(cols_show):
            val = heat_data[i, j]
            fmt = (".3f" if col in ["E_Score", "S_Score", "G_Score"]
                   else ".2f"  if col == "Sharpe_Ratio"
                   else ".1%"  if col in ["Annual_Return", "Annual_Volatility"]
                   else ".1f")
            txt_color = ("white"
                         if heat_norm[i, j] < 0.35 or heat_norm[i, j] > 0.75
                         else "black")
            ax3.text(j, i, f"{val:{fmt}}",
                     ha="center", va="center", fontsize=7.5, color=txt_color)

    plt.colorbar(im, ax=ax3, fraction=0.02, pad=0.02).set_label(
        "Normalised score (1=best, 0=worst)", fontsize=9)
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight", facecolor=LIGHT_BG)
    plt.close()
    print(f"      {heatmap_path} saved")


# ─────────────────────────────────────────────
#  STAGE 1 vs STAGE 2 COMPARISON PLOT
# ─────────────────────────────────────────────

def plot_stage_comparison(s1_df, s2_rows,
                           out_path="minimax_stage_comparison.png"):
    """
    Four-panel figure comparing Stage 1 (minimax ESG) vs Stage 2 (max-Sharpe
    within ESG lock) across all 19 weight combinations.

      Panel 1 — Risk-Return scatter with S1→S2 arrows
      Panel 2 — Sharpe Ratio: S1 vs S2 grouped bar
      Panel 3 — ESG Score: S1 vs S2 grouped bar  (lower = better)
      Panel 4 — Sharpe improvement (S2 − S1) bar
    """
    import matplotlib.pyplot as plt

    LIGHT_BG = "#f8f9fa"
    C_S1     = "#2980b9"   # blue  — Stage 1
    C_S2     = "#e74c3c"   # red   — Stage 2
    ALPHA    = 0.82

    combos   = list(s1_df["Combo_Name"])
    labels   = [c.split(" (")[0] for c in combos]   # short label

    s1_ret   = s1_df["Annual_Return"].values * 100
    s1_vol   = s1_df["Annual_Volatility"].values * 100
    s1_sh    = s1_df["Sharpe_Ratio"].values
    s1_esg   = s1_df["ESG_Score"].values

    s2_lookup = {r["Combo_Name"]: r for r in s2_rows}
    s2_ret  = np.array([s2_lookup[c]["Annual_Return"]    * 100 for c in combos])
    s2_vol  = np.array([s2_lookup[c]["Annual_Volatility"] * 100 for c in combos])
    s2_sh   = np.array([s2_lookup[c]["Sharpe_Ratio"]          for c in combos])
    s2_esg  = np.array([s2_lookup[c]["ESG_Score"]             for c in combos])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor(LIGHT_BG)
    fig.suptitle(
        "Minimax ESG — Stage 1 (min t*) vs Stage 2 (max Sharpe within ESG lock)\n"
        f"19 E/S/G weight combinations  |  ESG_MAX={ESG_MAX}  W_MAX={W_MAX:.0%}  "
        f"\u03b2 \u2208 [{BETA_LB},{BETA_UB}]  \u0394UB={DELTA_UB:.0%}",
        fontsize=12, fontweight="bold", y=1.01
    )

    def style(ax, title, xlabel, ylabel):
        ax.set_facecolor("white")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.tick_params(labelsize=9)
        ax.grid(True, color="#e0e0e0", linewidth=0.6)
        for sp in ax.spines.values():
            sp.set_edgecolor("#aaaaaa")

    # ── Panel 1: Risk-Return scatter with arrows ──────────────────────────────
    ax = axes[0, 0]
    ax.scatter(s1_vol, s1_ret, color=C_S1, s=80, zorder=3, label="Stage 1", alpha=ALPHA)
    ax.scatter(s2_vol, s2_ret, color=C_S2, s=80, marker="D", zorder=3,
               label="Stage 2", alpha=ALPHA)
    for x1, y1, x2, y2 in zip(s1_vol, s1_ret, s2_vol, s2_ret):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#888888",
                                   lw=0.9, mutation_scale=10))
    style(ax, "Risk-Return: Stage 1 \u2192 Stage 2",
          "Annual Volatility (%)", "Annual Return (%)")
    ax.legend(fontsize=9)

    # ── Panel 2: Sharpe grouped bar ───────────────────────────────────────────
    ax = axes[0, 1]
    x  = np.arange(len(combos))
    bw = 0.38
    ax.bar(x - bw/2, s1_sh, bw, color=C_S1, alpha=ALPHA, edgecolor="white", label="Stage 1")
    ax.bar(x + bw/2, s2_sh, bw, color=C_S2, alpha=ALPHA, edgecolor="white", label="Stage 2")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.legend(fontsize=9)
    style(ax, "Sharpe Ratio per Combination", "", "Sharpe Ratio")

    # ── Panel 3: ESG Score grouped bar ────────────────────────────────────────
    ax = axes[1, 0]
    ax.bar(x - bw/2, s1_esg, bw, color=C_S1, alpha=ALPHA, edgecolor="white", label="Stage 1")
    ax.bar(x + bw/2, s2_esg, bw, color=C_S2, alpha=ALPHA, edgecolor="white", label="Stage 2")
    ax.axhline(ESG_MAX, color="#c0392b", linewidth=1.5, linestyle="--",
               label=f"ESG ceiling = {ESG_MAX}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.legend(fontsize=9)
    style(ax, "ESG Risk Score (lower = better)", "", "totalEsg Score")

    # ── Panel 4: Sharpe improvement bar ──────────────────────────────────────
    ax = axes[1, 1]
    delta_sh = s2_sh - s1_sh
    bar_colors = [C_S2 if d >= 0 else "#7f8c8d" for d in delta_sh]
    bars = ax.bar(x, delta_sh, color=bar_colors, alpha=ALPHA, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    for bar, val in zip(bars, delta_sh):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + (0.005 if val >= 0 else -0.015),
                f"{val:+.3f}", ha="center",
                va="bottom" if val >= 0 else "top", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    # Zoom y-axis to the actual range so subtle differences between combos are visible
    d_min, d_max = delta_sh.min(), delta_sh.max()
    pad = max((d_max - d_min) * 0.3, 0.02)
    ax.set_ylim(max(0, d_min - pad), d_max + pad)
    style(ax, "Sharpe Improvement (Stage 2 \u2212 Stage 1)\n(y-axis zoomed to show variation)",
          "", "\u0394Sharpe")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=LIGHT_BG)
    plt.close()
    print(f"      {out_path} saved")


# ─────────────────────────────────────────────
#  PORTFOLIO VISUALISATION  (dark theme)
# ─────────────────────────────────────────────

def plot_minimax_portfolio(df_s2, tickers, esg_df, out_path="minimax_portfolio_plots.png"):
    """
    3-panel portfolio analysis figure for the equal-weight (33/33/33) Stage 2 portfolio:
      (1) Top 10 holdings horizontal bar
      (2) Sector breakdown pie
      (3) Key metrics summary
    """
    DARK_BG  = "#0f1117"
    PANEL_BG = "#1a1d27"
    ACCENT   = "#00d4ff"
    GRID_C   = "#2a2d3a"

    weight_cols = [f"w_{t}" for t in tickers]
    sector_map  = esg_df.set_index("Symbol")["GICS Sector"].to_dict()

    # Select the equal-weight portfolio; fall back to best Sharpe if not found
    mask = df_s2["Combo_Name"].str.strip() == "Equal (33/33/33)"
    row  = df_s2.loc[mask].iloc[0] if mask.any() else df_s2.loc[df_s2["Sharpe_Ratio"].idxmax()]
    w    = row[weight_cols].values.astype(float)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.patch.set_facecolor(DARK_BG)

    # ── (1) Top 10 holdings
    ax1  = axes[0]
    idx  = np.argsort(w)[::-1][:10]
    top_t = [tickers[i] for i in idx]
    top_w = w[idx] * 100
    bars = ax1.barh(top_t[::-1], top_w[::-1], color=ACCENT, alpha=0.85)
    for bar, val in zip(bars, top_w[::-1]):
        ax1.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}%", va="center", fontsize=9, color="white")
    ax1.set_facecolor(PANEL_BG)
    ax1.tick_params(colors="white", labelsize=9)
    ax1.xaxis.label.set_color("white")
    ax1.title.set_color("white")
    ax1.set_title("Top 10 Holdings\n(Equal E/S/G Weights, Stage 2)",
                  fontsize=11, fontweight="bold", pad=8)
    ax1.set_xlabel("Weight (%)", fontsize=9)
    ax1.grid(axis="x", color=GRID_C, linewidth=0.5, alpha=0.7)
    for spine in ax1.spines.values():
        spine.set_edgecolor("#333649")

    # ── (2) Sector breakdown pie
    ax2 = axes[1]
    sector_weights: dict = {}
    for t, wt in zip(tickers, w):
        sec = sector_map.get(t, "Unknown")
        sector_weights[sec] = sector_weights.get(sec, 0.0) + wt
    sorted_secs = sorted(sector_weights.items(), key=lambda x: -x[1])
    s_labels    = [s for s, _ in sorted_secs]
    s_vals      = [v for _, v in sorted_secs]
    colors_pie  = plt.cm.Set3(np.linspace(0, 1, len(s_labels)))
    wedges, _, autotexts = ax2.pie(
        s_vals, labels=None,
        autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
        colors=colors_pie, startangle=90, pctdistance=0.75,
        wedgeprops={"edgecolor": DARK_BG, "linewidth": 0.8},
    )
    for at in autotexts:
        at.set_color("white"); at.set_fontsize(9)
    ax2.legend(wedges, s_labels, loc="center left", bbox_to_anchor=(-0.5, 0.5),
               fontsize=8, facecolor=PANEL_BG, labelcolor="white", framealpha=0.8)
    ax2.set_facecolor(DARK_BG)
    ax2.set_title("Sector Breakdown\n(Equal E/S/G Weights, Stage 2)",
                  fontsize=11, fontweight="bold", color="white", pad=8)

    # ── (3) Key metrics summary
    ax3 = axes[2]
    ax3.set_facecolor(PANEL_BG)
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
    ax3.axis("off")
    metrics = [
        ("Annual Return",     f"{row['Annual_Return']:.2%}"),
        ("Annual Volatility", f"{row['Annual_Volatility']:.2%}"),
        ("Sharpe Ratio",      f"{row['Sharpe_Ratio']:.4f}"),
        ("ESG Risk Score",    f"{row['ESG_Score']:.2f}"),
        ("Minimax t*",        f"{row['Minimax_T']:.4f}"),
        ("E/S/G Weights",     "10 / 10 / 10"),
    ]
    y_start = 0.82
    ax3.text(0.5, 0.95, "Portfolio Metrics", ha="center", va="top",
             fontsize=12, fontweight="bold", color="white",
             transform=ax3.transAxes)
    for label, val in metrics:
        ax3.text(0.15, y_start, label, ha="left", va="center",
                 fontsize=10, color="#aaaaaa", transform=ax3.transAxes)
        ax3.text(0.85, y_start, val, ha="right", va="center",
                 fontsize=10, fontweight="bold", color=ACCENT, transform=ax3.transAxes)
        ax3.plot([0.05, 0.95], [y_start - 0.04, y_start - 0.04],
                 color=GRID_C, linewidth=0.5, transform=ax3.transAxes)
        y_start -= 0.13
    for spine in ax3.spines.values():
        spine.set_edgecolor("#333649")

    fig.suptitle("Minimax ESG — Equal-Weight Stage 2 Portfolio Analysis",
                 fontsize=13, fontweight="bold", color="white", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Minimax portfolio plots saved → {out_path}")


# ─────────────────────────────────────────────
#  MAIN — standalone run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  Minimax ESG Portfolio Optimisation (Term 1, Xidonas-style)")
    print("=" * 65)

    print("\n[1/3] Loading data...")
    mu, Sigma, erp, srp, grp, esg_risk_perf, beta, raw_esg, tickers, esg_df = \
        load_minimax_data()
    print(f"      {len(tickers)} assets  |  RF={RF_RATE:.2%}  W_MAX={W_MAX:.2%}  ESG_MAX={ESG_MAX}")
    print(f"      ERP range: {erp.min():.3f}–{erp.max():.3f}  "
          f"SRP: {srp.min():.3f}–{srp.max():.3f}  "
          f"GRP: {grp.min():.3f}–{grp.max():.3f}")
    print(f"      Raw ESG range: {raw_esg.min():.1f}–{raw_esg.max():.1f}  "
          f"(hard ceiling = {ESG_MAX})")

    print("\n[2/3] Running two-stage optimisation (equal E/S/G weights 10/10/10)...")
    t0 = time.perf_counter()
    _w_s1, _t_s1, w_s2, _t_s2, sh_s2 = minimax_esg_max_sharpe(
        mu, Sigma, erp, srp, grp, raw_esg, beta,
        w_erp=10, w_srp=10, w_grp=10, seed=42
    )
    elapsed = time.perf_counter() - t0
    ret_s1, vol_s1, sh_s1 = portfolio_metrics(_w_s1, mu, Sigma)
    ret_s2, vol_s2, _     = portfolio_metrics(w_s2,  mu, Sigma)
    esg_s2 = float(raw_esg @ w_s2)
    print(f"      Done in {elapsed:.1f}s")
    print(f"      Stage 1: Sharpe={sh_s1:.4f}")
    print(f"      Stage 2: Ret={ret_s2:.2%}  Vol={vol_s2:.2%}  "
          f"Sharpe={sh_s2:.4f}  ESG={esg_s2:.2f}  t*={_t_s2:.4f}")

    row_s2 = {
        "Portfolio_ID": 1, "Combo_Name": "Equal (33/33/33)",
        "Annual_Return": ret_s2, "Annual_Volatility": vol_s2,
        "Sharpe_Ratio": sh_s2, "ESG_Score": esg_s2, "Minimax_T": _t_s2,
    }
    for ticker, wt in zip(tickers, w_s2):
        row_s2[f"w_{ticker}"] = round(float(wt), 6)
    df_s2 = pd.DataFrame([row_s2])

    print("\n[3/3] Runtime benchmark...")
    timing = benchmark_minimax(mu, erp, srp, grp, raw_esg, beta,
                                sample_sizes=[50, 100, 150], n_timing_runs=3)

    summary_cols_s2 = ["Portfolio_ID","Combo_Name","Annual_Return","Annual_Volatility",
                        "Sharpe_Ratio","ESG_Score","Minimax_T"]
    with pd.ExcelWriter("minimax_results.xlsx", engine="openpyxl") as writer:
        df_s2[summary_cols_s2].to_excel(writer, sheet_name="Stage2_MaxSharpe", index=False)
        df_s2.to_excel(writer, sheet_name="Stage2_Weights", index=False)
        pd.DataFrame([{"Asset_Size": sz, "Mean_Time_s": v["mean"], "Std_Time_s": v["std"]}
                      for sz, v in timing.items()]).to_excel(
            writer, sheet_name="Timing", index=False)
    print("  minimax_results.xlsx saved.")

    print("\n[3/3] Generating visualisations...")
    plot_minimax_portfolio(df_s2, tickers, esg_df)

    print("\n" + "=" * 65)
    print("  Files saved:")
    print("    minimax_results.xlsx")
    print("    minimax_portfolio_plots.png")
    print("=" * 65)
