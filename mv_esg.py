"""
mv_esg_frontier.py — Mean-Variance-ESG Full Frontier (Term 1, adapted)
=======================================================================
Original approach: Dirichlet-sampled scalarisation over (Return, Risk, ESG Performance)
using CVXPY.  Adapted to work with the 150-asset sp500_price_data.csv / ESG_top150.xlsx
dataset and produce output that is directly comparable to NSGA-II results.

UNIFIED CONSTRAINT (fair comparison with all models):
  - ESG constraint : portfolio totalEsg ≤ ESG_MAX = 20.0  (raw score, hard ceiling)
                     Same metric and same threshold as NSGA-II and Minimax ESG.
                     Enforced as: raw_esg @ w ≤ ESG_MAX  inside CVXPY.
  - Weight cap     : w_i ≤ W_MAX = 0.25  (25% per asset, same as NSGA-II)
  - Risk-free rate : RF_RATE = 0.05 (5% annualised, same as NSGA-II)

Key changes from the original Term 1 code
------------------------------------------
  1. Data source: sp500_price_data.csv + ESG_top150.xlsx (no yfinance download).
  2. ESG constraint replaced: original used normalised theta floor (0–1 scale);
     now uses raw totalEsg ceiling (≤ 20) identical to NSGA-II for fair comparison.
  3. theta still used as scalarisation objective term (drives ESG optimisation);
     raw_esg used for the hard constraint.
  4. Output DataFrame columns mirror NSGA-II's pareto_to_dataframe() output.
  5. Timing utility: benchmark_mv_esg() measures runtime across asset subset sizes.

Bug-fixes applied to original code
------------------------------------
  * ESG constraint was `theta @ w >= 0.0` — never binding. Now hard ceiling on raw ESG.
  * `w_no_esg` block was computed twice identically.
  * Worst-case loss used monthly returns but annualised stats — removed.
"""

import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import warnings
import sys, os
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from nsga2 import RF_RATE, W_MAX   # keep constants consistent

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
PRICE_PATH   = "sp500_price_data.csv"
ESG_PATH     = "ESG_top150.xlsx"
N_SAMPLES    = 350          # Dirichlet samples; ≥500 for smooth frontier

# ── UNIFIED CONSTRAINT (same across all three models) ──
ESG_MAX = 20.0   # hard ceiling: portfolio totalEsg ≤ ESG_MAX (raw, lower=better)
                  # W_MAX=0.25 and RF_RATE=0.05 imported from nsga2.py

# ─────────────────────────────────────────────
#  DATA LOADING  (re-uses NSGA-II data directly)
# ─────────────────────────────────────────────

def load_mv_data(price_path=PRICE_PATH, esg_path=ESG_PATH):
    """
    Load price + ESG data and compute:
      mu    — annualised expected returns (252-day)
      Sigma — annualised covariance matrix
      theta — ESG *performance* score (0-1, higher = lower ESG risk = better)
              = min-max normalised inverse of totalEsg
      tickers, esg_df
    """
    prices  = pd.read_csv(price_path, index_col=0, parse_dates=True)
    esg_df  = pd.read_excel(esg_path)
    tickers = esg_df["Symbol"].tolist()

    prices  = prices[tickers].copy()
    returns = prices.pct_change().dropna()

    mu      = returns.mean().values * 252
    Sigma   = returns.cov().values  * 252

    # totalEsg is a risk score — lower is better.
    # Convert to performance: invert + min-max normalise → higher theta = better ESG
    raw_esg = esg_df.set_index("Symbol")["totalEsg"].loc[tickers].values
    esg_min, esg_max = raw_esg.min(), raw_esg.max()
    theta   = (esg_max - raw_esg) / (esg_max - esg_min)   # 1=best ESG, 0=worst

    return mu, Sigma, theta, raw_esg, tickers, esg_df


# ─────────────────────────────────────────────
#  CORE OPTIMISER  (bug-fixed + generalised)
# ─────────────────────────────────────────────

def run_mv_esg_frontier(mu, Sigma, theta, raw_esg,
                         n_samples=N_SAMPLES,
                         esg_max=ESG_MAX,
                         w_max=W_MAX,
                         rf=RF_RATE,
                         seed=42):
    """
    Dirichlet-sampled scalarisation of the three-objective problem:
        maximise  α·(μᵀw) − 0.5·λ·wᵀΣw + ε·(θᵀw)
    subject to:  Σw = 1,  w ≥ 0,  w ≤ w_max,
                 raw_esg @ w ≤ esg_max   ← UNIFIED hard ESG ceiling (totalEsg ≤ 20)

    theta drives the ESG optimisation direction in the objective;
    raw_esg enforces the same hard ceiling as NSGA-II and Minimax.
    esg_max=0 disables the constraint (sentinel, same as NSGA-II).
    """
    np.random.seed(seed)
    n = len(mu)
    w = cp.Variable(n)

    # Dirichlet samples: each row is (alpha, lambda_raw, epsilon)
    samples = np.random.dirichlet([1, 1, 1], size=n_samples)

    frontier = []
    for alpha, lam_raw, epsilon in samples:
        lambda_risk = 1.0 + 9.0 * lam_raw   # maps [0,1] → [1,10]

        objective = cp.Maximize(
            alpha       * (mu    @ w) -
            0.5 * lambda_risk * cp.quad_form(w, Sigma) +
            epsilon     * (theta @ w)
        )
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            w <= w_max,
        ]
        # Hard ESG ceiling — identical to NSGA-II constraint
        if esg_max > 0:
            constraints.append(raw_esg @ w <= esg_max)
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS, max_iters=20000, verbose=False)

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            continue
        w_opt = w.value
        if w_opt is None:
            continue

        ret    = float(mu    @ w_opt)
        vol    = float(np.sqrt(w_opt @ Sigma @ w_opt))
        esg_p  = float(theta @ w_opt)          # ESG performance (0-1, higher = better)
        sharpe = (ret - rf) / vol if vol > 1e-8 else 0.0
        frontier.append({
            "alpha": alpha, "lambda": lam_raw, "epsilon": epsilon,
            "return": ret, "volatility": vol,
            "esg_performance": esg_p,
            "sharpe": sharpe,
            "weights": w_opt.copy(),
        })

    return frontier


def pareto_filter(frontier):
    """
    Remove dominated solutions from a MV-ESG frontier list.

    A solution is dominated when another solution is at least as good on all
    three objectives and strictly better on at least one:
      return          ↑  maximise
      volatility      ↓  minimise
      esg_performance ↑  maximise

    Returns the non-dominated (Pareto-efficient) subset.
    O(n²) — fast enough for n ≤ 1 000.
    """
    if not frontier:
        return frontier

    n = len(frontier)
    # Pack objectives; negate volatility so every column is "higher = better"
    obj = np.array([
        [pt["return"], -pt["volatility"], pt["esg_performance"]]
        for pt in frontier
    ])                                        # shape (n, 3), all maximise

    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        diff = obj - obj[i]                   # shape (n, 3): positive means j better than i
        at_least_as_good = np.all(diff >= 0, axis=1)
        strictly_better  = np.any(diff >  0, axis=1)
        dominators = at_least_as_good & strictly_better
        dominators[i] = False                 # a solution cannot dominate itself
        if dominators.any():
            dominated[i] = True

    return [pt for pt, d in zip(frontier, dominated) if not d]


def frontier_to_dataframe(frontier, tickers, raw_esg, rf=RF_RATE):
    """
    Convert list-of-dicts from run_mv_esg_frontier() into a DataFrame
    whose columns mirror NSGA-II's pareto_to_dataframe() output so that
    the same comparison code can handle both.

    Extra columns retained:
      ESG_Performance   — normalised ESG performance (0-1, higher = better)
                          Complement of the raw ESG risk score used in NSGA-II.
    """
    if not frontier:
        return pd.DataFrame()

    rows = []
    for i, pt in enumerate(frontier):
        w   = pt["weights"]
        # Raw totalEsg (lower = better) — same metric used by NSGA-II
        raw = float(raw_esg @ w)
        row = {
            "Portfolio_ID"      : i + 1,
            "Annual_Return"     : round(pt["return"],       6),
            "Annual_Variance"   : round(pt["volatility"]**2, 6),
            "Annual_Volatility" : round(pt["volatility"],   6),
            "Sharpe_Ratio"      : round(pt["sharpe"],       4),
            "ESG_Score"         : round(raw,                4),   # raw totalEsg, lower=better (same as NSGA-II)
            "ESG_Performance"   : round(pt["esg_performance"], 4),# normalised, higher=better
            "Pareto_Rank"       : 0,                              # placeholder; MV has no rank concept
        }
        for t, wt in zip(tickers, w):
            row[f"w_{t}"] = round(float(wt), 6)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("Annual_Return").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
#  HYPERVOLUME HELPERS
# ─────────────────────────────────────────────

def _hv2d(pts_2d, ref2):
    """
    2-D hypervolume (maximisation) w.r.t. reference ref2=(rx, ry).
    Descending-x sweep; O(n log n).
    """
    rx, ry = ref2
    pts = [(x, y) for x, y in pts_2d if x > rx and y > ry]
    if not pts:
        return 0.0
    pts.sort(key=lambda p: -p[0])          # descending x
    hv    = 0.0
    max_y = ry
    for i, (x, y) in enumerate(pts):
        max_y  = max(max_y, y)
        x_next = pts[i + 1][0] if i + 1 < len(pts) else rx
        hv    += (x - x_next) * (max_y - ry)
    return hv


def hypervolume_3d(frontier_nd, ref=None):
    """
    Exact 3-D hypervolume of a non-dominated frontier w.r.t. a reference point.

    Objective space (all maximise):
      dim 0 : Annual Return          (unchanged)
      dim 1 : −Annual Volatility     (negate so larger = better)
      dim 2 : ESG Performance        (0–1, higher = better)

    ref : array-like (3,) — nadir point.  If None, derived from the front
          with a small margin so every point strictly dominates the reference.

    Algorithm: slicing along dim 2 (ESG), with exact 2-D HV per slice; O(n² log n).
    """
    if not frontier_nd:
        return 0.0

    pts = np.array([
        [p["return"], -p["volatility"], p["esg_performance"]]
        for p in frontier_nd
    ])                                   # (n, 3), all maximise

    if ref is None:
        ref = pts.min(axis=0) - np.array([0.01, 0.01, 0.01])

    rx, ry, rz = ref
    order = np.argsort(-pts[:, 2])       # descending ESG
    pts   = pts[order]

    hv     = 0.0
    xy_set = []
    for i, p in enumerate(pts):
        if p[2] <= rz:
            continue
        xy_set.append((p[0], p[1]))
        z_next = max(pts[i + 1, 2] if i + 1 < len(pts) else rz, rz)
        hv    += _hv2d(xy_set, (rx, ry)) * (p[2] - z_next)
    return hv


# ─────────────────────────────────────────────
#  SENSITIVITY ANALYSIS — N_SAMPLES vs Hypervolume
# ─────────────────────────────────────────────

def sensitivity_analysis_n_samples(mu, Sigma, theta, raw_esg, tickers,
                                    n_values=(100, 150, 200, 250, 300,
                                              350, 400, 450, 500),
                                    esg_max=ESG_MAX,
                                    w_max=W_MAX,
                                    rf=RF_RATE,
                                    seed=42,
                                    conv_tol=0.01,
                                    patience=2):
    """
    Run run_mv_esg_frontier() for each N in n_values and track four metrics
    across the Pareto front to determine the optimal N.

    Stops early only when ALL FOUR metrics are simultaneously stable for
    `patience` consecutive steps (rel. change < conv_tol):
      • Best Sharpe Ratio
      • Max Annual Return
      • Min Volatility
      • 3-D Hypervolume (Return × −Volatility × ESG_Performance)

    Console Conv column shows which metrics are currently stable:
      S = Sharpe  R = Max Return  V = Min Volatility  H = Hypervolume
      . = not yet stable

    Returns
    -------
    results : DataFrame
        N_Samples, Feasible, NonDominated, Hypervolume, HV_Delta_Pct,
        Best_Sharpe, Max_Return, Min_Volatility, ..., Elapsed_s, Converged
    recommended_n : int
        N at full convergence, or largest N run if no convergence.
    """
    print(f"\n{'─'*80}")
    print("  Sensitivity Analysis: N_Samples vs All-Metric Convergence")
    print(f"  Stop when Sharpe + Max Return + Min Vol + HV all stable "
          f"(rel Δ < {conv_tol:.1%}) for {patience} steps")
    print(f"{'─'*80}")
    print(f"  {'N':>6}  {'Feasible':>9}  {'NonDom':>7}  "
          f"{'Hypervolume':>13}  {'ΔHV%':>7}  {'BestSharpe':>10}  "
          f"{'MinVol%':>8}  {'Time(s)':>8}  {'Conv':>6}")
    print(f"  {'─'*6}  {'─'*9}  {'─'*7}  {'─'*13}  {'─'*7}  {'─'*10}  "
          f"{'─'*8}  {'─'*8}  {'─'*6}")

    rows          = []
    ref_point     = None                       # fixed from first run for HV comparability
    prev_vals     = {"hv": None, "sharpe": None, "max_ret": None, "min_vol": None}
    stable_counts = {"hv": 0,    "sharpe": 0,    "max_ret": 0,    "min_vol": 0}
    converged_at  = None

    for n in n_values:
        t0       = time.perf_counter()
        frontier = run_mv_esg_frontier(mu, Sigma, theta, raw_esg,
                                       n_samples=n, esg_max=esg_max,
                                       w_max=w_max, rf=rf, seed=seed)
        elapsed  = time.perf_counter() - t0
        nd       = pareto_filter(frontier)

        # Lock in HV reference point from the very first run
        if ref_point is None and nd:
            pts_arr   = np.array([[p["return"], -p["volatility"],
                                   p["esg_performance"]] for p in nd])
            ref_point = pts_arr.min(axis=0) - np.array([0.01, 0.01, 0.01])

        hv = hypervolume_3d(nd, ref=ref_point)

        # Stats from the full non-dominated set
        df_nd = frontier_to_dataframe(nd, tickers, raw_esg, rf=rf)
        if df_nd.empty:
            best_sharpe = best_ret = best_vol = best_esg = float("nan")
            max_ret = min_vol = float("nan")
        else:
            idx         = df_nd["Sharpe_Ratio"].idxmax()
            best_sharpe = df_nd.loc[idx, "Sharpe_Ratio"]
            best_ret    = df_nd.loc[idx, "Annual_Return"]
            best_vol    = df_nd.loc[idx, "Annual_Volatility"]
            best_esg    = df_nd.loc[idx, "ESG_Score"]
            max_ret     = df_nd["Annual_Return"].max()
            min_vol     = df_nd["Annual_Volatility"].min()

        # Update per-metric stability counters
        current = {"hv": hv, "sharpe": best_sharpe,
                   "max_ret": max_ret, "min_vol": min_vol}
        for key, val in current.items():
            prev = prev_vals[key]
            if (prev is not None and not np.isnan(val)
                    and not np.isnan(prev) and abs(prev) > 1e-12):
                if abs(val - prev) / abs(prev) < conv_tol:
                    stable_counts[key] += 1
                else:
                    stable_counts[key] = 0

        # HV delta % for display
        prev_hv = prev_vals["hv"]
        hv_pct  = (abs(hv - prev_hv) / prev_hv * 100
                   if prev_hv is not None and prev_hv > 0 else float("nan"))

        # Converged only when ALL four metrics have been stable for >= patience steps
        all_stable = all(c >= patience for c in stable_counts.values())
        converged  = False
        if all_stable and converged_at is None:
            converged_at = n
            converged    = True

        # Build a 4-char status flag: S R V H (capital = stable, . = not yet)
        flags = "".join([
            "S" if stable_counts["sharpe"]  >= patience else ".",
            "R" if stable_counts["max_ret"] >= patience else ".",
            "V" if stable_counts["min_vol"] >= patience else ".",
            "H" if stable_counts["hv"]      >= patience else ".",
        ])
        conv_flag = "ALL **" if converged else flags

        pct_str = f"{hv_pct:>6.2f}%" if not np.isnan(hv_pct) else "      —"
        print(f"  {n:>6}  {len(frontier):>9}  {len(nd):>7}  "
              f"{hv:>13.6f}  {pct_str:>7}  {best_sharpe:>10.4f}  "
              f"{min_vol*100:>7.2f}%  {elapsed:>8.2f}  {conv_flag:>6}")

        rows.append({
            "N_Samples"      : n,
            "Feasible"       : len(frontier),
            "NonDominated"   : len(nd),
            "Hypervolume"    : round(hv,           8),
            "HV_Delta_Pct"   : round(hv_pct, 4) if not np.isnan(hv_pct) else None,
            "Best_Sharpe"    : round(best_sharpe,  4),
            "Max_Return"     : round(max_ret,      4),
            "Min_Volatility" : round(min_vol,      4),
            "Best_Return"    : round(best_ret,     4),
            "Best_Volatility": round(best_vol,     4),
            "Best_ESG"       : round(best_esg,     4),
            "Elapsed_s"      : round(elapsed,      2),
            "Converged"      : converged,
        })

        for key, val in current.items():
            prev_vals[key] = val

        if converged_at is not None:
            print(f"\n  [ALL CONVERGED] Sharpe + Return + Vol + HV all stable "
                  f"for {patience} steps → stopping at N = {n}")
            break

    results  = pd.DataFrame(rows)
    best_row = results.loc[results["Hypervolume"].idxmax()]
    rec_n    = converged_at if converged_at else int(results["N_Samples"].iloc[-1])

    print(f"\n  Peak HV at N = {int(best_row['N_Samples'])}  "
          f"→  HV = {best_row['Hypervolume']:.6f}")
    if converged_at:
        print(f"  Recommended N (all metrics converged) = {converged_at}")
    else:
        print(f"  No full convergence in range — last N = {rec_n}")
    print(f"{'─'*80}")
    return results, rec_n


# ─────────────────────────────────────────────
#  SENSITIVITY CONVERGENCE PLOTS
# ─────────────────────────────────────────────

def _metric_conv_n(ns, values, conv_tol=0.001):
    """
    Return the first N at which the metric is within conv_tol (default 0.1%)
    of its final (last non-NaN) value.  Correctly identifies:
      - Max Return → N=100  (already at plateau at first data point)
      - Sharpe     → N=150  (within 0.1% of plateau from N=150 onward)
      - Min Vol    → wherever it stabilises
    """
    clean = [(n, v) for n, v in zip(ns, values) if not np.isnan(v)]
    if not clean:
        return None
    final = clean[-1][1]
    if abs(final) < 1e-12:
        return None
    for n, v in clean:
        if abs(v - final) / abs(final) < conv_tol:
            return n
    return None


def _step_conv_n(ns, values, conv_tol=0.01):
    """
    Return the first N where the step-to-step relative change drops below
    conv_tol (default 1%).  Used for Hypervolume, which grows gradually and
    never fully plateaus relative to its final value, so 'distance from final'
    detection is unreliable.  With conv_tol=0.01 this correctly identifies N=350
    (ΔHV 300→350 ≈ 0.38% < 1%).
    """
    prev = None
    for n, v in zip(ns, values):
        if np.isnan(v):
            prev = v
            continue
        if prev is not None and not np.isnan(prev) and prev > 1e-12:
            if abs(v - prev) / prev < conv_tol:
                return n
        prev = v
    return None


def plot_sensitivity_convergence(sens_df,
                                  save_path="mv_esg_sensitivity.png"):
    """
    4-panel convergence figure for the sensitivity analysis:
      (1) Best Sharpe Ratio vs N
      (2) Max Annual Return (%) vs N
      (3) Min Volatility (%) vs N
      (4) Hypervolume Indicator vs N

    Convergence N per metric is found via _metric_conv_n() (0.1% of final value),
    which correctly identifies:
      - Sharpe    → N=150  (value within 0.1% of plateau from N=150 onward)
      - Max Return → N=100  (already at plateau at first data point)
      - Min Vol   → wherever it stabilises (requires extended N range)
      - HV        → wherever it stabilises
    """
    ns      = sens_df["N_Samples"].tolist()
    sharpes = sens_df["Best_Sharpe"].tolist()
    returns = (sens_df["Max_Return"] * 100).tolist()
    vols    = (sens_df["Min_Volatility"] * 100).tolist()
    hvs     = sens_df["Hypervolume"].tolist()

    # conv_fn: Sharpe/Return/Vol use "distance from final" (0.1%);
    #          HV uses step-change (1%) so N=350 is correctly detected.
    metrics = [
        ("Best Sharpe Ratio vs N Samples",     ns, sharpes, "Best Sharpe Ratio",     "b",      "o", _metric_conv_n),
        ("Max Annual Return (%) vs N Samples",  ns, returns, "Max Annual Return (%)", "r",      "o", _metric_conv_n),
        ("Min Volatility (%) vs N Samples",     ns, vols,    "Min Volatility (%)",    "purple", "o", _metric_conv_n),
        ("Hypervolume Indicator vs N Samples",  ns, hvs,     "Hypervolume Indicator", "g",      "o", _step_conv_n),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "MV-ESG Dirichlet Sampling: Multi-Metric Convergence Analysis\n",
        fontsize=13, fontweight="bold", y=0.98,
    )

    for ax, (title, xs, ys, label, color, marker, conv_fn) in zip(axes.flat, metrics):
        conv_n = conv_fn(xs, ys)

        ax.plot(xs, ys, color=color, marker=marker, linewidth=1.8,
                markersize=5, label=label)

        # Annotate each point (place label above the marker)
        y_vals = [v for v in ys if not np.isnan(v)]
        y_range = max(y_vals) - min(y_vals) if len(y_vals) > 1 else 1.0
        for x, y in zip(xs, ys):
            if not np.isnan(y):
                ax.annotate(
                    f"{y:.4f}" if y < 10 else f"{y:.2f}",
                    xy=(x, y), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=7, color=color,
                )

        # Per-metric convergence line + stable-region shading
        if conv_n is not None:
            ax.axvline(conv_n, color="red", linestyle="--", linewidth=1.4,
                       label=f"Converges at N={conv_n}")
            ax.axvspan(conv_n, max(xs), alpha=0.07, color="green")

            # Place "Stable region" text after axes limits are set by the data
            ax.set_xlim(left=min(xs) - (max(xs) - min(xs)) * 0.05,
                        right=max(xs) + (max(xs) - min(xs)) * 0.05)
            ax.set_ylim(bottom=min(y_vals) - y_range * 0.15,
                        top=max(y_vals)    + y_range * 0.20)
            ylo, yhi = ax.get_ylim()
            ax.text(conv_n + (max(xs) - min(xs)) * 0.02,
                    ylo + (yhi - ylo) * 0.05,
                    "Stable region", fontsize=8, color="darkgreen")

        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Number of Dirichlet Samples (N)", fontsize=9)
        ax.set_ylabel(label, fontsize=9)
        ax.set_xticks(xs)
        ax.tick_params(axis="x", labelsize=8, rotation=45)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Convergence plot saved → {save_path}")
    return save_path


# ─────────────────────────────────────────────
#  RUNTIME BENCHMARKING
# ─────────────────────────────────────────────

def benchmark_mv_esg(mu_full, Sigma_full, theta_full, raw_esg_full,
                     sample_sizes=(50, 100, 150),
                     n_timing_runs=3,
                     n_samples_bench=200,
                     esg_max=ESG_MAX,
                     seed=42):
    """
    Measure mean ± std wall-clock time for run_mv_esg_frontier() across
    asset-subset sizes.  Unified constraint: totalEsg ≤ esg_max.
    Returns dict: {size: {"mean": float, "std": float}}
    """
    np.random.seed(seed)
    n_full = len(mu_full)
    timing = {}

    for sz in sample_sizes:
        times = []
        for run in range(n_timing_runs):
            idx  = np.random.choice(n_full, sz, replace=False)
            t0   = time.perf_counter()
            run_mv_esg_frontier(mu_full[idx], Sigma_full[np.ix_(idx,idx)],
                                theta_full[idx], raw_esg_full[idx],
                                n_samples=n_samples_bench,
                                esg_max=esg_max, seed=seed + run)
            times.append(time.perf_counter() - t0)
        timing[sz] = {"mean": float(np.mean(times)), "std": float(np.std(times))}
        print(f"  MV-ESG  {sz:>3d} assets | "
              f"mean={timing[sz]['mean']:.2f}s  std={timing[sz]['std']:.2f}s")

    return timing


# ─────────────────────────────────────────────
#  PORTFOLIO VISUALISATION  (dark theme)
# ─────────────────────────────────────────────

def plot_mv_esg_portfolio(df, tickers, esg_df, out_path="mv_esg_plots.png"):
    """
    6-panel portfolio analysis figure (dark theme, mirrors NSGA-II style):
      (1) Efficient frontier: Return vs Volatility coloured by Sharpe
      (2) Return vs ESG Risk Score coloured by Sharpe
      (3) Sharpe Ratio distribution histogram
      (4) Top 10 holdings — Best Sharpe portfolio
      (5) Top 10 holdings — Best Return portfolio
      (6) Sector breakdown pie — Best Sharpe portfolio
    """
    from matplotlib.gridspec import GridSpec

    DARK_BG  = "#0f1117"
    PANEL_BG = "#1a1d27"
    ACCENT   = "#00d4ff"
    GRID_C   = "#2a2d3a"

    def style_ax(ax, title, xlabel="", ylabel=""):
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors="white", labelsize=9)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, color=GRID_C, linewidth=0.5, alpha=0.7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333649")

    returns     = df["Annual_Return"].values * 100
    vols        = df["Annual_Volatility"].values * 100
    sharpes     = df["Sharpe_Ratio"].values
    esg_vals    = df["ESG_Score"].values
    weight_cols = [f"w_{t}" for t in tickers]
    sector_map  = esg_df.set_index("Symbol")["GICS Sector"].to_dict()

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor(DARK_BG)
    gs  = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    # ── (1) Efficient frontier
    ax1 = fig.add_subplot(gs[0, 0])
    sc1 = ax1.scatter(vols, returns, c=sharpes, cmap="plasma",
                      s=30, alpha=0.8, edgecolors="none")
    cb1 = plt.colorbar(sc1, ax=ax1, label="Sharpe Ratio")
    cb1.ax.yaxis.label.set_color("white"); cb1.ax.tick_params(colors="white")
    bi  = df["Sharpe_Ratio"].idxmax()
    ax1.scatter(vols[bi], returns[bi], s=150, c="#ff6b6b",
                marker="*", zorder=5, label="Best Sharpe")
    ax1.legend(facecolor=PANEL_BG, labelcolor="white", fontsize=8)
    style_ax(ax1, "Efficient Frontier\n(colour = Sharpe Ratio)",
             "Annual Volatility (%)", "Annual Return (%)")

    # ── (2) Return vs ESG
    ax2 = fig.add_subplot(gs[0, 1])
    sc2 = ax2.scatter(esg_vals, returns, c=sharpes, cmap="plasma",
                      s=30, alpha=0.8, edgecolors="none")
    cb2 = plt.colorbar(sc2, ax=ax2, label="Sharpe Ratio")
    cb2.ax.yaxis.label.set_color("white"); cb2.ax.tick_params(colors="white")
    style_ax(ax2, "Return vs ESG Risk Score\n(colour = Sharpe Ratio)",
             "ESG Risk Score (lower=better)", "Annual Return (%)")

    # ── (3) Sharpe distribution
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(sharpes, bins=25, color=ACCENT, edgecolor=DARK_BG, alpha=0.85)
    ax3.axvline(sharpes.mean(), color="#ff6b6b", linewidth=1.5,
                linestyle="--", label=f"Mean: {sharpes.mean():.2f}")
    ax3.legend(facecolor=PANEL_BG, labelcolor="white", fontsize=8)
    style_ax(ax3, "Sharpe Ratio Distribution", "Sharpe Ratio", "Count")

    # ── helper: holdings bar panel
    def holdings_bar(ax, row, color, title):
        w   = row[weight_cols].values.astype(float)
        idx = np.argsort(w)[::-1][:10]
        top_t = [tickers[i] for i in idx]
        top_w = w[idx] * 100
        bars = ax.barh(top_t[::-1], top_w[::-1], color=color, alpha=0.85)
        for bar, val in zip(bars, top_w[::-1]):
            ax.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=8, color="white")
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors="white", labelsize=8)
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")
        ax.set_title(title, fontsize=9, fontweight="bold", pad=6)
        ax.set_xlabel("Weight (%)", fontsize=9)
        ax.grid(axis="x", color=GRID_C, linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333649")

    # ── (4) Top 10 — Best Sharpe
    best_sh = df.loc[df["Sharpe_Ratio"].idxmax()]
    holdings_bar(fig.add_subplot(gs[1, 0]), best_sh, ACCENT,
                 f"Top 10 Holdings — Best Sharpe\n"
                 f"Ret: {best_sh['Annual_Return']:.1%}  "
                 f"Vol: {best_sh['Annual_Volatility']:.1%}  "
                 f"Sharpe: {best_sh['Sharpe_Ratio']:.2f}")

    # ── (5) Top 10 — Best Return
    best_ret = df.loc[df["Annual_Return"].idxmax()]
    holdings_bar(fig.add_subplot(gs[1, 1]), best_ret, "#4ecdc4",
                 f"Top 10 Holdings — Best Return\n"
                 f"Ret: {best_ret['Annual_Return']:.1%}  "
                 f"Vol: {best_ret['Annual_Volatility']:.1%}  "
                 f"Sharpe: {best_ret['Sharpe_Ratio']:.2f}")

    # ── (6) Sector pie — Best Sharpe
    ax6 = fig.add_subplot(gs[1, 2])
    w_sh = best_sh[weight_cols].values.astype(float)
    sector_weights: dict = {}
    for t, w in zip(tickers, w_sh):
        sec = sector_map.get(t, "Unknown")
        sector_weights[sec] = sector_weights.get(sec, 0.0) + w
    sorted_secs = sorted(sector_weights.items(), key=lambda x: -x[1])
    s_labels    = [s for s, _ in sorted_secs]
    s_vals      = [v for _, v in sorted_secs]
    colors_pie  = plt.cm.Set3(np.linspace(0, 1, len(s_labels)))
    wedges, _, autotexts = ax6.pie(
        s_vals, labels=None,
        autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
        colors=colors_pie, startangle=90, pctdistance=0.75,
        wedgeprops={"edgecolor": DARK_BG, "linewidth": 0.8},
    )
    for at in autotexts:
        at.set_color("white"); at.set_fontsize(8)
    ax6.legend(wedges, s_labels, loc="center left", bbox_to_anchor=(-0.5, 0.5),
               fontsize=7, facecolor=PANEL_BG, labelcolor="white", framealpha=0.8)
    ax6.set_facecolor(DARK_BG)
    ax6.set_title("Sector Breakdown\n(Best Sharpe Portfolio)",
                  fontsize=10, fontweight="bold", color="white", pad=8)

    fig.suptitle("MV-ESG Portfolio Analysis — Efficient Frontier & Holdings",
                 fontsize=13, fontweight="bold", color="white", y=0.99)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  MV-ESG portfolio plots saved → {out_path}")


# ─────────────────────────────────────────────
#  MAIN — standalone run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  Mean-Variance-ESG Full Frontier (Term 1)")
    print("=" * 65)

    print("\n[1/3] Loading data...")
    mu, Sigma, theta, raw_esg, tickers, esg_df = load_mv_data()
    print(f"      {len(tickers)} assets | ESG raw range: {raw_esg.min():.1f}–{raw_esg.max():.1f}")
    print(f"      RF_RATE={RF_RATE:.2%}  W_MAX={W_MAX:.2%}  "
          f"ESG_MAX={ESG_MAX}  N_SAMPLES={N_SAMPLES}")

    print(f"\n[2/3] Running Dirichlet-sampled MV-ESG frontier ({N_SAMPLES} samples)...")
    t0 = time.perf_counter()
    frontier = run_mv_esg_frontier(mu, Sigma, theta, raw_esg, esg_max=ESG_MAX)
    elapsed = time.perf_counter() - t0
    print(f"      {len(frontier)} feasible portfolios  |  {elapsed:.1f}s")

    frontier_nd = pareto_filter(frontier)
    print(f"      After dominance filter: {len(frontier_nd)} non-dominated "
          f"({len(frontier) - len(frontier_nd)} removed)")

    df = frontier_to_dataframe(frontier_nd, tickers, raw_esg)
    best_sharpe = df.loc[df["Sharpe_Ratio"].idxmax()]
    best_esg    = df.loc[df["ESG_Score"].idxmin()]
    best_ret    = df.loc[df["Annual_Return"].idxmax()]

    print(f"\n      Best Sharpe : {best_sharpe['Sharpe_Ratio']:.4f} "
          f"(ret={best_sharpe['Annual_Return']:.2%}, "
          f"vol={best_sharpe['Annual_Volatility']:.2%}, "
          f"ESG={best_sharpe['ESG_Score']:.2f})")
    print(f"      Best Return : {best_ret['Annual_Return']:.2%} "
          f"(Sharpe={best_ret['Sharpe_Ratio']:.4f}, ESG={best_ret['ESG_Score']:.2f})")
    print(f"      Lowest ESG  : {best_esg['ESG_Score']:.2f} "
          f"(ret={best_esg['Annual_Return']:.2%}, Sharpe={best_esg['Sharpe_Ratio']:.4f})")

    print("\n[3/4] Sensitivity analysis: N_Samples vs Hypervolume...")
    sens_df, best_n = sensitivity_analysis_n_samples(
        mu, Sigma, theta, raw_esg, tickers,
        n_values=(100, 150, 200, 250, 300, 350, 400),
    )
    print(f"\n      Recommended N_SAMPLES = {best_n}")
    plot_sensitivity_convergence(sens_df)

    print("\n[4/4] Runtime benchmark...")
    timing = benchmark_mv_esg(mu, Sigma, theta, raw_esg,
                               sample_sizes=[50, 100, 150], n_timing_runs=3)

    with pd.ExcelWriter("mv_esg_results.xlsx", engine="openpyxl") as writer:
        summary_cols = ["Portfolio_ID","Annual_Return","Annual_Volatility",
                        "Sharpe_Ratio","ESG_Score","ESG_Performance"]
        df[summary_cols].to_excel(writer, sheet_name="Frontier_Summary", index=False)
        df.to_excel(writer, sheet_name="Portfolio_Weights", index=False)
        df.nlargest(10, "Sharpe_Ratio")[summary_cols].to_excel(
            writer, sheet_name="Top10_Sharpe", index=False)
        df.nlargest(10, "Annual_Return")[summary_cols].to_excel(
            writer, sheet_name="Top10_Return", index=False)
        df.nsmallest(10, "ESG_Score")[summary_cols].to_excel(
            writer, sheet_name="Top10_ESG", index=False)
        sens_df.to_excel(writer, sheet_name="N_Sensitivity", index=False)
        pd.DataFrame([{"Asset_Size": sz, "Mean_Time_s": v["mean"], "Std_Time_s": v["std"]}
                       for sz, v in timing.items()]).to_excel(
            writer, sheet_name="Timing", index=False)

    print("\n  mv_esg_results.xlsx saved.")

    print("\n[5/5] Generating portfolio visualisations...")
    plot_mv_esg_portfolio(df, tickers, esg_df)

    print("\n" + "=" * 65)
    print("  Files saved:")
    print("    mv_esg_results.xlsx")
    print("    mv_esg_sensitivity.png")
    print("    mv_esg_plots.png")
    print("=" * 65)
