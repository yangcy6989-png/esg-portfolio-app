"""
NSGA-II Multi-Objective Portfolio Optimisation with ESG Constraints
====================================================================
Objectives:
  1. Maximise expected annual return       → minimise -return
  2. Minimise portfolio variance (risk)
  3. Minimise portfolio ESG risk score     (lower totalEsg = less unmanaged ESG risk = better)

Note on totalEsg (Yahoo Finance): this is an ESG *risk* rating — lower is better.
Higher scores indicate more unmanaged ESG risk, so we minimise this objective directly.

Constraints:
  • Weights sum to 1  (Σwᵢ = 1)
  • No short-selling  (wᵢ ≥ 0)
  • ESG ceiling       (Σwᵢ·ESGᵢ ≤ esg_max)   [constraint domination; 0 = no constraint]
"""

# Risk-free rate used in Sharpe ratio calculations.
# Set to 0.0 for a return/volatility ratio (common in academic comparisons).
# NOTE: the 2023-24 US T-bill yield averaged ~5.0-5.3%.  Setting RF_RATE=0.05
# would materially lower Sharpe values and can change portfolio rankings.
RF_RATE = 0.05   # annualised; change to e.g. 0.05 for T-bill-adjusted Sharpe

# Maximum weight allowed for any single asset.
# Prevents single-stock concentration (e.g. 100% NVDA) that produces
# unrealistic volatility figures.  With 150 assets and W_MAX=0.10 the
# portfolio must hold at least 10 positions.
W_MAX = 0.25    # 25% per asset; change to 1.0 to disable

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1.  DATA LOADING
# ─────────────────────────────────────────────

def load_data(price_path: str, esg_path: str):
    prices = pd.read_csv(price_path, index_col=0, parse_dates=True)
    esg_df = pd.read_excel(esg_path)

    tickers = esg_df["Symbol"].tolist()
    prices  = prices[tickers].copy()

    # Daily → annualised statistics
    returns     = prices.pct_change().dropna()
    mu          = returns.mean().values * 252          # annualised expected return
    cov         = returns.cov().values  * 252          # annualised covariance
    esg_scores  = esg_df.set_index("Symbol")["totalEsg"].loc[tickers].values
    names       = esg_df.set_index("Symbol")["Full Name"].loc[tickers].values
    sectors     = esg_df.set_index("Symbol")["GICS Sector"].loc[tickers].values

    return mu, cov, esg_scores, tickers, names, sectors


def _project_weights(w: np.ndarray, w_max: float) -> np.ndarray:
    """Project w onto {w ≥ 0, Σw = 1, w_i ≤ w_max} using iterative clipping.

    A single clip+renormalise pass fails when multiple weights simultaneously
    hit the cap (renormalisation can push them above w_max again).  Iterating
    until convergence solves this in at most n steps.
    """
    w = np.maximum(w, 0.0)
    for _ in range(len(w) + 1):
        total = w.sum()
        if total < 1e-12:
            # degenerate: spread equally across all assets
            return np.full(len(w), min(w_max, 1.0 / len(w)))
        w = w / total
        excess = w - w_max
        if excess.max() < 1e-9:
            break
        w = np.minimum(w, w_max)
    return w


# ─────────────────────────────────────────────
# 2.  PORTFOLIO INDIVIDUAL
# ─────────────────────────────────────────────

class Portfolio:
    """Represents one candidate solution (portfolio weights)."""

    def __init__(self, weights: np.ndarray):
        # Repair: project to box-simplex {w ≥ 0, Σw = 1, w_i ≤ W_MAX}
        self.weights = _project_weights(weights, W_MAX)

        # Objectives (set after evaluation)
        self.obj_return  = None   # to be maximised  → stored as negative
        self.obj_risk    = None   # to be minimised (raw variance — no penalty)
        self.obj_esg     = None   # to be minimised (lower totalEsg = less ESG risk)
        self.raw_risk    = None   # alias for obj_risk; kept for reporting clarity

        # Constraint handling (Deb et al. 2002 constraint domination)
        self.constraint_violation = 0.0   # ESG ceiling violation; 0 = feasible

        # NSGA-II bookkeeping
        self.rank              = None
        self.crowding_distance = 0.0
        self.domination_count  = 0
        self.dominated_set     = []

    @property
    def objectives(self):
        """Returns (neg_return, raw_variance, esg_risk) — all minimised. No penalties; violations handled via Deb constraint domination."""
        return (self.obj_return, self.obj_risk, self.obj_esg)

    def evaluate(self, mu, cov, esg_scores, esg_max):
        w = self.weights
        port_return = float(w @ mu)
        port_risk   = float(w @ cov @ w)   # raw variance
        port_esg    = float(w @ esg_scores)

        # ESG ceiling constraint (Deb et al. 2002 constraint domination — no penalty needed).
        # esg_max=0 is a sentinel meaning "no ESG constraint".
        # Any positive esg_max enforces portfolio_totalEsg ≤ esg_max.
        # Note: with typical S&P 500 scores ~9–42, an esg_max near 0 would be infeasible;
        # only esg_max=0 (sentinel) should be used when no constraint is desired.
        self.constraint_violation = max(0.0, port_esg - esg_max) if esg_max > 0 else 0.0

        self.raw_risk   =  port_risk    # raw variance — used in all reports/plots
        self.obj_return = -port_return  # minimise → negate
        self.obj_risk   =  port_risk    # minimise raw variance (no penalty; violation handled separately)
        self.obj_esg    =  port_esg     # minimise directly (lower totalEsg = less ESG risk = better)

    # Decoded (human-readable) values
    @property
    def annual_return(self): return -self.obj_return
    @property
    def annual_risk(self):
        if self.raw_risk is None:
            raise RuntimeError("Portfolio.raw_risk is None — call evaluate() before accessing annual_risk")
        return self.raw_risk   # raw variance, no penalty
    @property
    def esg_score(self):     return self.obj_esg    # lower = less ESG risk = better


# ─────────────────────────────────────────────
# 3.  NON-DOMINATED SORTING
# ─────────────────────────────────────────────

def dominates(a: Portfolio, b: Portfolio) -> bool:
    """Constraint domination (Deb et al. 2002 NSGA-II):

      - A feasible solution always dominates an infeasible one.
      - Two infeasible solutions: the one with less total violation dominates.
      - Two feasible solutions: standard Pareto dominance on objectives.

    This replaces the penalty method entirely, so no penalty coefficient needs
    tuning and the ESG constraint operates independently of objective scales.
    """
    a_viol = a.constraint_violation
    b_viol = b.constraint_violation
    a_feas = a_viol <= 0.0
    b_feas = b_viol <= 0.0

    if a_feas and not b_feas:
        return True                         # feasible beats infeasible
    if not a_feas and b_feas:
        return False                        # infeasible never beats feasible
    if not a_feas and not b_feas:
        return a_viol < b_viol              # both infeasible: less violation wins

    # Both feasible — standard Pareto dominance on objectives
    objs_a = a.objectives
    objs_b = b.objectives
    return (all(x <= y for x, y in zip(objs_a, objs_b)) and
            any(x <  y for x, y in zip(objs_a, objs_b)))


def non_dominated_sort(population: list) -> list:
    """Returns list-of-fronts (each front is a list of Portfolio objects).

    Uses the upper-triangle pass from Deb et al. (2002): each pair (i, j) with
    i < j is examined once, updating both sides simultaneously.  This halves the
    number of dominance comparisons vs the naive N×N grid while keeping the same
    O(MN²) worst-case complexity (M = number of objectives).
    """
    n = len(population)
    for p in population:
        p.domination_count = 0
        p.dominated_set    = []

    fronts = [[]]
    for i in range(n):
        pi = population[i]
        for j in range(i + 1, n):
            pj = population[j]
            if dominates(pi, pj):
                pi.dominated_set.append(pj)
                pj.domination_count += 1
            elif dominates(pj, pi):
                pj.dominated_set.append(pi)
                pi.domination_count += 1
        if pi.domination_count == 0:
            pi.rank = 0
            fronts[0].append(pi)

    k = 0
    while fronts[k]:
        next_front = []
        for p in fronts[k]:
            for q in p.dominated_set:
                q.domination_count -= 1
                if q.domination_count == 0:
                    q.rank = k + 1
                    next_front.append(q)
        k += 1
        fronts.append(next_front)

    return [f for f in fronts if f]   # drop empty last list


# ─────────────────────────────────────────────
# 4.  CROWDING DISTANCE
# ─────────────────────────────────────────────

def crowding_distance_assignment(front: list):
    n = len(front)
    if n == 0:
        return
    num_obj = 3
    for p in front:
        p.crowding_distance = 0.0

    for m in range(num_obj):
        front.sort(key=lambda p: p.objectives[m])
        obj_min = front[0].objectives[m]
        obj_max = front[-1].objectives[m]
        front[0].crowding_distance  = float("inf")
        front[-1].crowding_distance = float("inf")
        span = obj_max - obj_min if obj_max != obj_min else 1e-10
        for i in range(1, n - 1):
            front[i].crowding_distance += (
                (front[i + 1].objectives[m] - front[i - 1].objectives[m]) / span
            )


# ─────────────────────────────────────────────
# 5.  SELECTION, CROSSOVER, MUTATION
# ─────────────────────────────────────────────

def crowded_comparison(a: Portfolio, b: Portfolio) -> Portfolio:
    """NSGA-II crowded comparison operator."""
    if a.rank < b.rank:
        return a
    if b.rank < a.rank:
        return b
    if a.crowding_distance > b.crowding_distance:
        return a
    return b


def tournament_selection(population: list) -> Portfolio:
    """Binary tournament selection."""
    a, b = random.sample(population, 2)
    return crowded_comparison(a, b)


def sbx_crossover(p1: Portfolio, p2: Portfolio, eta_c: float = 15.0, prob: float = 0.9):
    """Simulated Binary Crossover (SBX) on weight vectors."""
    if random.random() > prob:
        # Return fresh Portfolio objects from weight vectors — deepcopy is unsafe
        # here because dominated_set forms a recursive reference graph after
        # non_dominated_sort, which exceeds Python's recursion limit.
        return Portfolio(p1.weights.copy()), Portfolio(p2.weights.copy())

    n  = len(p1.weights)
    c1 = p1.weights.copy()
    c2 = p2.weights.copy()

    for i in range(n):
        if random.random() > 0.5:
            continue
        if abs(p1.weights[i] - p2.weights[i]) < 1e-10:
            continue
        y1 = min(p1.weights[i], p2.weights[i])
        y2 = max(p1.weights[i], p2.weights[i])
        u  = random.random()

        beta_q = (2 * u) ** (1 / (eta_c + 1)) if u <= 0.5 else \
                 (1 / (2 * (1 - u))) ** (1 / (eta_c + 1))

        c1[i] = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))
        c2[i] = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))

    return Portfolio(c1), Portfolio(c2)


def polynomial_mutation(ind: Portfolio, eta_m: float = 20.0, prob: float = None):
    """Polynomial mutation on weight vector."""
    n    = len(ind.weights)
    prob = prob or (1.0 / n)
    w    = ind.weights.copy()

    for i in range(n):
        if random.random() < prob:
            delta1 = w[i]           # distance to lower bound (0)
            delta2 = W_MAX - w[i]   # distance to upper bound (W_MAX)
            u      = random.random()
            if u < 0.5:
                delta_q = (2 * u + (1 - 2 * u) * (1 - delta1) ** (eta_m + 1)) ** (1 / (eta_m + 1)) - 1
            else:
                delta_q = 1 - (2 * (1 - u) + 2 * (u - 0.5) * (1 - delta2) ** (eta_m + 1)) ** (1 / (eta_m + 1))
            w[i] = np.clip(w[i] + delta_q, 0, W_MAX)

    return Portfolio(w)


# ─────────────────────────────────────────────
# 6.  POPULATION INITIALISATION
# ─────────────────────────────────────────────

def random_portfolio(n_assets: int) -> Portfolio:
    w = np.random.dirichlet(np.ones(n_assets))
    return Portfolio(w)


def initialise_population(pop_size: int, n_assets: int) -> list:
    return [random_portfolio(n_assets) for _ in range(pop_size)]


def evaluate_population(pop: list, mu, cov, esg_scores, esg_max):
    for ind in pop:
        ind.evaluate(mu, cov, esg_scores, esg_max)


# ─────────────────────────────────────────────
# 7.  MAIN NSGA-II LOOP
# ─────────────────────────────────────────────

def nsga2(mu, cov, esg_scores,
          pop_size: int   = 200,
          n_gen: int      = 300,
          esg_max: float  = 20.0,
          eta_c: float    = 15.0,
          eta_m: float    = 20.0,
          seed: int       = 42,
          verbose: bool   = True):

    random.seed(seed)
    np.random.seed(seed)

    n_assets = len(mu)

    # ── Initialise
    population = initialise_population(pop_size, n_assets)
    evaluate_population(population, mu, cov, esg_scores, esg_max)
    fronts = non_dominated_sort(population)
    for f in fronts:
        crowding_distance_assignment(f)

    history = []   # track Pareto front size per generation

    for gen in range(n_gen):
        # ── Generate offspring
        offspring = []
        while len(offspring) < pop_size:
            p1 = tournament_selection(population)
            p2 = tournament_selection(population)
            c1, c2 = sbx_crossover(p1, p2, eta_c=eta_c)
            c1 = polynomial_mutation(c1, eta_m=eta_m)
            c2 = polynomial_mutation(c2, eta_m=eta_m)
            offspring.extend([c1, c2])
        offspring = offspring[:pop_size]

        evaluate_population(offspring, mu, cov, esg_scores, esg_max)

        # ── Combine & select next generation
        combined = population + offspring
        fronts   = non_dominated_sort(combined)
        for f in fronts:
            crowding_distance_assignment(f)

        new_pop = []
        for f in fronts:
            if len(new_pop) + len(f) <= pop_size:
                new_pop.extend(f)
            else:
                # Fill remaining slots by crowding distance (descending)
                f.sort(key=lambda p: -p.crowding_distance)
                new_pop.extend(f[:pop_size - len(new_pop)])
                break

        population = new_pop
        history.append(len(fronts[0]))

        if verbose and (gen + 1) % 50 == 0:
            pf = fronts[0]
            best_ret = max(-p.obj_return for p in pf)
            best_esg = min( p.obj_esg    for p in pf)   # lower totalEsg = less ESG risk = better
            min_risk = min( p.raw_risk   for p in pf)   # raw variance, no penalty
            print(f"  Gen {gen+1:>4d}/{n_gen} | Pareto size: {len(pf):>3d} | "
                  f"Best return: {best_ret:.2%} | Min vol: {np.sqrt(min_risk):.4f} | Min ESG risk: {best_esg:.2f}")

    # Final fronts
    fronts = non_dominated_sort(population)
    for f in fronts:
        crowding_distance_assignment(f)

    return fronts[0], history


# ─────────────────────────────────────────────
# 8.  RESULTS EXTRACTION & EXPORT
# ─────────────────────────────────────────────

def pareto_to_dataframe(pareto_front: list, tickers: list, mu, cov, esg_scores) -> pd.DataFrame:
    rows = []
    for i, p in enumerate(pareto_front):
        w = p.weights
        if p.raw_risk is None:
            raise RuntimeError(f"Portfolio {i} has raw_risk=None — ensure evaluate() was called on every portfolio before calling pareto_to_dataframe()")
        raw_risk = p.raw_risk   # w @ cov @ w stored during evaluate(), no penalty
        row = {
            "Portfolio_ID"   : i + 1,
            "Annual_Return"  : round(-p.obj_return, 6),
            "Annual_Variance": round(raw_risk, 6),
            "Annual_Volatility": round(np.sqrt(raw_risk), 6),
            "Sharpe_Ratio"   : round((-p.obj_return - RF_RATE) / np.sqrt(raw_risk), 4) if raw_risk > 0 else 0,
            "ESG_Score"      : round(p.obj_esg, 4),   # raw totalEsg; lower = less ESG risk
            "Pareto_Rank"    : p.rank,
        }
        for t, wt in zip(tickers, w):
            row[f"w_{t}"] = round(float(wt), 6)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("Annual_Return")
    return df


def export_results(pareto_df, esg_df_full,
                   out_path="nsga2_results.xlsx"):
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # Sheet 1: Pareto front summary
        summary_cols = ["Portfolio_ID","Annual_Return","Annual_Variance",
                        "Annual_Volatility","Sharpe_Ratio","ESG_Score","Pareto_Rank"]
        pareto_df[summary_cols].to_excel(writer, sheet_name="Pareto_Summary", index=False)

        # Sheet 2: Full weights
        pareto_df.to_excel(writer, sheet_name="Portfolio_Weights", index=False)

        # Sheet 3: Top portfolios by each criterion
        top_return = pareto_df.nlargest(10,  "Annual_Return")[summary_cols]
        top_sharpe = pareto_df.nlargest(10,  "Sharpe_Ratio")[summary_cols]
        top_esg    = pareto_df.nsmallest(10, "ESG_Score")[summary_cols]  # lower ESG risk = better
        top_return.to_excel(writer, sheet_name="Top10_Return",  index=False)
        top_sharpe.to_excel(writer, sheet_name="Top10_Sharpe",  index=False)
        top_esg.to_excel(writer,    sheet_name="Top10_ESG",     index=False)

        # Sheet 4: ESG reference
        esg_df_full.to_excel(writer, sheet_name="ESG_Reference", index=False)

    print(f"\n  Results saved to: {out_path}")


# ─────────────────────────────────────────────
# 9.  VISUALISATION
# ─────────────────────────────────────────────

def plot_pareto_front(pareto_front, history,
                      out_path="nsga2_plots.png"):

    returns   = np.array([-p.obj_return         for p in pareto_front])
    risks     = np.array([np.sqrt(p.raw_risk)   for p in pareto_front])  # raw volatility, no penalty
    esg_vals  = np.array([p.obj_esg             for p in pareto_front])  # lower = less ESG risk
    sharpes   = (returns - RF_RATE) / risks

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#0f1117")
    gs  = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35)

    ACCENT = "#00d4ff"
    GRID_C = "#2a2d3a"

    def style_ax(ax, title, xlabel, ylabel):
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="white", labelsize=9)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, color=GRID_C, linewidth=0.5, alpha=0.7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333649")

    # ── Plot 1: Return vs Risk (coloured by ESG risk — yellow=low/good, red=high/bad)
    ax1 = fig.add_subplot(gs[0, 0])
    sc  = ax1.scatter(risks * 100, returns * 100, c=esg_vals,
                      cmap="YlOrRd", s=40, alpha=0.85, edgecolors="none")
    plt.colorbar(sc, ax=ax1, label="ESG Risk Score (lower=better)").ax.yaxis.label.set_color("white")
    style_ax(ax1, "Pareto Front: Return vs Volatility\n(colour = ESG Risk Score)",
             "Annual Volatility (%)", "Annual Return (%)")

    # ── Plot 2: Return vs ESG Risk (coloured by Sharpe)
    ax2 = fig.add_subplot(gs[0, 1])
    sc2 = ax2.scatter(esg_vals, returns * 100, c=sharpes,
                      cmap="plasma", s=40, alpha=0.85, edgecolors="none")
    plt.colorbar(sc2, ax=ax2, label="Sharpe Ratio").ax.yaxis.label.set_color("white")
    style_ax(ax2, "Pareto Front: Return vs ESG Risk\n(colour = Sharpe Ratio)",
             "ESG Risk Score (lower=better)", "Annual Return (%)")

    # ── Plot 3: Volatility vs ESG Risk (coloured by Return)
    ax3 = fig.add_subplot(gs[0, 2])
    sc3 = ax3.scatter(esg_vals, risks * 100, c=returns * 100,
                      cmap="RdYlGn", s=40, alpha=0.85, edgecolors="none")
    plt.colorbar(sc3, ax=ax3, label="Annual Return (%)").ax.yaxis.label.set_color("white")
    style_ax(ax3, "Pareto Front: Volatility vs ESG Risk\n(colour = Return)",
             "ESG Risk Score (lower=better)", "Annual Volatility (%)")

    # ── Plot 4: Sharpe ratio distribution
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(sharpes, bins=25, color=ACCENT, edgecolor="#0f1117", alpha=0.85)
    ax4.axvline(sharpes.mean(), color="#ff6b6b", linewidth=1.5,
                linestyle="--", label=f"Mean: {sharpes.mean():.2f}")
    ax4.legend(facecolor="#1a1d27", labelcolor="white", fontsize=8)
    style_ax(ax4, "Sharpe Ratio Distribution\nacross Pareto Front",
             "Sharpe Ratio", "Count")

    # ── Plot 5: ESG distribution
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(esg_vals, bins=25, color="#4ecdc4", edgecolor="#0f1117", alpha=0.85)
    ax5.axvline(esg_vals.mean(), color="#ff6b6b", linewidth=1.5,
                linestyle="--", label=f"Mean: {esg_vals.mean():.2f}")
    ax5.legend(facecolor="#1a1d27", labelcolor="white", fontsize=8)
    style_ax(ax5, "ESG Risk Score Distribution\nacross Pareto Front",
             "ESG Risk Score (lower=better)", "Count")

    # ── Plot 6: Convergence (Pareto front size per generation)
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(history, color=ACCENT, linewidth=1.5, alpha=0.9)
    ax6.fill_between(range(len(history)), history, alpha=0.15, color=ACCENT)
    style_ax(ax6, "Convergence: Pareto Front Size\nper Generation",
             "Generation", "Pareto Front Size")

    fig.suptitle("NSGA-II Multi-Objective Portfolio Optimisation with ESG Constraints",
                 fontsize=14, fontweight="bold", color="white", y=0.98)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Plots saved to:   {out_path}")


def plot_top_portfolios(pareto_df, tickers,
                        out_path="nsga2_top_portfolios.png"):
    """Bar charts of top-3 portfolios by return, Sharpe, ESG showing top-10 holdings."""

    criteria = [
        ("Annual_Return", "Top Return",         "#00d4ff", "max"),
        ("Sharpe_Ratio",  "Best Sharpe",         "#4ecdc4", "max"),
        ("ESG_Score",     "Lowest ESG Risk",     "#a8e063", "min"),  # lower = less ESG risk
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#0f1117")

    weight_cols = [f"w_{t}" for t in tickers]

    for ax, (criterion, label, color, direction) in zip(axes, criteria):
        best = pareto_df.loc[(pareto_df[criterion].idxmin() if direction == "min"
                              else pareto_df[criterion].idxmax())]
        w    = best[weight_cols].values.astype(float)
        idx  = np.argsort(w)[::-1][:10]
        top_tickers = [tickers[i] for i in idx]
        top_w       = w[idx] * 100

        ax.set_facecolor("#1a1d27")
        bars = ax.barh(top_tickers[::-1], top_w[::-1], color=color, alpha=0.85)
        for bar, val in zip(bars, top_w[::-1]):
            ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}%", va="center", fontsize=8, color="white")

        ax.set_title(f"{label} Portfolio\n"
                     f"Ret: {best.Annual_Return:.1%} | "
                     f"Vol: {best.Annual_Volatility:.1%} | "
                     f"ESG: {best.ESG_Score:.1f}",
                     color="white", fontsize=10, fontweight="bold")
        ax.tick_params(colors="white", labelsize=9)
        ax.set_xlabel("Weight (%)", color="white", fontsize=9)
        ax.grid(axis="x", color="#2a2d3a", linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333649")

    fig.suptitle("Top Pareto Portfolios — Top 10 Holdings",
                 fontsize=13, fontweight="bold", color="white", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Holdings plot saved: {out_path}")
