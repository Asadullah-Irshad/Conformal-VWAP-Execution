"""Regenerate the paper's figures from the simulator.

Every simulator-based figure in the article (Figs 1-10) is produced here from a
fixed seed, so the qualitative story is fully reproducible. The two real-data
figures (Figs 11-12) live in ``real_data.py`` because they require downloading
intraday market data.

Run via ``python Scripts/make_figures.py`` which calls :func:`make_all`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .benchmarks import almgren_chriss_schedule, twap_schedule, vwap_tracking_schedule
from .conformal import ConformalReturnPredictor
from .experiments import ExperimentConfig, _make_paths
from .features import build_xy, stack_sessions
from .policies import forecast_tilt_schedule, kappa_grid, per_interval_forecasts
from .simulator import execute_schedule

_C = {
    "twap": "#555555", "vwap": "#2ca02c", "ac": "#d62728",
    "greedy": "#9467bd", "gate": "#1f77b4", "gate2": "#8c564b",
}


# --------------------------------------------------------------------------- #
#  Shared context: train / calibrate / test once, reuse for every figure       #
# --------------------------------------------------------------------------- #
class FigureContext:
    def __init__(self, cfg: ExperimentConfig | None = None):
        self.cfg = cfg or ExperimentConfig()
        c = self.cfg
        p = c.params
        s0 = c.seed
        self.train = _make_paths(p, range(s0, s0 + c.n_train))
        s1 = s0 + c.n_train
        self.calib = _make_paths(p, range(s1, s1 + c.n_calib))
        s2 = s1 + c.n_calib
        self.test = _make_paths(p, range(s2, s2 + c.n_test))
        self.pred = ConformalReturnPredictor(alpha=c.alpha)
        self.pred.fit(self.train)
        self.pred.calibrate(self.calib)
        self.Q = c.parent_qty
        self.direction = c.direction
        self.beta = c.beta

    # --- helpers -------------------------------------------------------------
    def slip_array(self, policy_fn) -> np.ndarray:
        out = []
        for pth in self.test:
            res = execute_schedule(pth, policy_fn(pth), direction=self.direction)
            out.append(res["slippage_vs_vwap_bps"])
        return np.asarray(out)

    def baseline_arrays(self) -> dict:
        Q, c = self.Q, self.cfg
        n = c.params.n_intervals

        def immediate(_p):
            s = np.zeros(n); s[0] = Q; return s

        d = {
            "Immediate": self.slip_array(immediate),
            "TWAP": self.slip_array(lambda _p: twap_schedule(Q, n)),
            "Almgren-Chriss": self.slip_array(
                lambda _p: almgren_chriss_schedule(Q, n, c.ac_risk_aversion)),
            "VWAP-tracking": self.slip_array(lambda _p: vwap_tracking_schedule(Q, c.params)),
            "Forecast-greedy": self.slip_array(
                lambda pth: forecast_tilt_schedule(Q, pth, self.pred, self.beta,
                                                   self.direction, kappa=0.0)),
        }
        for k in c.kappa_report:
            d[f"Conformal-gated (K={k})"] = self.slip_array(
                lambda pth, _k=k: forecast_tilt_schedule(Q, pth, self.pred, self.beta,
                                                         self.direction, kappa=_k))
        return d

    def frontier(self) -> list[dict]:
        Q = self.Q
        rows = []
        for k in kappa_grid(n=14, k_max=4.0):
            arr = self.slip_array(
                lambda pth, _k=k: forecast_tilt_schedule(Q, pth, self.pred, self.beta,
                                                         self.direction, kappa=_k))
            rows.append({"kappa": float(k), "mean": float(arr.mean()), "std": float(arr.std())})
        return rows

    def test_forecasts(self):
        X, y, scale = stack_sessions(self.test)
        yhat, half = self.pred.predict(X, scale)
        return y, yhat, half, scale


# --------------------------------------------------------------------------- #
#  Figure 1 - schematic pipeline                                               #
# --------------------------------------------------------------------------- #
def fig01_pipeline(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 30)

    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc=fc, ec="black", lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=14, lw=1.4, color="#333333"))

    box(1, 12, 15, 8, "Market state\n(causal features)", "#eaf2fb")
    box(20, 12, 17, 8, "Gradient-boosted\nreturn forecast", "#eaf2fb")
    box(41, 12, 20, 8, "Normalised split-\nconformal interval\n(half-width ∝ local vol)", "#eaf2fb")
    box(65, 12, 14, 8, "Gate:\n|f| > κ · half-width?", "#fdf0e6")
    box(85, 21, 14, 7, "Tilt VWAP\nschedule (act)", "#e8f6ec")
    box(85, 3, 14, 7, "Track VWAP\n(abstain)", "#f6e8e8")
    arrow(16, 16, 20, 16); arrow(37, 16, 41, 16); arrow(61, 16, 65, 16)
    arrow(79, 17, 85, 24); arrow(79, 15, 85, 8)
    ax.text(82, 25, "yes", fontsize=8, color="#2ca02c")
    ax.text(82, 9.5, "no", fontsize=8, color="#d62728")
    ax.set_title("Figure 1. The uncertainty-aware AI pipeline", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------- #
#  Figure 2 - sample session with conformal band                               #
# --------------------------------------------------------------------------- #
def fig02_sample_session(ctx: FigureContext, path: Path) -> None:
    p = ctx.test[0]
    X, y, scale = build_xy(p)
    yhat, half = ctx.pred.predict(X, scale)
    T = p.n_intervals
    t = np.arange(T)
    mid = p.mid[:-1]
    upper = mid * np.exp(half); lower = mid * np.exp(-half)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True,
                                 gridspec_kw={"height_ratios": [2.4, 1]})
    a1.plot(t, mid, color="#1f77b4", lw=1.8, label="Mid price")
    a1.axhline(p.market_vwap, color="#2ca02c", ls="--", lw=1.3, label="Day VWAP")
    a1.fill_between(t, lower, upper, color="#1f77b4", alpha=0.15,
                    label="90% conformal band")
    a1.set_ylabel("Price"); a1.legend(fontsize=8, loc="best")
    a1.set_title("Figure 2. A sample intraday session with the 90% conformal band")
    a2.plot(t, half * 1e4, color="#d62728", lw=1.5)
    a2.set_ylabel("Half-width\n(bps)"); a2.set_xlabel("Interval (5-min)")
    a2.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------- #
#  Figure 3 - temporary impact vs participation                                #
# --------------------------------------------------------------------------- #
def fig03_impact(ctx: FigureContext, path: Path) -> None:
    p = ctx.cfg.params
    part = np.linspace(0, 0.5, 200)
    temp = p.eta * part          # bps at 100% participation = eta
    perm = p.gamma * part
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(part * 100, temp, color="#d62728", lw=2, label=f"Temporary (η={p.eta} bps)")
    ax.plot(part * 100, perm, color="#1f77b4", lw=2, ls="--", label=f"Permanent (γ={p.gamma} bps)")
    ax.axhline(p.half_spread_bps, color="#888888", ls=":", lw=1.3,
               label=f"Spread floor ({p.half_spread_bps} bps)")
    ax.set_xlabel("Participation rate (% of interval volume)")
    ax.set_ylabel("Impact (bps)")
    ax.set_title("Figure 3. Impact as a function of participation rate")
    ax.grid(True, alpha=0.25); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------- #
#  Figure 4 - nonconformity score distribution                                 #
# --------------------------------------------------------------------------- #
def fig04_nonconformity(ctx: FigureContext, path: Path) -> None:
    X, y, scale = stack_sessions(ctx.calib)
    yhat = ctx.pred.model.predict(X)
    scores = np.abs(y - yhat) / np.maximum(scale, 1e-8)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.hist(scores, bins=60, color="#1f77b4", alpha=0.75, edgecolor="white")
    ax.axvline(ctx.pred.q_, color="#d62728", lw=2,
               label=f"90% empirical quantile q = {ctx.pred.q_:.2f}")
    ax.set_xlabel("Normalised nonconformity score  |y - ŷ| / σ̂(x)")
    ax.set_ylabel("Count")
    ax.set_title("Figure 4. Distribution of normalised nonconformity scores (calibration)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------- #
#  Figure 5 - coverage reliability curve                                       #
# --------------------------------------------------------------------------- #
def fig05_coverage(ctx: FigureContext, path: Path) -> None:
    Xc, yc, sc = stack_sessions(ctx.calib)
    Xt, yt, st = stack_sessions(ctx.test)
    sc_scores = np.abs(yc - ctx.pred.model.predict(Xc)) / np.maximum(sc, 1e-8)
    st_scores = np.abs(yt - ctx.pred.model.predict(Xt)) / np.maximum(st, 1e-8)
    nominal = np.linspace(0.50, 0.99, 25)
    emp = []
    for lv in nominal:
        q = np.quantile(sc_scores, lv, method="higher")
        emp.append(float(np.mean(st_scores <= q)))
    fig, ax = plt.subplots(figsize=(6, 5.6))
    ax.plot([0.5, 1], [0.5, 1], color="#888888", ls="--", lw=1.2, label="Ideal (y = x)")
    ax.plot(nominal, emp, "-o", color="#1f77b4", ms=4, lw=1.8, label="Empirical")
    ax.set_xlabel("Nominal coverage 1 - α"); ax.set_ylabel("Empirical coverage")
    ax.set_title("Figure 5. Conformal coverage reliability (simulated)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------- #
#  Figure 6 - gate activation vs threshold                                     #
# --------------------------------------------------------------------------- #
def fig06_gate_activation(ctx: FigureContext, path: Path) -> None:
    _, yhat, half, _ = ctx.test_forecasts()
    ks = np.linspace(0, 4, 60)
    frac = [float(np.mean(np.abs(yhat) > k * half)) for k in ks]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ks, frac, color="#1f77b4", lw=2)
    for k in ctx.cfg.kappa_report:
        f = float(np.mean(np.abs(yhat) > k * half))
        ax.scatter([k], [f], color="#d62728", zorder=5)
        ax.annotate(f"κ={k}", (k, f), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlabel("Gate threshold κ"); ax.set_ylabel("Fraction of intervals gate fires")
    ax.set_title("Figure 6. Gate activation versus threshold")
    ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------- #
#  Figure 7 - cost-risk frontier                                               #
# --------------------------------------------------------------------------- #
def fig07_frontier(ctx: FigureContext, base: dict, front: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    fr = sorted(front, key=lambda r: r["std"])
    ax.plot([r["std"] for r in fr], [r["mean"] for r in fr], "-o", color=_C["gate"],
            lw=2, ms=4, label="Conformal gate (κ sweep)", zorder=3)
    marks = {"TWAP": ("s", _C["twap"]), "VWAP-tracking": ("D", _C["vwap"]),
             "Almgren-Chriss": ("^", _C["ac"]), "Forecast-greedy": ("*", _C["greedy"])}
    for name, (mk, col) in marks.items():
        a = base[name]
        ax.scatter([a.std()], [a.mean()], marker=mk, s=150, color=col,
                   edgecolor="black", lw=0.6, zorder=5, label=name)
    ax.set_xlabel("Cost variability - std of slippage vs VWAP (bps)")
    ax.set_ylabel("Mean cost - slippage vs VWAP (bps)")
    ax.set_title("Figure 7. Cost-risk frontier (Immediate omitted for scale)")
    ax.grid(True, alpha=0.25); ax.legend(fontsize=9, loc="best")
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------- #
#  Figure 8 - mean slippage with bootstrap CIs                                 #
# --------------------------------------------------------------------------- #
def fig08_bootstrap(ctx: FigureContext, base: dict, path: Path) -> None:
    rng = np.random.default_rng(0)
    names = [n for n in base if n != "Immediate"]
    means, los, his = [], [], []
    for n in names:
        a = base[n]
        boot = [rng.choice(a, size=len(a), replace=True).mean() for _ in range(2000)]
        means.append(a.mean()); los.append(np.percentile(boot, 2.5)); his.append(np.percentile(boot, 97.5))
    y = np.arange(len(names))
    means = np.array(means); los = np.array(los); his = np.array(his)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(means, y, xerr=[means - los, his - means], fmt="o", color="#1f77b4",
                capsize=4, lw=1.5, ms=7)
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_xlabel("Mean slippage vs VWAP (bps), 95% bootstrap CI")
    ax.set_title("Figure 8. Mean slippage with 95% bootstrap CIs (250 seeds)")
    ax.grid(True, axis="x", alpha=0.25); ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------- #
#  Figure 9 - execution trajectories on a sample session                       #
# --------------------------------------------------------------------------- #
def fig09_trajectories(ctx: FigureContext, path: Path) -> None:
    p = ctx.test[0]; Q = ctx.Q; n = p.n_intervals
    scheds = {
        "VWAP-tracking": vwap_tracking_schedule(Q, ctx.cfg.params),
        "Forecast-greedy": forecast_tilt_schedule(Q, p, ctx.pred, ctx.beta, ctx.direction, kappa=0.0),
        "Conformal-gated (κ=1.0)": forecast_tilt_schedule(Q, p, ctx.pred, ctx.beta, ctx.direction, kappa=1.0),
        "TWAP": twap_schedule(Q, n),
    }
    cols = {"VWAP-tracking": _C["vwap"], "Forecast-greedy": _C["greedy"],
            "Conformal-gated (κ=1.0)": _C["gate"], "TWAP": _C["twap"]}
    fig, ax = plt.subplots(figsize=(8, 5))
    t = np.arange(n + 1)
    for name, s in scheds.items():
        remaining = Q - np.concatenate([[0], np.cumsum(s)])
        ax.plot(t, remaining / Q * 100, lw=2, color=cols[name], label=name)
    ax.set_xlabel("Interval (5-min)"); ax.set_ylabel("Remaining inventory (%)")
    ax.set_title("Figure 9. Execution trajectories on a sample session")
    ax.grid(True, alpha=0.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------- #
#  Figure 10 - the conformal gate as a cost-risk dial                          #
# --------------------------------------------------------------------------- #
def fig10_dial(ctx: FigureContext, front: list[dict], path: Path) -> None:
    ks = np.array([r["kappa"] for r in front])
    xs = np.array([r["std"] for r in front])
    ys = np.array([r["mean"] for r in front])
    order = np.argsort(ks)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(xs[order], ys[order], color="#bbbbbb", lw=1.5, zorder=1)
    sc = ax.scatter(xs, ys, c=ks, cmap="viridis", s=80, edgecolor="black", lw=0.5, zorder=3)
    cb = fig.colorbar(sc, ax=ax); cb.set_label("Gate threshold κ")
    ax.set_xlabel("Cost variability - std of slippage (bps)")
    ax.set_ylabel("Mean slippage vs VWAP (bps)")
    ax.set_title("Figure 10. The conformal gate as a cost-risk dial")
    ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------- #
#  Orchestration                                                               #
# --------------------------------------------------------------------------- #
def make_all(out_dir: str | Path = "Figures") -> list[str]:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    ctx = FigureContext()
    base = ctx.baseline_arrays()
    front = ctx.frontier()

    fig01_pipeline(out / "figure01_uncertainty_aware_ai_pipeline.png")
    fig02_sample_session(ctx, out / "figure02_sample_intraday_session.png")
    fig03_impact(ctx, out / "figure03_temporary_impact_vs_participation.png")
    fig04_nonconformity(ctx, out / "figure04_nonconformity_score_distribution.png")
    fig05_coverage(ctx, out / "figure05_conformal_coverage_reliability.png")
    fig06_gate_activation(ctx, out / "figure06_gate_activation_vs_threshold.png")
    fig07_frontier(ctx, base, front, out / "figure07_cost_risk_frontier.png")
    fig08_bootstrap(ctx, base, out / "figure08_mean_slippage_bootstrap_ci.png")
    fig09_trajectories(ctx, out / "figure09_execution_trajectories_sample_session.png")
    fig10_dial(ctx, front, out / "figure10_conformal_gate_cost_risk_dial.png")

    written = sorted(str(p.name) for p in out.glob("figure*.png"))
    return written
