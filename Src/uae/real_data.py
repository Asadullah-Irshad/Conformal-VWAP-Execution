"""Real-data validation (paper Section 6): Figs 11-12 and Table 2.

This reproduces the *methodology* of the paper's real-data section: a normalised
split-conformal predictor and the conformal execution gate applied to intraday
equity data, on data downloaded at run time via yfinance.

IMPORTANT / HONESTY NOTE
------------------------
* Requires network access to Yahoo Finance (``pip install yfinance``).
* Free intraday history is limited (roughly the last ~60 days of 5-minute bars),
  so this script CANNOT reproduce the paper's exact 450-session numbers; treat
  the output as an indicative, up-to-date re-run of the same pipeline, not a
  bit-for-bit reproduction of Table 2 / Figs 11-12 in the article. The published
  figures and table are archived in ``Figures/`` and ``Tables/``.

Run:  python Scripts/run_real_data.py --tickers AAPL MSFT ... --days 55
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .conformal import ConformalReturnPredictor
from .features import build_xy
from .simulator import MarketParams, MarketPath, execute_schedule

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM", "V", "JNJ",
    "WMT", "PG", "MA", "HD", "XOM", "BAC", "KO", "PFE", "CSCO", "PEP",
    "COST", "ADBE", "CRM", "NFLX", "INTC", "AMD", "QCOM", "TXN", "ORCL", "DIS",
]


def _sessions_from_prices(df: pd.DataFrame, ticker: str) -> list[MarketPath]:
    """Turn one ticker's intraday bars into per-day MarketPath sessions."""
    out: list[MarketPath] = []
    df = df.dropna()
    if df.empty:
        return out
    df = df.copy()
    df["day"] = df.index.date
    for _, g in df.groupby("day"):
        close = g["Close"].to_numpy(dtype=float).ravel()
        vol = g["Volume"].to_numpy(dtype=float).ravel()
        if len(close) < 40 or vol.sum() <= 0:
            continue
        T = len(close) - 1
        returns = np.diff(np.log(close))
        mid = close.copy()
        # rolling realised-vol proxy as the conformal normaliser scale
        rv = np.array([np.std(returns[max(0, t - 10):t]) if t >= 2
                       else np.std(returns) for t in range(T)])
        interval_price = mid[:-1]
        market_vwap = float(np.sum(vol[:T] * interval_price) / np.sum(vol[:T]))
        params = MarketParams(n_intervals=T, start_price=float(mid[0]))
        out.append(MarketPath(
            params=params, mid=mid, returns=returns, vol=np.maximum(rv, 1e-6),
            volume=np.maximum(vol[:T], 1.0), arrival_price=float(mid[0]),
            market_vwap=market_vwap,
        ))
    return out


def _empirical_volume_curve(paths: list[MarketPath], T: int) -> np.ndarray:
    """Average normalised intraday volume curve over sessions of length T."""
    curves = [p.volume / p.volume.sum() for p in paths if p.n_intervals == T]
    if not curves:
        return np.full(T, 1.0 / T)
    return np.mean(curves, axis=0)


def _tilt_schedule(Q, path, predictor, curve, beta, direction, kappa):
    """VWAP-tracking base (empirical curve) tilted by the gated forecast."""
    X, _, scale = build_xy(path)
    yhat, half = predictor.predict(X, scale)
    sign = 1.0 if direction.upper() == "BUY" else -1.0
    raw = np.clip(-beta * sign * yhat, -1.5, 1.5)
    if kappa and kappa > 0.0:
        raw = np.where(np.abs(yhat) > kappa * half, raw, 0.0)
    base = curve if len(curve) == path.n_intervals else np.full(path.n_intervals, 1.0 / path.n_intervals)
    w = base * np.exp(raw)
    return Q * w / w.sum()


@dataclass
class RealDataConfig:
    parent_qty: float = 50_000.0
    direction: str = "SELL"
    alpha: float = 0.10
    beta: float = 260.0
    kappa_report: tuple = (0.5, 1.0)
    train_frac: float = 0.5
    calib_frac: float = 0.25  # remaining 0.25 is test


def run_real_data(tickers=None, days: int = 55, out_results="Results",
                  out_figures="Figures", cfg: RealDataConfig | None = None) -> dict:
    import yfinance as yf  # imported here so the rest of the package needs no network
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = cfg or RealDataConfig()
    tickers = tickers or DEFAULT_TICKERS
    outR = Path(out_results); outR.mkdir(parents=True, exist_ok=True)
    outF = Path(out_figures); outF.mkdir(parents=True, exist_ok=True)

    # --- download + build sessions -----------------------------------------
    sessions: list[MarketPath] = []
    for tk in tickers:
        try:
            df = yf.download(tk, period=f"{days}d", interval="5m", progress=False, auto_adjust=False)
        except Exception as e:  # noqa: BLE001
            print(f"  {tk}: download failed ({e})"); continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        sessions += _sessions_from_prices(df, tk)
    if len(sessions) < 20:
        raise SystemExit(f"Only {len(sessions)} sessions retrieved; need network access and more history.")
    print(f"Built {len(sessions)} real intraday sessions from {len(tickers)} tickers.")

    # --- time-ordered walk-forward split -----------------------------------
    n = len(sessions)
    i1 = int(n * cfg.train_frac); i2 = int(n * (cfg.train_frac + cfg.calib_frac))
    train, calib, test = sessions[:i1], sessions[i1:i2], sessions[i2:]

    predictor = ConformalReturnPredictor(alpha=cfg.alpha)
    predictor.fit(train); predictor.calibrate(calib)
    cov = predictor.evaluate_coverage(test)

    # --- execution comparison (Table 2) ------------------------------------
    Q, direction = cfg.parent_qty, cfg.direction

    def slip(policy):
        arr = [execute_schedule(p, policy(p), direction=direction)["slippage_vs_vwap_bps"] for p in test]
        return np.asarray(arr)

    def twap(p):
        return np.full(p.n_intervals, Q / p.n_intervals)

    def vwap_track(p):
        c = _empirical_volume_curve(train, p.n_intervals)
        return Q * c

    rows = []
    rows.append(("TWAP", slip(twap)))
    rows.append(("VWAP-tracking", slip(vwap_track)))
    rows.append(("Forecast-greedy", slip(
        lambda p: _tilt_schedule(Q, p, predictor, _empirical_volume_curve(train, p.n_intervals),
                                 cfg.beta, direction, 0.0))))
    for k in cfg.kappa_report:
        rows.append((f"Conformal-gated (kappa={k})", slip(
            lambda p, _k=k: _tilt_schedule(Q, p, predictor, _empirical_volume_curve(train, p.n_intervals),
                                           cfg.beta, direction, _k))))

    table = pd.DataFrame([{"method": n_, "slippage_mean_bps": float(a.mean()),
                           "slippage_std_bps": float(a.std())} for n_, a in rows])
    table.to_csv(outR / "realdata_results_table.csv", index=False)

    coverage = {"nominal_coverage": 1 - cfg.alpha, "empirical_coverage": cov.coverage,
                "n_test_points": cov.n_test, "n_sessions": n}
    (outR / "realdata_coverage.json").write_text(json.dumps(coverage, indent=2))

    # --- Figure 11: real-data coverage -------------------------------------
    fig, ax = plt.subplots(figsize=(5, 5.4))
    ax.bar(["nominal", "empirical"], [coverage["nominal_coverage"], coverage["empirical_coverage"]],
           color=["#bbbbbb", "#1f77b4"], edgecolor="black")
    ax.axhline(coverage["nominal_coverage"], color="#d62728", ls="--")
    ax.set_ylim(0.8, 1.0); ax.set_ylabel("coverage")
    ax.set_title(f"Figure 11. Conformal coverage on real data\nempirical {cov.coverage:.3f} vs nominal "
                 f"{coverage['nominal_coverage']:.2f}")
    fig.tight_layout(); fig.savefig(outF / "figure11_conformal_coverage_real_data.png", dpi=150); plt.close(fig)

    # --- Figure 12: real-data cost-risk frontier ---------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for n_, a in rows:
        ax.scatter([a.std()], [a.mean()], s=120, edgecolor="black", label=n_)
    ax.set_xlabel("Cost variability - std of slippage (bps)")
    ax.set_ylabel("Mean slippage vs VWAP (bps)")
    ax.set_title("Figure 12. Cost-risk frontier on real intraday data")
    ax.grid(True, alpha=0.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(outF / "figure12_cost_risk_frontier_real_data.png", dpi=150); plt.close(fig)

    print("Wrote realdata_results_table.csv, realdata_coverage.json, figure11_*, figure12_*")
    print(table.to_string(index=False))
    return {"table": table, "coverage": coverage}
