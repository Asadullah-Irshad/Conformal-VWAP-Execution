# Figures

All 12 figures from the paper.

| File | Paper figure | Source |
|---|---|---|
| `figure01_uncertainty_aware_ai_pipeline.png` | Fig. 1: pipeline schematic | code (`make_figures.py`) |
| `figure02_sample_intraday_session.png` | Fig. 2: sample session + conformal band | code |
| `figure03_temporary_impact_vs_participation.png` | Fig. 3: impact vs participation | code |
| `figure04_nonconformity_score_distribution.png` | Fig. 4: nonconformity scores | code |
| `figure05_conformal_coverage_reliability.png` | Fig. 5: coverage reliability | code |
| `figure06_gate_activation_vs_threshold.png` | Fig. 6: gate activation | code |
| `figure07_cost_risk_frontier.png` | Fig. 7: cost–risk frontier | code |
| `figure08_mean_slippage_bootstrap_ci.png` | Fig. 8: bootstrap CIs | code |
| `figure09_execution_trajectories_sample_session.png` | Fig. 9: execution trajectories | code |
| `figure10_conformal_gate_cost_risk_dial.png` | Fig. 10: cost–risk dial | code |
| `figure11_conformal_coverage_real_data.png` | Fig. 11: real-data coverage | published (needs live data to regenerate) |
| `figure12_cost_risk_frontier_real_data.png` | Fig. 12: real-data frontier | published (needs live data to regenerate) |

**Regenerating:** `python Scripts/make_figures.py` rebuilds Figs 1–10 from a fixed
seed (overwriting these with fresh reproductions). Figs 11–12 require live market
data; see `Scripts/run_real_data.py` and the real-data note in the top-level README.
The versions committed here are the ones from the published paper.
