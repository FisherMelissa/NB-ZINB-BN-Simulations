"""Utility functions for the ZINB simulation and Streamlit visualisations."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP

    STATSMODELS_AVAILABLE = True
except ImportError:  # pragma: no cover
    STATSMODELS_AVAILABLE = False
    sm = None  # type: ignore
    ZeroInflatedNegativeBinomialP = None  # type: ignore

try:  # pragma: no cover
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    MATPLOTLIB_AVAILABLE = False
    plt = None  # type: ignore


def create_synthetic_dataset(n_months: int = 24, seed: Optional[int] = None) -> pd.DataFrame:
    """Create a synthetic dataset with zero inflation for monthly incident counts."""
    rng = np.random.default_rng(seed)

    dates = pd.date_range(start="2022-01-01", periods=n_months, freq="ME")
    data = pd.DataFrame(
        {
            "date": dates,
            "month": np.arange(1, n_months + 1),
            "intervention": (dates.month >= 7).astype(int),
            "seasonal_factor": np.sin(2 * np.pi * np.arange(n_months) / 12) * 0.5 + 1.0,
            "lag_count": 0.0,
        }
    )

    true_counts = np.zeros(n_months)
    alpha = 1.0

    for i in range(n_months):
        base_mean = 3.0 * data.loc[i, "seasonal_factor"] * np.exp(-0.5 * data.loc[i, "intervention"])
        if i > 0:
            base_mean += 0.15 * true_counts[i - 1]
        n_param = 1.0 / alpha
        p_param = n_param / (n_param + base_mean)
        true_counts[i] = rng.negative_binomial(n_param, p_param)

    data["true_count"] = true_counts.astype(int)

    zero_inflation_prob = 0.2 + 0.15 * np.exp(-0.1 * data["month"])
    zero_inflation_prob = np.clip(zero_inflation_prob, 0.05, 0.8)
    zero_inflation_prob[data["intervention"] == 1] *= 1.2
    zero_inflation_prob = np.clip(zero_inflation_prob, 0.05, 0.85)

    structural_zero = rng.random(n_months) < zero_inflation_prob
    data["incident_count"] = np.where(structural_zero, 0, data["true_count"])

    for i in range(1, n_months):
        data.loc[i, "lag_count"] = data.loc[i - 1, "incident_count"]

    return data


def fit_zinb_model(data: pd.DataFrame):
    """Fit a Zero-Inflated Negative Binomial model to the data."""
    if not STATSMODELS_AVAILABLE:
        raise ImportError(
            "statsmodels is required for the ZINB simulation. Install with `pip install statsmodels>=0.14`."
        )

    exog_vars = ["intervention", "seasonal_factor", "lag_count"]
    exog_infl_vars = ["intervention", "month"]

    exog = sm.add_constant(data[exog_vars])
    exog_infl = sm.add_constant(data[exog_infl_vars])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ZeroInflatedNegativeBinomialP(
            data["incident_count"],
            exog,
            exog_infl=exog_infl,
            inflation="logit",
        )
        results = model.fit(disp=0)

    return results, exog, exog_infl


def simulate_forward(
    model,
    exog_base: pd.DataFrame,
    n_runs: int = 1000,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Monte Carlo forward simulation using the fitted ZINB model."""
    rng = np.random.default_rng(seed)
    n_runs = max(1, n_runs)

    n_periods = exog_base.shape[0]
    simulations = np.zeros((n_runs, n_periods))

    params = model.params
    n_count_params = exog_base.shape[1]
    params_count = params[:n_count_params]
    params_infl = params[n_count_params:-1]
    alpha = max(float(params[-1]), 0.1)

    for run in range(n_runs):
        for t in range(n_periods):
            x_count = exog_base.iloc[t].to_numpy()
            eta_count = np.clip(np.dot(x_count, params_count), -10, 10)
            mu_count = np.exp(eta_count)

            infl_x = np.array([1.0, exog_base.iloc[t]["intervention"], t + 1.0])
            eta_infl = np.dot(infl_x, params_infl)
            p_infl = 1.0 / (1.0 + np.exp(-eta_infl))

            if rng.random() < p_infl:
                simulations[run, t] = 0.0
            else:
                n_param = 1.0 / alpha
                p_param = n_param / (n_param + mu_count)
                p_param = np.clip(p_param, 1e-3, 0.99)
                simulations[run, t] = rng.negative_binomial(n_param, p_param)

    return simulations


def create_scenarios(
    data: pd.DataFrame,
    intervention_months: Optional[Iterable[int]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create intervention and no-intervention scenario datasets for comparison."""
    if intervention_months is None:
        intervention_months = list(range(13, min(len(data) + 1, 25)))

    scenario_data = data.copy()
    control_data = data.copy()

    scenario_data["intervention"] = scenario_data["month"].isin(list(intervention_months)).astype(int)
    control_data["intervention"] = 0

    control_data.loc[0, "lag_count"] = 0
    for i in range(1, len(control_data)):
        control_data.loc[i, "lag_count"] = control_data.loc[i - 1, "incident_count"]

    return scenario_data, control_data


def prepare_zinb_intervention_designs(
    data: pd.DataFrame,
    auth_parenting: bool = False,
    peer_shift: float = 0.0,
    zero_floor: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare design matrices for baseline vs intervention adjustments used in the dashboard."""
    if not STATSMODELS_AVAILABLE:
        raise ImportError(
            "statsmodels is required for the ZINB simulation. Install with `pip install statsmodels>=0.14`."
        )

    scenario_data = data.copy()
    if auth_parenting:
        scenario_data["intervention"] = 0

    if peer_shift != 0.0:
        shifted = np.clip(scenario_data["lag_count"] + peer_shift, zero_floor, None)
        scenario_data["lag_count"] = shifted

    exog_vars = ["intervention", "seasonal_factor", "lag_count"]
    infl_vars = ["intervention", "month"]

    baseline_exog = sm.add_constant(data[exog_vars])
    baseline_infl = sm.add_constant(data[infl_vars])

    scenario_exog = sm.add_constant(scenario_data[exog_vars])
    scenario_infl = sm.add_constant(scenario_data[infl_vars])

    return baseline_exog, baseline_infl, scenario_exog, scenario_infl, scenario_data


def summarize_zinb_effects(
    baseline_sim: np.ndarray,
    scenario_sim: np.ndarray,
    months_mask: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    """Summarise totals and deltas between two simulation arrays."""
    if months_mask is None:
        months_mask = np.ones(baseline_sim.shape[1], dtype=bool)

    def _totals(sim: np.ndarray) -> Dict[str, float]:
        selected = sim[:, months_mask].sum(axis=1)
        return {
            "mean": float(np.mean(selected)),
            "p10": float(np.percentile(selected, 10)),
            "p90": float(np.percentile(selected, 90)),
        }

    base_metrics = _totals(baseline_sim)
    scenario_metrics = _totals(scenario_sim)

    delta = scenario_metrics["mean"] - base_metrics["mean"]

    return {
        "baseline": base_metrics,
        "scenario": scenario_metrics,
        "delta": float(delta),
    }


def plot_results(
    data: pd.DataFrame,
    simulations_intervention: np.ndarray,
    simulations_no_intervention: np.ndarray,
    output_dir: str = "plots",
) -> None:
    """Generate matplotlib plots; used by the CLI script."""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Skipping plots.")
        return

    Path(output_dir).mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    months = data["month"].to_numpy()

    ax1.plot(months, data["incident_count"], "ko-", label="Actual", linewidth=2)

    intervention_mean = simulations_intervention.mean(axis=0)
    intervention_pct = np.percentile(simulations_intervention, [2.5, 97.5], axis=0)
    ax1.plot(months, intervention_mean, "b-", label="With intervention", linewidth=2)
    ax1.fill_between(months, intervention_pct[0], intervention_pct[1], alpha=0.3, color="blue")

    control_mean = simulations_no_intervention.mean(axis=0)
    control_pct = np.percentile(simulations_no_intervention, [2.5, 97.5], axis=0)
    ax1.plot(months, control_mean, "r--", label="Without intervention", linewidth=2)
    ax1.fill_between(months, control_pct[0], control_pct[1], alpha=0.3, color="red")

    ax1.set_xlabel("Month")
    ax1.set_ylabel("Incident Count")
    ax1.set_title("ZINB Model: Actual vs Simulated Incident Counts")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    months_mask = months >= 13
    cumulative_intervention = np.cumsum(simulations_intervention[:, months_mask], axis=1)
    cumulative_control = np.cumsum(simulations_no_intervention[:, months_mask], axis=1)

    ax2.plot(range(13, 13 + cumulative_intervention.shape[1]), cumulative_intervention.mean(axis=0), "b-", label="With intervention", linewidth=2)
    ax2.plot(range(13, 13 + cumulative_control.shape[1]), cumulative_control.mean(axis=0), "r--", label="Without intervention", linewidth=2)

    intervention_pct_cum = np.percentile(cumulative_intervention, [2.5, 97.5], axis=0)
    control_pct_cum = np.percentile(cumulative_control, [2.5, 97.5], axis=0)

    ax2.fill_between(range(13, 13 + cumulative_intervention.shape[1]), intervention_pct_cum[0], intervention_pct_cum[1], alpha=0.3, color="blue")
    ax2.fill_between(range(13, 13 + cumulative_control.shape[1]), control_pct_cum[0], control_pct_cum[1], alpha=0.3, color="red")

    ax2.set_xlabel("Month")
    ax2.set_ylabel("Cumulative Incidents")
    ax2.set_title("Cumulative Incident Counts (Months 13-24)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / "zinb_simulation_results.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plots saved to {output_path}")
    plt.close(fig)
