#!/usr/bin/env python3
"""Zero-Inflated Negative Binomial (ZINB) simulation CLI wrapper."""

from __future__ import annotations

import argparse
import sys

import statsmodels.api as sm

from app.simlib.zinb import (
    MATPLOTLIB_AVAILABLE,
    STATSMODELS_AVAILABLE,
    create_scenarios,
    create_synthetic_dataset,
    fit_zinb_model,
    plot_results,
    simulate_forward,
    summarize_zinb_effects,
)


def print_results(model, data, simulations_intervention, simulations_no_intervention) -> None:
    print("=" * 60)
    print("ZINB MODEL RESULTS")
    print("=" * 60)

    print("\nModel Summary:")
    print(model.summary())

    exog = model.model.exog
    exog_infl = model.model.exog_infl

    predictions = model.predict(exog, exog_infl=exog_infl)
    print("\n" + "=" * 60)
    print("IN-SAMPLE FIT")
    print("=" * 60)
    for i in range(min(12, len(data))):
        print(
            f"Month {i+1:2d}: Expected={predictions[i]:.2f}, Actual={data['incident_count'].iloc[i]}"
        )

    print("\n" + "=" * 60)
    print("SCENARIO COMPARISON (Months 13-24)")
    print("=" * 60)

    months_mask = data["month"] >= 13
    metrics = summarize_zinb_effects(
        simulations_intervention,
        simulations_no_intervention,
        months_mask.to_numpy(dtype=bool),
    )

    print("\nExpected total incidents (months 13-24):")
    print(f"  With intervention:    {metrics['scenario']['mean']:.1f}")
    print(f"  Without intervention: {metrics['baseline']['mean']:.1f}")
    print(
        f"  Intervention effect:  {metrics['scenario']['mean'] - metrics['baseline']['mean']:+.1f}"
    )

    print("\n90% intervals:")
    print(
        f"  With intervention:    [{metrics['scenario']['p10']:.1f}, {metrics['scenario']['p90']:.1f}]"
    )
    print(
        f"  Without intervention: [{metrics['baseline']['p10']:.1f}, {metrics['baseline']['p90']:.1f}]"
    )


def main() -> None:
    if not STATSMODELS_AVAILABLE:
        print("statsmodels is required. Install with `pip install statsmodels>=0.14`.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="ZINB Simulation for Monthly Incident Counts")
    parser.add_argument("--runs", type=int, default=1000, help="Number of Monte Carlo simulations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--plot", action="store_true", help="Generate plots if matplotlib is available")
    parser.add_argument("--months", type=int, default=24, help="Number of months to simulate")

    args = parser.parse_args()

    print("ZINB Simulation for Monthly Incident Counts")
    print("=" * 60)
    print(f"Parameters: runs={args.runs}, seed={args.seed}, months={args.months}")

    print("\nGenerating synthetic dataset...")
    data = create_synthetic_dataset(n_months=args.months, seed=args.seed)
    print(f"Created dataset with {len(data)} months")
    zero_prop = (data["incident_count"] == 0).mean()
    print(f"Zero inflation: {zero_prop:.1%} of observations are zeros")

    print("\nFitting ZINB model...")
    model, exog, exog_infl = fit_zinb_model(data)
    print("Model fitting complete")

    print("\nCreating intervention scenarios...")
    scenario_data, no_intervention_data = create_scenarios(data)

    exog_vars = ["intervention", "seasonal_factor", "lag_count"]
    exog_scenario = sm.add_constant(scenario_data[exog_vars], has_constant="add")
    exog_no_intervention = sm.add_constant(no_intervention_data[exog_vars], has_constant="add")

    print(f"\nRunning {args.runs} Monte Carlo simulations...")
    simulations_intervention = simulate_forward(model, exog_scenario, n_runs=args.runs, seed=args.seed)
    simulations_no_intervention = simulate_forward(
        model, exog_no_intervention, n_runs=args.runs, seed=args.seed + 1
    )
    print("Simulations complete")

    print_results(model, data, simulations_intervention, simulations_no_intervention)

    if args.plot:
        if MATPLOTLIB_AVAILABLE:
            plot_results(data, simulations_intervention, simulations_no_intervention)
        else:
            print("Matplotlib not available. Skipping plots.")

    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
