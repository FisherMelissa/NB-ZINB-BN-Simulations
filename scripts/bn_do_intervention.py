#!/usr/bin/env python3
"""
Bayesian Network do-intervention demo for risk/protection factors and school discipline.

This CLI wrapper now reuses the shared helpers located in ``app.simlib.bn`` so that the
Streamlit dashboard and the standalone script stay consistent.
"""

from __future__ import annotations

from typing import Dict, List

from app.simlib.bn import (
    calculate_probabilities,
    create_bayesian_network_structure,
    generate_synthetic_data,
    sample_trajectories,
    simulate_intervention,
)


def perform_do_interventions(data: List[Dict[str, object]]) -> None:
    """Perform standard do-intervention comparisons and print the results."""
    print("\n" + "=" * 60)
    print("DO-INTERVENTION ANALYSIS")
    print("=" * 60)

    baseline_probs = calculate_probabilities(data, "SchoolDiscipline")

    print("\nBASELINE P(SchoolDiscipline):")
    for state in ["low", "med", "high"]:
        prob = baseline_probs.get(state, 0.0)
        print(f"  P(SchoolDiscipline={state}) = {prob:.3f}")

    interventions = [
        ("do(ParentingStyle=protect)", simulate_intervention(data, "ParentingStyle", "protect", seed=1)),
        ("do(LawEdu=high)", simulate_intervention(data, "LawEdu", 2, seed=2)),
        ("do(NightNet=low)", simulate_intervention(data, "NightNet", "low", seed=3)),
    ]

    for name, dataset in interventions:
        probs = calculate_probabilities(dataset, "SchoolDiscipline")
        print("\n" + "-" * 40)
        print(f"INTERVENTION: {name}")
        for state in ["low", "med", "high"]:
            prob = probs.get(state, 0.0)
            baseline = baseline_probs.get(state, 0.0)
            delta = prob - baseline
            print(f"  P(SchoolDiscipline={state}) = {prob:.3f}")
            print(f"    Δ = {delta:+.3f}")

    print("\n" + "=" * 60)
    print("INTERVENTION EFFECTIVENESS SUMMARY")
    print("=" * 60)
    print(f"{'Intervention':<25} {'ΔP(high)':<10} {'ΔP(med)':<10} {'ΔP(low)':<10}")
    print("-" * 60)

    for (name, dataset) in interventions:
        probs = calculate_probabilities(dataset, "SchoolDiscipline")
        delta_high = probs.get("high", 0.0) - baseline_probs.get("high", 0.0)
        delta_med = probs.get("med", 0.0) - baseline_probs.get("med", 0.0)
        delta_low = probs.get("low", 0.0) - baseline_probs.get("low", 0.0)
        print(f"{name:<25} {delta_high:+10.3f} {delta_med:+10.3f} {delta_low:+10.3f}")


def print_sampled_trajectories(data: List[Dict[str, object]], n_students: int = 10) -> None:
    sampled = sample_trajectories(data, n_students=n_students, seed=42)

    print(f"\n" + "=" * 60)
    print(f"SAMPLED TRAJECTORIES FOR {len(sampled)} STUDENTS")
    print("=" * 60)
    print(f"{'Parenting':<10} {'PeerRisk':<9} {'LawEdu':<7} {'NightNet':<9} {'PsychReg':<9} {'Discipline':<11}")
    print("-" * 60)

    for sample in sampled:
        print(
            f"{sample['ParentingStyle']:<10} {sample['PeerRisk']:<9} {sample['LawEdu']:<7} "
            f"{sample['NightNet']:<9} {sample['PsychReg']:<9} {sample['SchoolDiscipline']:<11}"
        )


def main() -> None:
    print("Bayesian Network do-intervention Demo")
    print("Risk/Protection Factors -> School Discipline")

    print("\n1. Creating Bayesian Network structure...")
    nodes, edges = create_bayesian_network_structure()
    print(f"   Nodes: {nodes}")
    print(f"   Edges: {edges}")

    print("\n2. Generating synthetic data...")
    data = generate_synthetic_data(n_samples=1000, seed=42)
    print(f"   Generated {len(data)} samples")

    print("\n   Variable distributions:")
    for var in nodes:
        probs = calculate_probabilities(data, var)
        print(f"   {var}: {dict(sorted(probs.items()))}")

    print("\n3. Performing do-intervention analysis...")
    perform_do_interventions(data)

    print("\n4. Sampling student trajectories...")
    print_sampled_trajectories(data, n_students=10)

    print("\n" + "=" * 60)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
