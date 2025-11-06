"""Shared helpers for the Bayesian Network simulation and interventions."""

from __future__ import annotations

import random
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def create_bayesian_network_structure() -> Tuple[List[str], List[Tuple[str, str]]]:
    nodes = [
        "ParentingStyle",
        "PeerRisk",
        "LawEdu",
        "NightNet",
        "PsychReg",
        "SchoolDiscipline",
    ]

    edges = [
        ("ParentingStyle", "PsychReg"),
        ("ParentingStyle", "PeerRisk"),
        ("PeerRisk", "SchoolDiscipline"),
        ("NightNet", "SchoolDiscipline"),
        ("LawEdu", "SchoolDiscipline"),
        ("PsychReg", "SchoolDiscipline"),
    ]
    return nodes, edges


def generate_synthetic_data(n_samples: int = 1000, seed: int = 42) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    data: List[Dict[str, object]] = []

    for _ in range(n_samples):
        sample: Dict[str, object] = {}

        sample["ParentingStyle"] = "protect" if rng.random() < 0.6 else "risk"
        sample["LawEdu"] = rng.choices([0, 1, 2], weights=[0.3, 0.4, 0.3])[0]

        night_cont = max(0.0, min(10.0, rng.gauss(5.0, 2.0)))
        if night_cont < 3.33:
            sample["NightNet"] = "low"
        elif night_cont < 6.67:
            sample["NightNet"] = "med"
        else:
            sample["NightNet"] = "high"

        if sample["ParentingStyle"] == "protect":
            sample["PeerRisk"] = rng.choices([0, 1, 2, 3], weights=[0.4, 0.3, 0.2, 0.1])[0]
            sample["PsychReg"] = rng.choices([0, 1, 2], weights=[0.5, 0.3, 0.2])[0]
        else:
            sample["PeerRisk"] = rng.choices([0, 1, 2, 3], weights=[0.1, 0.2, 0.3, 0.4])[0]
            sample["PsychReg"] = rng.choices([0, 1, 2], weights=[0.2, 0.3, 0.5])[0]

        sample["SchoolDiscipline"] = _discipline_from_features(sample)
        data.append(sample)

    return data


def calculate_probabilities(data: Sequence[Dict[str, object]], target_var: str) -> Dict[object, float]:
    counts = Counter(sample[target_var] for sample in data)
    total = len(data)
    return {key: value / total for key, value in counts.items()}


def _discipline_from_features(sample: Dict[str, object]) -> str:
    risk_score = float(sample.get("PeerRisk", 0)) * 0.3

    night_net = sample.get("NightNet", "med")
    if night_net == "high":
        risk_score += 0.4
    elif night_net == "med":
        risk_score += 0.2

    law_edu = int(sample.get("LawEdu", 1))
    risk_score += (2 - law_edu) * 0.2

    psych_reg = int(sample.get("PsychReg", 1))
    risk_score += psych_reg * 0.25

    if risk_score < 1.0:
        return "low"
    if risk_score < 2.0:
        return "med"
    return "high"


def apply_interventions(
    data: Sequence[Dict[str, object]],
    interventions: Dict[str, object],
    seed: int = 99,
) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    adjusted: List[Dict[str, object]] = []

    for original in data:
        updated = dict(original)

        for key, value in interventions.items():
            updated[key] = value

        if "ParentingStyle" in interventions:
            if interventions["ParentingStyle"] == "protect":
                updated["PeerRisk"] = rng.choices([0, 1, 2, 3], weights=[0.4, 0.3, 0.2, 0.1])[0]
                updated["PsychReg"] = rng.choices([0, 1, 2], weights=[0.5, 0.3, 0.2])[0]
            else:
                updated["PeerRisk"] = rng.choices([0, 1, 2, 3], weights=[0.1, 0.2, 0.3, 0.4])[0]
                updated["PsychReg"] = rng.choices([0, 1, 2], weights=[0.2, 0.3, 0.5])[0]

        updated["SchoolDiscipline"] = _discipline_from_features(updated)
        adjusted.append(updated)

    return adjusted


def simulate_intervention(
    data: Sequence[Dict[str, object]],
    intervention_var: str,
    intervention_value: object,
    seed: int = 101,
) -> List[Dict[str, object]]:
    return apply_interventions(data, {intervention_var: intervention_value}, seed=seed)


def sample_trajectories(
    data: Sequence[Dict[str, object]],
    n_students: int = 10,
    seed: int = 123,
) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    n_students = min(len(data), max(1, n_students))
    sampled = rng.sample(list(data), n_students)
    return sampled


def build_network_graph():  # pragma: no cover - optional dependency
    try:
        import networkx as nx
    except ImportError:
        return None

    nodes, edges = create_bayesian_network_structure()
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph
