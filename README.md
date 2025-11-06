# NB-ZINB-BN-Simulations
# cto.new
Ai-service

## Bayesian Network do-intervention Demo

This repository includes a Bayesian Network demonstration for modeling risk/protection factors and their impact on school discipline outcomes using do-intervention analysis.

### Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Usage

Run the Bayesian Network do-intervention demo:

```bash
python scripts/bn_do_intervention.py
```

### Features

- Bayesian Network with nodes: ParentingStyle, PeerRisk, LawEdu, NightNet, PsychReg, SchoolDiscipline
- Synthetic data generation consistent with the DAG structure
- Do-intervention analysis to evaluate causal effects
- Comparison of baseline vs intervention probabilities
- Student trajectory sampling

### Network Structure

The Bayesian Network models the following relationships:
- ParentingStyle → PsychReg
- ParentingStyle → PeerRisk  
- PeerRisk → SchoolDiscipline
- NightNet → SchoolDiscipline
- LawEdu → SchoolDiscipline
- PsychReg → SchoolDiscipline

### Interventions Analyzed

- `do(ParentingStyle=protect)`: Setting parenting style to protective
- `do(LawEdu=high)`: Setting law education to high level
- `do(NightNet=low)`: Setting night net usage to low level

The script compares the probability of `SchoolDiscipline=high` under baseline conditions vs each intervention.

## ZINB Simulation

This repository includes a Zero-Inflated Negative Binomial (ZINB) simulation script for modeling monthly incident counts with many zeros.

### Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Usage

Run the ZINB simulation:

```bash
python scripts/zinb_simulation.py
```

Optional parameters:

```bash
python scripts/zinb_simulation.py --runs 2000 --seed 123 --plot --months 24
```

- `--runs N`: Number of Monte Carlo simulations (default: 1000)
- `--seed N`: Random seed for reproducibility (default: 42)
- `--plot`: Generate plots if matplotlib is available
- `--months N`: Number of months to simulate (default: 24)

### Features

- Synthetic dataset generation with zero inflation
- ZINB model fitting with logit inflation model
- Forward simulation (deterministic + Monte Carlo)
- Intervention vs no-intervention scenario comparison
- Coefficient summary and in-sample expected counts
- Optional plot generation

## Streamlit Dashboard

Launch the interactive dashboard, which combines the NB baseline, ZINB simulation, and Bayesian Network do-intervention views:

```bash
streamlit run app/app.py
```

The app provides three tabs:

- **NB Baseline** – adjust months 4–6 risk levers and view Negative Binomial expectations.
- **ZINB** – explore Zero-Inflated Negative Binomial simulations with parenting/peer interventions.
- **BN do-intervention** – compare baseline vs. evidence-informed causal queries and visualise the network.
