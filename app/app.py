from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st

from app.simlib.bn import (
    apply_interventions,
    build_network_graph,
    calculate_probabilities,
    create_bayesian_network_structure,
    generate_synthetic_data,
)
from app.simlib.nb_baseline import (
    DEFAULT_MONTHS,
    generate_nb_baseline_dataset,
    run_nb_scenario,
)
from app.simlib.zinb import (
    STATSMODELS_AVAILABLE,
    create_synthetic_dataset,
    fit_zinb_model,
    prepare_zinb_intervention_designs,
    simulate_forward,
    summarize_zinb_effects,
)

try:  # Optional dependency for interactive charts
    import plotly.express as px
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:  # pragma: no cover
    PLOTLY_AVAILABLE = False
    px = None  # type: ignore
    go = None  # type: ignore

st.set_page_config(page_title="NB / ZINB / BN Simulation Dashboard", layout="wide")
st.title("Interactive Causal Simulation Playground")
st.caption(
    "Experiment with a Negative Binomial baseline scenario, a Zero-Inflated Negative Binomial model, "
    "and a Bayesian Network do-intervention analysis using synthetic data."
)


def render_nb_tab() -> None:
    st.subheader("Negative Binomial baseline")
    base_data = generate_nb_baseline_dataset()
    months_mask = base_data["month"].isin(DEFAULT_MONTHS)

    defaults = {
        "parenting_risk": float(base_data.loc[months_mask, "parenting_risk"].mean()),
        "peer_risk": float(base_data.loc[months_mask, "peer_risk"].mean()),
        "law_edu": float(base_data.loc[months_mask, "law_edu"].mean()),
        "night_net": float(base_data.loc[months_mask, "night_net"].mean()),
    }

    control_cols = st.columns(4)
    scenario_values = {
        "parenting_risk": control_cols[0].slider(
            "Parenting risk (0=protective, 1=high risk)",
            min_value=0.0,
            max_value=1.0,
            value=round(defaults["parenting_risk"], 2),
            step=0.01,
        ),
        "peer_risk": control_cols[1].slider(
            "Peer risk index (0-3.5)",
            min_value=0.0,
            max_value=3.5,
            value=round(defaults["peer_risk"], 2),
            step=0.05,
        ),
        "law_edu": control_cols[2].slider(
            "Law education engagement (0-2)",
            min_value=0.0,
            max_value=2.0,
            value=round(defaults["law_edu"], 2),
            step=0.05,
        ),
        "night_net": control_cols[3].slider(
            "Night-time social media use (hours)",
            min_value=0.0,
            max_value=10.0,
            value=round(defaults["night_net"], 2),
            step=0.1,
        ),
    }

    sim_cols = st.columns(3)
    alpha = sim_cols[0].slider("Dispersion (alpha)", min_value=0.1, max_value=2.0, value=0.65, step=0.05)
    runs = sim_cols[1].slider("Monte Carlo runs", min_value=100, max_value=5000, value=1000, step=100)
    seed = sim_cols[2].number_input("Simulation seed", min_value=0, max_value=9999, value=2024, step=1)

    results = run_nb_scenario(base_data, scenario_values, alpha=alpha, runs=runs, seed=seed)

    summary = results["summary"]
    baseline_metrics = summary["baseline"]
    scenario_metrics = summary["scenario"]

    scenario_df = results["scenario_data"].copy()
    scenario_df = scenario_df.rename(columns={"observed_count": "Observed"})
    scenario_df["Baseline μ"] = results["baseline_mu"]
    scenario_df["Scenario μ"] = results["scenario_mu"]

    focused_df = scenario_df[scenario_df["month"].isin(DEFAULT_MONTHS)][
        ["month", "Observed", "Baseline μ", "Scenario μ"]
    ].reset_index(drop=True)

    st.markdown("### Months 4–6 expectations")
    st.dataframe(
        focused_df.style.format({"Observed": "{:.0f}", "Baseline μ": "{:.2f}", "Scenario μ": "{:.2f}"}),
        use_container_width=True,
    )

    if PLOTLY_AVAILABLE:
        melt_df = scenario_df[scenario_df["month"].isin(DEFAULT_MONTHS)][
            ["month", "Observed", "Baseline μ", "Scenario μ"]
        ].melt(id_vars="month", var_name="Series", value_name="Count")
        fig = px.bar(
            melt_df,
            x="month",
            y="Count",
            color="Series",
            barmode="group",
            title="Observed vs predicted counts (months 4–6)",
        )
        fig.update_layout(xaxis_title="Month", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    else:  # pragma: no cover
        st.line_chart(
            scenario_df.set_index("month")[
                ["Observed", "Baseline μ", "Scenario μ"]
            ],
            height=320,
        )

    st.markdown("### Scenario totals (months 4–6)")
    metric_cols = st.columns(3)
    metric_cols[0].metric(
        "Baseline expected total",
        f"{baseline_metrics['mean']:.1f}",
        f"90% interval: {baseline_metrics['p10']:.1f} – {baseline_metrics['p90']:.1f}",
    )
    metric_cols[1].metric(
        "Scenario expected total",
        f"{scenario_metrics['mean']:.1f}",
        f"90% interval: {scenario_metrics['p10']:.1f} – {scenario_metrics['p90']:.1f}",
    )
    metric_cols[2].metric(
        "Change vs baseline",
        f"{summary['delta_mean']:+.1f}",
        f"Δ scenario - baseline",
    )

    st.caption(
        "Adjust the risk levers for months 4–6 to see how protective actions or higher risk exposure "
        "shift the expected incident totals under a Negative Binomial model."
    )


def render_zinb_tab() -> None:
    st.subheader("Zero-Inflated Negative Binomial (ZINB)")

    if not STATSMODELS_AVAILABLE:
        st.error(
            "statsmodels is required for the ZINB simulation. Install it with "
            "`pip install statsmodels>=0.14` to enable this tab."
        )
        return

    ctrl_cols = st.columns(3)
    n_months = ctrl_cols[0].slider("Months to simulate", min_value=12, max_value=48, value=24, step=1)
    seed = ctrl_cols[1].number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)
    runs = ctrl_cols[2].slider("Monte Carlo runs", min_value=200, max_value=4000, value=1000, step=100)

    with st.expander("Intervention levers", expanded=True):
        auth_parenting = st.checkbox(
            "Activate supportive parenting campaign (reduces structural zeros after month 6)",
            value=False,
        )
        peer_shift = st.slider(
            "Peer risk shift applied to lagged counts",
            min_value=-1.5,
            max_value=1.5,
            value=0.0,
            step=0.1,
        )
        show_sample = st.checkbox("Show a sampled simulation path", value=False)

    data = create_synthetic_dataset(n_months=n_months, seed=seed)
    zero_prop = (data["incident_count"] == 0).mean()

    model, _, _ = fit_zinb_model(data)
    baseline_exog, baseline_infl, scenario_exog, scenario_infl, scenario_data = prepare_zinb_intervention_designs(
        data,
        auth_parenting=auth_parenting,
        peer_shift=peer_shift,
    )

    baseline_sim = simulate_forward(model, baseline_exog, n_runs=runs, seed=seed)
    scenario_sim = simulate_forward(model, scenario_exog, n_runs=runs, seed=seed + 1)

    impact_mask = scenario_data["intervention"] == 1
    if not impact_mask.any():
        impact_mask = scenario_data["month"] > scenario_data["month"].median()

    totals = summarize_zinb_effects(baseline_sim, scenario_sim, impact_mask.to_numpy(dtype=bool))

    st.markdown("### Model diagnostics")
    diag_cols = st.columns(3)
    diag_cols[0].metric("Zero proportion", f"{zero_prop:.1%}")
    diag_cols[1].metric("Intervention window months", f"{impact_mask.sum()} of {len(data)}")
    diag_cols[2].metric("Mean Δ expected incidents", f"{totals['delta']:+.2f}")

    try:
        coef_table = model.summary2().tables[1].reset_index().rename(columns={"index": "parameter"})
        display_table = coef_table[["parameter", "Coef.", "Std.Err.", "P>|z|"]]
    except Exception:  # pragma: no cover - fallback for older statsmodels
        params = model.params
        display_table = pd.DataFrame(
            {
                "parameter": params.index,
                "Coef.": params.values,
                "Std.Err.": getattr(model, "bse", pd.Series([float("nan")] * len(params))).values,
                "P>|z|": getattr(model, "pvalues", pd.Series([float("nan")] * len(params))).values,
            }
        )
    st.dataframe(display_table.round(3), use_container_width=True)

    if PLOTLY_AVAILABLE:
        months = scenario_data["month"].to_numpy()
        actual = data["incident_count"].to_numpy()
        baseline_mean = baseline_sim.mean(axis=0)
        scenario_mean = scenario_sim.mean(axis=0)
        baseline_p10 = np.quantile(baseline_sim, 0.1, axis=0)
        baseline_p90 = np.quantile(baseline_sim, 0.9, axis=0)
        scenario_p10 = np.quantile(scenario_sim, 0.1, axis=0)
        scenario_p90 = np.quantile(scenario_sim, 0.9, axis=0)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=months,
                y=actual,
                mode="lines+markers",
                name="Actual",
                line=dict(color="#1f77b4"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=months,
                y=baseline_mean,
                mode="lines",
                name="Baseline mean",
                line=dict(color="#ff7f0e"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=months,
                y=scenario_mean,
                mode="lines",
                name="Scenario mean",
                line=dict(color="#2ca02c"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([months, months[::-1]]),
                y=np.concatenate([baseline_p10, baseline_p90[::-1]]),
                fill="toself",
                fillcolor="rgba(255, 127, 14, 0.15)",
                line=dict(color="rgba(255,127,14,0)"),
                hoverinfo="skip",
                showlegend=True,
                name="Baseline 80%",
                mode="lines",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([months, months[::-1]]),
                y=np.concatenate([scenario_p10, scenario_p90[::-1]]),
                fill="toself",
                fillcolor="rgba(44, 160, 44, 0.15)",
                line=dict(color="rgba(44,160,44,0)"),
                hoverinfo="skip",
                showlegend=True,
                name="Scenario 80%",
                mode="lines",
            )
        )
        fig.update_layout(
            title="Actual vs simulated counts",
            xaxis_title="Month",
            yaxis_title="Incident count",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:  # pragma: no cover
        line_df = pd.DataFrame(
            {
                "Actual": data["incident_count"],
                "Baseline mean": baseline_sim.mean(axis=0),
                "Scenario mean": scenario_sim.mean(axis=0),
            },
            index=scenario_data["month"],
        )
        st.line_chart(line_df)

    st.markdown("### Expected totals across intervention window")
    totals_cols = st.columns(3)
    totals_cols[0].metric(
        "Baseline mean total",
        f"{totals['baseline']['mean']:.1f}",
        f"80%: {totals['baseline']['p10']:.1f}-{totals['baseline']['p90']:.1f}",
    )
    totals_cols[1].metric(
        "Scenario mean total",
        f"{totals['scenario']['mean']:.1f}",
        f"80%: {totals['scenario']['p10']:.1f}-{totals['scenario']['p90']:.1f}",
    )
    totals_cols[2].metric("Delta", f"{totals['delta']:+.1f}")

    if show_sample:
        sample_path = pd.DataFrame(
            {
                "month": scenario_data["month"],
                "baseline": baseline_sim[0],
                "intervention": scenario_sim[0],
            }
        )
        st.write("Sampled simulation path (first Monte Carlo run)")
        st.dataframe(sample_path, hide_index=True, use_container_width=True)

    st.caption(
        "Toggle interventions to see how reducing structural zeros or shifting peer risk modifies "
        "expected incident totals under a ZINB model."
    )


def render_bn_tab() -> None:
    st.subheader("Bayesian Network do-intervention")
    bn_cols = st.columns(2)
    n_samples = bn_cols[0].slider("Synthetic cohort size", min_value=200, max_value=5000, value=1000, step=100)
    seed = bn_cols[1].number_input("Seed", min_value=0, max_value=9999, value=42, step=1)

    data = generate_synthetic_data(n_samples=n_samples, seed=seed)
    baseline_probs = calculate_probabilities(data, "SchoolDiscipline")

    with st.expander("Intervention settings", expanded=True):
        set_parent = st.checkbox("Set ParentingStyle evidence", value=False)
        parent_choice = "protect"
        if set_parent:
            parent_choice = st.selectbox("ParentingStyle value", ["protect", "risk"], index=0)

        set_law = st.checkbox("Set LawEdu evidence", value=False)
        law_choice = 2
        if set_law:
            law_choice = st.selectbox("LawEdu level", options=[0, 1, 2], index=2)

        combined = st.checkbox(
            "Apply combined protective package (Parenting=protect & LawEdu=high)",
            value=False,
        )

    interventions: Dict[str, object] = {}
    if combined:
        interventions = {"ParentingStyle": "protect", "LawEdu": 2}
    else:
        if set_parent:
            interventions["ParentingStyle"] = parent_choice
        if set_law:
            interventions["LawEdu"] = law_choice

    scenario_data = apply_interventions(data, interventions, seed=seed + 7) if interventions else data
    scenario_probs = calculate_probabilities(scenario_data, "SchoolDiscipline")

    states = ["low", "med", "high"]
    probs_df = pd.DataFrame(
        {
            "State": states,
            "Baseline": [baseline_probs.get(state, 0.0) for state in states],
            "Intervention": [scenario_probs.get(state, 0.0) for state in states],
        }
    )
    probs_df["Δ"] = probs_df["Intervention"] - probs_df["Baseline"]

    st.dataframe(probs_df.set_index("State").applymap(lambda v: f"{v:.3f}"), use_container_width=True)

    if PLOTLY_AVAILABLE:
        fig = px.bar(
            probs_df.melt(id_vars="State", value_vars=["Baseline", "Intervention"], var_name="Scenario", value_name="Probability"),
            x="State",
            y="Probability",
            color="Scenario",
            barmode="group",
            title="School discipline distribution",
        )
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    else:  # pragma: no cover
        st.bar_chart(probs_df.set_index("State")[["Baseline", "Intervention"]])

    st.markdown("### Δ probabilities")
    delta_cols = st.columns(3)
    for col, state in zip(delta_cols, states):
        col.metric(f"Δ P({state})", f"{probs_df.loc[probs_df['State'] == state, 'Δ'].iloc[0]:+.3f}")

    graph = build_network_graph()
    if graph is not None and PLOTLY_AVAILABLE:
        try:
            import networkx as nx

            pos = nx.spring_layout(graph, seed=seed)
            edge_x, edge_y = [], []
            for source, target in graph.edges():
                x0, y0 = pos[source]
                x1, y1 = pos[target]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color="#888"), hoverinfo="none", mode="lines")

            node_x, node_y, text = [], [], []
            for node in graph.nodes():
                node_x.append(pos[node][0])
                node_y.append(pos[node][1])
                text.append(node)

            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                text=text,
                textposition="top center",
                marker=dict(size=16, color="#636EFA"),
                hoverinfo="text",
            )

            fig = go.Figure(data=[edge_trace, node_trace])
            fig.update_layout(
                title="Bayesian Network structure",
                showlegend=False,
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:  # pragma: no cover
            st.info("Install networkx to view the network structure.")
    else:
        st.info("Plotly and networkx are needed to render the network graph.")

    st.caption(
        "Set evidence on parenting style or law education, or apply a combined intervention to "
        "instantly see how the distribution of SchoolDiscipline outcomes shifts."
    )


nb_tab, zinb_tab, bn_tab = st.tabs(["NB Baseline", "ZINB", "BN do-intervention"])

with nb_tab:
    render_nb_tab()

with zinb_tab:
    render_zinb_tab()

with bn_tab:
    render_bn_tab()
