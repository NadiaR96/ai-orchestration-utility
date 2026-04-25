import os
from typing import Any, Dict, List

import pandas as pd
import requests
import streamlit as st


DEFAULT_API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
STRATEGIES = ["balanced", "quality", "cost_aware", "latency_aware", "rag"]

LAYER_DESCRIPTIONS = {
    "API": "FastAPI routes for task execution, comparison, recommendation, and leaderboard retrieval.",
    "Orchestration": "Coordinates model invocation, retrieval mode, and evaluation flow.",
    "Evaluation": "Computes metrics, normalizes values, and applies scoring strategies.",
    "RAG": "Retrieval and prompt-context construction used for grounded responses.",
    "Experiment": "Logs and reads historical/live run data for leaderboard and recommendations.",
}

LAYER_COLORS = {
    "API": "#0ea5e9",
    "Orchestration": "#6366f1",
    "Evaluation": "#22c55e",
    "RAG": "#f59e0b",
    "Experiment": "#ec4899",
}

VIEW_LAYER_FLOW = {
    "Run Task": ["API", "Orchestration", "RAG", "Evaluation", "Experiment"],
    "Compare": ["API", "Orchestration", "RAG", "Evaluation", "Experiment"],
    "Experiments": ["API", "Orchestration", "RAG", "Evaluation", "Experiment"],
    "Recommendation": ["API", "Experiment", "Evaluation"],
    "Live Degradation": ["API", "Experiment", "Evaluation"],
}


st.set_page_config(page_title="LLM Evaluation Dashboard", layout="wide")


@st.cache_data(ttl=20)
def _get(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=5)
def _post(url: str, payload: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _winner_badge(model: str, winner: str, tied_winners: List[str] | None = None) -> str:
    if tied_winners and model in tied_winners:
        return "TIED"
    return "WINNER" if model == winner else ""


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_compare_selection_message(
    selected_variant: str,
    strategy: str,
    best_score: float,
    fastest_variant: str,
    cheapest_variant: str,
    tie_variants: List[str] | None = None,
) -> str:
    tie_variants = tie_variants or []

    if tie_variants:
        tied = ", ".join(tie_variants)
        return (
            f"Models selected (tie): {tied} based on {strategy} strategy and trade-offs "
            f"(best_score={best_score:.4f}, fastest={fastest_variant}, cheapest={cheapest_variant})."
        )

    return (
        f"Model selected: {selected_variant} based on {strategy} strategy and trade-offs "
        f"(best_score={best_score:.4f}, fastest={fastest_variant}, cheapest={cheapest_variant})."
    )


def _confidence_label(confidence: float) -> str:
    if confidence < 0.35:
        return "INSUFFICIENT EVIDENCE"
    if confidence >= 0.8:
        return "HIGH"
    if confidence >= 0.55:
        return "MEDIUM"
    return "LOW"


def _confidence_style(label: str) -> tuple[str, str]:
    if label == "HIGH":
        return ("#14532d", "#ffffff")
    if label == "MEDIUM":
        return ("#854d0e", "#ffffff")
    if label == "INSUFFICIENT EVIDENCE":
        return ("#334155", "#ffffff")
    return ("#7f1d1d", "#ffffff")


def _validity_style(status: str) -> tuple[str, str]:
    if status == "VALID":
        return ("#14532d", "#ffffff")
    if status == "WARNING":
        return ("#854d0e", "#ffffff")
    return ("#7f1d1d", "#ffffff")


def _normalize_failure_mode(mode: str) -> str:
    canonical_to_legacy = {
        "SCORE_SCALE_DRIFT": "score_scale_drift",
        "INSUFFICIENT_DATA": "insufficient_data",
        "LOW_SEPARATION": "weak_separation",
        "HIGH_VARIANCE": "high_variance",
        "LOW_QUALITY": "low_quality",
        "LOW_CONSISTENCY": "low_consistency",
        "USE_CASE_MISMATCH": "use_case_mismatch",
        "ALL_MODELS_WEAK": "all_models_weak",
        "COST_DOMINATED": "cost_dominated",
        "LATENCY_DOMINATED": "latency_dominated",
    }
    return canonical_to_legacy.get(mode, str(mode).lower())


def _failure_mode_label(mode: str) -> str:
    mode_key = _normalize_failure_mode(mode)

    labels = {
        "score_scale_drift": "Score scale drift",
        "insufficient_data": "Insufficient data",
        "weak_separation": "Weak winner separation",
        "high_variance": "High variance",
        "low_quality": "Low quality",
        "low_consistency": "Low consistency",
        "use_case_mismatch": "Use-case mismatch",
        "all_models_weak": "All models weak",
        "cost_dominated": "Cost dominated",
        "latency_dominated": "Latency dominated",
    }
    return labels.get(mode_key, mode_key.replace("_", " ").title())


def _failure_mode_severity(mode: str) -> str:
    mode_key = _normalize_failure_mode(mode)

    high = {"score_scale_drift", "insufficient_data", "high_variance", "all_models_weak"}
    medium = {
        "weak_separation",
        "low_quality",
        "low_consistency",
        "use_case_mismatch",
        "cost_dominated",
        "latency_dominated",
    }
    if mode_key in high:
        return "HIGH"
    if mode_key in medium:
        return "MEDIUM"
    return "LOW"


def _failure_mode_style(severity: str) -> tuple[str, str, str]:
    if severity == "HIGH":
        return ("#7f1d1d", "#ffffff", "!!")
    if severity == "MEDIUM":
        return ("#854d0e", "#ffffff", "!")
    return ("#334155", "#ffffff", "i")


def _failure_mode_priority(severity: str) -> int:
    if severity == "HIGH":
        return 0
    if severity == "MEDIUM":
        return 1
    return 2


def _decision_outcome_from_state(decision_state: str) -> str:
    state = str(decision_state or "").upper()
    return {
        "RECOMMENDED": "RECOMMENDED",
        "CONSTRAINED": "CONSTRAINED",
        "CONSTRAINED_RECOMMENDATION": "CONSTRAINED",
        "ABSTAIN": "NONE",
        "INVALID": "NONE",
    }.get(state, "NONE")


def _recommendation_tradeoff_lines(df: pd.DataFrame, winner: str) -> List[str]:
    lines: List[str] = []
    for _, row in df.iterrows():
        model = str(row["model"])
        if model == winner:
            continue

        delta = float(row.get("delta_from_best", 0.0))
        cheaper = float(row.get("avg_cost_usd", 0.0)) < float(df[df["model"] == winner]["avg_cost_usd"].iloc[0])
        faster = float(row.get("avg_latency_s", 0.0)) < float(df[df["model"] == winner]["avg_latency_s"].iloc[0])
        below_threshold = float(row.get("score", 0.0)) < 0.7

        parts: List[str] = []
        if cheaper:
            parts.append("cheaper")
        if faster:
            parts.append("faster")
        if below_threshold:
            parts.append("below quality threshold")

        if parts:
            lines.append(f"- {model}: {', '.join(parts)} but worse overall ({delta:+.2f} score)")
        else:
            lines.append(f"- {model}: lower overall score ({delta:+.2f})")

    return lines


def _layer_chip(layer: str, active: bool = True) -> str:
    color = LAYER_COLORS.get(layer, "#6b7280")
    if active:
        return (
            f"<span style='display:inline-block;padding:0.2rem 0.55rem;border-radius:999px;"
            f"background:{color};color:#ffffff;font-weight:600;font-size:0.82rem'>{layer}</span>"
        )

    return (
        f"<span style='display:inline-block;padding:0.2rem 0.55rem;border-radius:999px;"
        f"border:1px solid {color};color:{color};font-weight:600;font-size:0.82rem'>{layer}</span>"
    )


def _render_architecture_layers(view: str) -> None:
    st.markdown("### Architecture Layers")

    flow = VIEW_LAYER_FLOW.get(view, [])
    if flow:
        flow_markup = " <span style='color:#9ca3af'>→</span> ".join(_layer_chip(layer, active=True) for layer in flow)
        st.markdown(f"<div style='margin:0.1rem 0 0.85rem 0'>{flow_markup}</div>", unsafe_allow_html=True)

    cols = st.columns(5)
    ordered_layers = ["API", "Orchestration", "Evaluation", "RAG", "Experiment"]

    for col, layer in zip(cols, ordered_layers):
        active = layer in flow
        status = "Active in this view" if active else "Available"
        with col:
            st.markdown(_layer_chip(layer, active=active), unsafe_allow_html=True)
            st.caption(status)
            st.caption(LAYER_DESCRIPTIONS[layer])


def _render_view_layer_context(view: str) -> None:
    flow = VIEW_LAYER_FLOW.get(view, [])
    if not flow:
        return

    st.markdown("#### Workflow Layer Path")
    chips = " ".join(_layer_chip(layer, active=True) for layer in flow)
    st.markdown(chips, unsafe_allow_html=True)
    st.caption("Color-coded execution path for this view from request handling to scoring/logging.")



def _render_compare(base_url: str) -> None:
    st.subheader("Compare Models")
    _render_view_layer_context("Compare")
    
    st.caption("This is the input layer of the system where you can specify a prompt and compare the outputs of different models using a selected scoring strategy. You can also provide an optional ground-truth reference to enable quality metric computation. The results will show the output, score, latency, and cost for each model, along with an overall winner based on the chosen strategy.")

    c1, c2 = st.columns([2, 1])
    with c1:
        prompt = st.text_area("Prompt", value="Explain retrieval augmented generation in one paragraph.")
    with c2:
        models_text = st.text_input("Models (comma-separated)", value="small,quality,default")

    reference_payload = None
    with st.expander("Ground Truth Reference (optional)"):
        compare_multi_ref = st.checkbox(
            "Use multiple references",
            value=False,
            key="compare_multi_ref",
            help="Enable this when multiple valid answers are acceptable. Enter one reference per line.",
        )
        if compare_multi_ref:
            references_text = st.text_area(
                "References (one per line)",
                value="",
                placeholder="Reference 1\nReference 2\nReference 3",
                height=120,
                key="compare_references_multi",
            )
            refs = [line.strip() for line in references_text.splitlines() if line.strip()]
            if refs:
                reference_payload = refs
        else:
            reference_text = st.text_area(
                "Reference text",
                value="",
                placeholder="Provide a ground-truth reference for quality metric computation. Leave empty to skip.",
                height=100,
                key="compare_reference_single",
            )
            if reference_text.strip():
                reference_payload = reference_text.strip()

    c3, c4 = st.columns(2)
    with c3:
        strategy = st.selectbox("Scoring strategy", STRATEGIES, index=0, key="compare_strategy")
    with c4:
        retrieval_modes = st.multiselect(
            "Retrieval modes",
            options=["rag", "none"],
            default=["rag"],
            key="compare_retrieval_modes",
            help="Select one or both modes. Compare can run model x retrieval combinations side by side.",
        )

    compare_use_case = st.text_input(
        "Use case tag (optional)",
        value="",
        key="compare_use_case",
        help="Tag this run (e.g. summarisation, code_gen) so recommendations can match use-case-specific history.",
    )

    if st.button("Run Compare", type="primary"):
        models_raw = [m.strip() for m in models_text.split(",") if m.strip()]
        models = list(dict.fromkeys(models_raw))

        if not models:
            st.error("Please provide at least one model.")
            return
        if len(models_raw) != len(models):
            st.warning("Duplicate model names were removed before running compare.")
        if not retrieval_modes:
            st.error("Please select at least one retrieval mode.")
            return

        rows: List[Dict[str, Any]] = []
        mode_summaries: List[str] = []

        for retrieval_mode in retrieval_modes:
            payload: Dict[str, Any] = {
                "input": prompt,
                "models": models,
                "strategy": strategy,
                "retrieval": retrieval_mode,
            }
            if compare_use_case.strip():
                payload["use_case"] = compare_use_case.strip()
            if reference_payload:
                payload["reference"] = reference_payload

            try:
                data = _post(f"{base_url}/compare", payload)
            except requests.RequestException as exc:
                st.error(f"Compare request failed for retrieval={retrieval_mode}: {exc}")
                return

            runs = data.get("runs") or {}
            evaluations = data.get("evaluations") or {}
            comparison = data.get("comparison") or {}
            mode_winner = comparison.get("winner") or ""
            mode_ties = comparison.get("tied_winners") or []

            if mode_ties:
                mode_summaries.append(f"{retrieval_mode}: tie ({', '.join(mode_ties)})")
            elif mode_winner:
                mode_summaries.append(f"{retrieval_mode}: winner {mode_winner}")

            for model in models:
                run = runs.get(model, {})
                evaluation = evaluations.get(model, {})
                score = _safe_float(evaluation.get("score"))
                latency = _safe_float(run.get("latency"))
                cost = _safe_float(run.get("cost"))
                tradeoff_index = score - (0.25 * latency) - (80.0 * cost)
                variant = f"{model} [{retrieval_mode}]"

                rows.append(
                    {
                        "winner_key": variant,
                        "winner": "",
                        "model": model,
                        "retrieval": retrieval_mode,
                        "score_raw": score,
                        "score": round(score, 4),
                        "latency_s": round(latency, 4),
                        "cost_usd": round(cost, 6),
                        "tradeoff_index": round(tradeoff_index, 4),
                        "output_preview": (run.get("output") or "")[:120],
                    }
                )

        if not rows:
            st.warning("No results returned from compare endpoint.")
            return

        top_score = max(r["score_raw"] for r in rows)
        epsilon = 1e-9
        top_variants = list(
            dict.fromkeys(
                [r["winner_key"] for r in rows if abs(r["score_raw"] - top_score) <= epsilon]
            )
        )
        overall_tie = len(top_variants) > 1
        for r in rows:
            if r["winner_key"] in top_variants:
                r["winner"] = "TIED" if overall_tie else "WINNER"

        if mode_summaries:
            st.caption("Per-retrieval summary: " + " | ".join(mode_summaries))

        display_rows = []
        for r in rows:
            item = dict(r)
            item.pop("winner_key", None)
            item.pop("score_raw", None)
            display_rows.append(item)

        df = pd.DataFrame(display_rows).sort_values(by=["winner", "score"], ascending=[False, False])

        st.caption(
            "Interactive table: Colors highlight trade-offs — 🟢 quality leaders, 🔵 speed leaders, "
            "🟡 cost leaders, 🔴 quality/cost risk, 🟠 latency risk."
        )
        
        # Apply trade-off highlighting
        max_score = df["score"].max() if len(df) > 0 else 1.0
        min_score = df["score"].min() if len(df) > 0 else 0.0
        max_latency = df["latency_s"].max() if len(df) > 0 else 1.0
        max_cost = df["cost_usd"].max() if len(df) > 0 else 1.0
        
        def highlight_row(row):
            score_norm = (row["score"] - min_score) / (max_score - min_score) if max_score > min_score else 0
            latency_norm = row["latency_s"] / max_latency if max_latency > 0 else 0
            cost_norm = row["cost_usd"] / max_cost if max_cost > 0 else 0
            
            styles = []
            for col in row.index:
                if col == "winner" and row["winner"] == "WINNER":
                    styles.append("background-color: #1f7f1f; color: white; font-weight: bold")
                elif col == "winner" and row["winner"] == "TIED":
                    styles.append("background-color: #0f766e; color: white; font-weight: bold")
                elif col == "score" and score_norm > 0.65:
                    styles.append("background-color: #14532d; color: #ffffff; font-weight: bold")  # Dark green
                elif col == "latency_s" and latency_norm < 0.35:
                    styles.append("background-color: #1e3a8a; color: #ffffff; font-weight: bold")  # Dark blue
                elif col == "cost_usd" and cost_norm < 0.35:
                    styles.append("background-color: #7c5700; color: #ffffff; font-weight: bold")  # Dark yellow
                elif col == "score" and score_norm < 0.35:
                    styles.append("background-color: #7f1d1d; color: #ffffff")  # Dark red risk
                elif col == "latency_s" and latency_norm > 0.65:
                    styles.append("background-color: #9a3412; color: #ffffff")  # Dark orange risk
                elif col == "cost_usd" and cost_norm > 0.65:
                    styles.append("background-color: #7f1d1d; color: #ffffff")  # Dark red risk
                else:
                    styles.append("")
            return styles
        
        styled_df = df.style.apply(highlight_row, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        retrieval_set = set(df["retrieval"].tolist()) if not df.empty else set()
        if {"rag", "none"}.issubset(retrieval_set):
            st.markdown("### RAG vs None Summary")

            rag_df = df[df["retrieval"] == "rag"]
            none_df = df[df["retrieval"] == "none"]

            rag_wins = int((rag_df["winner"] == "WINNER").sum()) + int((rag_df["winner"] == "TIED").sum())
            none_wins = int((none_df["winner"] == "WINNER").sum()) + int((none_df["winner"] == "TIED").sum())

            rag_avg_score = float(rag_df["score"].mean()) if not rag_df.empty else 0.0
            none_avg_score = float(none_df["score"].mean()) if not none_df.empty else 0.0
            rag_avg_latency = float(rag_df["latency_s"].mean()) if not rag_df.empty else 0.0
            none_avg_latency = float(none_df["latency_s"].mean()) if not none_df.empty else 0.0
            rag_avg_cost = float(rag_df["cost_usd"].mean()) if not rag_df.empty else 0.0
            none_avg_cost = float(none_df["cost_usd"].mean()) if not none_df.empty else 0.0

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("RAG winner/tie count", f"{rag_wins}")
            with c2:
                st.metric("None winner/tie count", f"{none_wins}")
            with c3:
                st.metric("Avg score delta (RAG - None)", f"{(rag_avg_score - none_avg_score):+.4f}")
            with c4:
                st.metric("Avg latency delta (RAG - None)", f"{(rag_avg_latency - none_avg_latency):+.4f}s")

            st.caption(
                f"Cost delta (RAG - None): {(rag_avg_cost - none_avg_cost):+.6f} USD | "
                f"Avg RAG score={rag_avg_score:.4f}, Avg None score={none_avg_score:.4f}"
            )

            if rag_avg_score <= none_avg_score and rag_avg_latency > none_avg_latency:
                st.warning(
                    "RAG currently shows no average score lift over none while adding latency. "
                    "Try stronger grounding prompts and references to validate retrieval value."
                )

        if not df.empty:
            best = df.iloc[0]
            fastest = df.sort_values("latency_s", ascending=True).iloc[0]
            cheapest = df.sort_values("cost_usd", ascending=True).iloc[0]

            selection_msg = _build_compare_selection_message(
                selected_variant=top_variants[0],
                strategy=strategy,
                best_score=_safe_float(best["score"]),
                fastest_variant=f"{fastest['model']} [{fastest['retrieval']}]",
                cheapest_variant=f"{cheapest['model']} [{cheapest['retrieval']}]",
                tie_variants=top_variants if overall_tie else [],
            )

            if overall_tie:
                st.info(selection_msg)
            else:
                st.success(selection_msg)

            st.markdown("### Trade-offs (Explicit)")
            st.write(
                f"Best score: {best['model']} [{best['retrieval']}] ({best['score']}). "
                f"Fastest: {fastest['model']} [{fastest['retrieval']}] ({fastest['latency_s']}s). "
                f"Cheapest: {cheapest['model']} [{cheapest['retrieval']}] (${cheapest['cost_usd']})."
            )


def _render_run_task(base_url: str) -> None:
    st.subheader("Run Single Task")
    _render_view_layer_context("Run Task")

    prompt = st.text_area("Prompt", value="Explain retrieval augmented generation in one sentence.")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        model = st.selectbox("Model", ["small", "default", "quality"], index=0, key="run_model")
    with c2:
        strategy = st.selectbox("Scoring strategy", STRATEGIES, index=0, key="run_strategy")
    with c3:
        retrieval_mode = st.selectbox(
            "Retrieval mode",
            options=["rag", "none"],
            index=0,
            key="run_retrieval",
            help="rag = retrieve context chunks, none = skip retrieval and use prompt-only generation.",
        )

    run_use_case = st.text_input(
        "Use case tag (optional)",
        value="",
        key="run_use_case",
        help="Tag this run (e.g. summarisation, code_gen) so recommendations can match use-case-specific history.",
    )

    reference_payload = None
    with st.expander("Ground Truth Reference (optional)"):
        run_multi_ref = st.checkbox(
            "Use multiple references",
            value=False,
            key="run_multi_ref",
            help="Enable this when multiple valid answers are acceptable. Enter one reference per line.",
        )
        if run_multi_ref:
            references_text = st.text_area(
                "References (one per line)",
                value="",
                placeholder="Reference 1\nReference 2\nReference 3",
                height=120,
                key="run_references_multi",
            )
            refs = [line.strip() for line in references_text.splitlines() if line.strip()]
            if refs:
                reference_payload = refs
        else:
            reference_text = st.text_area(
                "Reference text",
                value="",
                placeholder="Provide a ground-truth reference for quality metric computation. Leave empty to skip.",
                height=100,
                key="run_reference_single",
            )
            if reference_text.strip():
                reference_payload = reference_text.strip()

    if st.button("Run Task", type="primary", key="run_button"):
        payload: Dict[str, Any] = {
            "input": prompt,
            "model": model,
            "strategy": strategy,
            "retrieval": retrieval_mode,
        }
        if run_use_case.strip():
            payload["use_case"] = run_use_case.strip()
        if reference_payload:
            payload["reference"] = reference_payload

        try:
            data = _post(f"{base_url}/run-task", payload)
        except requests.RequestException as exc:
            st.error(f"Run task request failed: {exc}")
            return

        run = data.get("run") or {}
        evaluation = data.get("evaluations", {}).get("single", {})
        metrics = evaluation.get("metrics") or {}

        st.success(f"Task completed for model: {model}")
        st.write(
            f"{model} produced the following output with a score of {evaluation.get('score'):.4f} "
            f"using {strategy} strategy (retrieval={run.get('retrieval', retrieval_mode)}):"
        )
        st.json({"output": run.get("output"), "score": evaluation.get("score"), "metrics": metrics})


def _render_recommend(base_url: str) -> None:
    st.subheader("Recommendation")
    _render_view_layer_context("Recommendation")

    c1, c2, c3 = st.columns(3)
    with c1:
        use_case = st.text_input("Use case", value="summarisation")
    with c2:
        strategy = st.selectbox("Strategy", STRATEGIES, index=0, key="recommend_strategy")
    with c3:
        source = st.selectbox("Source", ["all", "live", "experiment"], index=0)

    c4, c5 = st.columns(2)
    with c4:
        top_n = st.number_input("Top N", min_value=1, max_value=20, value=3)
    with c5:
        min_samples = st.number_input("Min samples", min_value=1, max_value=1000, value=1)

    quality_floor = st.slider(
        "Quality floor warning threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.70,
        step=0.01,
        help="Show a warning if the recommended model score is below this threshold.",
    )

    fetch_recommendation = st.button("Get Recommendation", type="primary")

    if fetch_recommendation or "recommendation_last_data" in st.session_state:
        if fetch_recommendation:
            params = {
                "use_case": use_case,
                "strategy": strategy,
                "top_n": int(top_n),
                "min_samples": int(min_samples),
                "source": source,
            }

            try:
                data = _get(f"{base_url}/recommend", params)
            except requests.RequestException as exc:
                st.error(f"Recommend request failed: {exc}")
                return

            st.session_state["recommendation_last_data"] = data

        data = st.session_state.get("recommendation_last_data")
        if not isinstance(data, dict):
            st.info("Run a recommendation first.")
            return

        evaluation_payload = data.get("evaluation") or {}
        decision_payload = evaluation_payload.get("decision") or {}
        reliability_payload = evaluation_payload.get("reliability") or {}
        metrics_payload = evaluation_payload.get("metrics") or {}
        failure_analysis_payload = evaluation_payload.get("failure_analysis") or {}

        winner = str(decision_payload.get("selected_model") or data.get("best_model") or "")
        decision_state = str(decision_payload.get("state") or "ABSTAIN")
        recommendation_available = decision_state in {"RECOMMENDED", "CONSTRAINED", "CONSTRAINED_RECOMMENDATION"}
        no_recommendation_reason = str(
            decision_payload.get("reason")
            or data.get("no_recommendation_reason")
            or ""
        ).strip()

        alternatives = data.get("alternatives") or []
        rows: List[Dict[str, Any]] = []
        for alt in alternatives:
            rows.append(
                {
                    "winner": _winner_badge(alt.get("model", ""), winner),
                    "model": alt.get("model"),
                    "score": _safe_float(alt.get("score")),
                    "score_stddev": _safe_float(alt.get("score_stddev")),
                    "delta_from_best": _safe_float(alt.get("score_delta_from_best")),
                    "confidence": _safe_float(alt.get("confidence")),
                    "p95_latency_s": _safe_float(alt.get("p95_latency")),
                    "consistency_above_threshold": _safe_float(alt.get("consistency_above_threshold")),
                    "sample_count": int(_safe_float(alt.get("sample_count"), 0)),
                    "avg_latency_s": _safe_float(alt.get("avg_latency")),
                    "avg_cost_usd": _safe_float(alt.get("avg_cost")),
                    "avg_total_tokens": _safe_float(alt.get("avg_total_tokens")),
                    "avg_cost_per_1k_tokens": _safe_float(alt.get("avg_cost_per_1k_tokens")),
                    "avg_quality_per_1k_tokens": _safe_float(alt.get("avg_quality_per_1k_tokens")),
                }
            )

        df = pd.DataFrame(rows)

        ranking_view = st.selectbox(
            "Alternative ranking view",
            [
                "Strategy score (recommended)",
                "Token efficiency: lowest cost per 1k",
                "Token efficiency: highest quality per 1k",
            ],
            index=0,
            key="recommendation_ranking_view",
            help="Exploration-only sorting. Recommendation decision and gate remain based on the selected strategy.",
        )

        if not df.empty:
            if ranking_view == "Token efficiency: lowest cost per 1k":
                df = df.sort_values(by=["avg_cost_per_1k_tokens", "score"], ascending=[True, False])
            elif ranking_view == "Token efficiency: highest quality per 1k":
                df = df.sort_values(by=["avg_quality_per_1k_tokens", "score"], ascending=[False, False])
            else:
                df = df.sort_values(by=["winner", "score"], ascending=[False, False])

        if not df.empty:
            best = df[df["model"] == winner].iloc[0] if winner in set(df["model"]) else df.iloc[0]
            exploration_best = df.iloc[0]
            runner_up = df[df["model"] != best["model"]].sort_values("score", ascending=False)
            runner_up_row = runner_up.iloc[0] if not runner_up.empty else None
            margin = abs(float(runner_up_row["delta_from_best"])) if runner_up_row is not None else 0.0
            confidence = _safe_float(reliability_payload.get("score"), float(best["confidence"]))
            confidence_label = str(reliability_payload.get("label") or _confidence_label(confidence))
            confidence_reasons = data.get("confidence_reasons") or []
            score_margin = _safe_float(metrics_payload.get("score_margin"), margin)
            score_variance = _safe_float(metrics_payload.get("score_variance"), float(best["score_stddev"]))
            validity_status = str(data.get("validity_status") or "WARNING")
            validity_reasons = data.get("validity_reasons") or []
            failure_modes = failure_analysis_payload.get("modes") or []
            system_state = str(evaluation_payload.get("system_health") or "DEGRADED")
            decision_result = _decision_outcome_from_state(decision_state)
            evaluation_status = str(evaluation_payload.get("evaluation_status") or "VALID")
            decision_reliability = _safe_float(reliability_payload.get("score"), confidence)
            gate_status = str(data.get("gate_status") or "PASS")
            gate_threshold = float(data.get("gate_threshold") or 0.0)
            gate_triggers = data.get("gate_triggers") or []
            badge_bg, badge_fg = _confidence_style(confidence_label)
            validity_bg, validity_fg = _validity_style(validity_status)

            st.markdown("### Trust Summary")
            if not recommendation_available:
                st.error(
                    "No recommendation available. "
                    + (no_recommendation_reason if no_recommendation_reason else "All candidate models are currently too weak.")
                )
            if ranking_view != "Strategy score (recommended)":
                st.caption(
                    f"Exploration view top model: {exploration_best['model']}"
                    f" | Recommendation winner (decision gate): {best['model'] if recommendation_available else 'none'}"
                )
            t1, t2, t3 = st.columns(3)
            with t1:
                st.markdown(
                    f"<span style='display:inline-block;padding:0.2rem 0.6rem;border-radius:999px;background:{badge_bg};color:{badge_fg};font-weight:700;font-size:0.85rem'>"
                    f"Confidence: {confidence_label} ({confidence:.2f})"
                    "</span>",
                    unsafe_allow_html=True,
                )
            with t2:
                st.metric("Score margin", f"{score_margin:+.4f}")
            with t3:
                st.metric("Score variance (std)", f"{score_variance:.4f}")

            st.caption(
                f"System state: {system_state} | Decision result: {decision_result} | "
                f"Decision state: {decision_state} | Evaluation status: {evaluation_status} | "
                f"Reliability: {decision_reliability:.2f}"
            )

            st.markdown(
                f"<span style='display:inline-block;padding:0.2rem 0.6rem;border-radius:999px;background:{validity_bg};color:{validity_fg};font-weight:700;font-size:0.85rem'>"
                f"Validity gate: {validity_status}"
                "</span>",
                unsafe_allow_html=True,
            )

            if validity_reasons:
                st.caption("Validity notes: " + " | ".join(str(r) for r in validity_reasons))

            # Decision Gate block
            gate_bg = "#d4edda" if gate_status == "PASS" else "#f8d7da"
            gate_fg = "#155724" if gate_status == "PASS" else "#721c24"
            gate_icon = "✓" if gate_status == "PASS" else "✗"
            st.markdown(
                f"<span style='display:inline-block;padding:0.2rem 0.6rem;border-radius:999px;background:{gate_bg};color:{gate_fg};font-weight:700;font-size:0.85rem'>"
                f"{gate_icon} Decision gate: {gate_status} (threshold: {gate_threshold:.2f} for '{strategy}')"
                "</span>",
                unsafe_allow_html=True,
            )
            if gate_triggers:
                st.caption("Gate triggers: " + ", ".join(str(t) for t in gate_triggers))

            if failure_modes:
                st.markdown("**Failure mode classification:**")
                ordered_failure_modes = sorted(
                    [str(mode) for mode in failure_modes],
                    key=lambda mode: (
                        _failure_mode_priority(_failure_mode_severity(mode)),
                        _failure_mode_label(mode),
                    ),
                )
                for mode_str in ordered_failure_modes:
                    severity = _failure_mode_severity(mode_str)
                    bg, fg, icon = _failure_mode_style(severity)
                    st.markdown(
                        f"<div style='margin:0.25rem 0'>"
                        f"<span style='display:inline-block;min-width:1.35rem;text-align:center;padding:0.1rem 0.35rem;border-radius:999px;background:{bg};color:{fg};font-weight:700;font-size:0.78rem'>{icon}</span> "
                        f"<span style='display:inline-block;padding:0.1rem 0.45rem;border-radius:999px;background:{bg};color:{fg};font-weight:700;font-size:0.75rem'>{severity}</span> "
                        f"<span style='font-weight:600'>{_failure_mode_label(mode_str)}</span>"
                        "</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown("**Failure mode classification:** none detected")

            reason_lines = [
                f"- Highest average score ({float(best['score']):.4f})",
                (
                    f"- Strong margin over second-best (+{margin:.4f} vs {runner_up_row['model']})"
                    if runner_up_row is not None and margin >= 0.1
                    else (f"- Narrow margin over second-best (+{margin:.4f})" if runner_up_row is not None else "- No runner-up available yet")
                ),
                (
                    f"- Strong evidence depth: {int(best['sample_count'])} samples"
                    if int(best['sample_count']) >= 10
                    else (
                        f"- Moderate evidence depth: {int(best['sample_count'])} samples"
                        if int(best['sample_count']) >= 5
                        else f"- Limited evidence: only {int(best['sample_count'])} samples"
                    )
                ),
                (
                    f"- {'Low' if float(best['score_stddev']) < 0.05 else 'Moderate' if float(best['score_stddev']) < 0.15 else 'High'} variability (std={float(best['score_stddev']):.4f})"
                    if int(best['sample_count']) >= 5
                    else f"- Variability std={float(best['score_stddev']):.4f} (unreliable with only {int(best['sample_count'])} sample(s))"
                ),
                (
                    f"- Confidence: <span style='display:inline-block;padding:0.2rem 0.6rem;border-radius:999px;background:{badge_bg};color:{badge_fg};font-weight:700;font-size:0.85rem'>{confidence_label} ({confidence:.2f})</span>"
                ),
            ]
            reason_lines.extend(f"- Guard rail: {reason}" for reason in confidence_reasons)

            tradeoff_lines = _recommendation_tradeoff_lines(df, str(best["model"]))
            tradeoffs_text = "\n".join(tradeoff_lines) if tradeoff_lines else "- No alternative trade-offs available yet"

            if recommendation_available:
                card_lines = [
                    f"### Recommended: {best['model']}",
                    "",
                    "**Why:**",
                    *reason_lines,
                    "",
                    "**Trade-offs:**",
                    tradeoffs_text,
                    "",
                    "**Performance:**",
                    f"- avg latency: {float(best['avg_latency_s']):.2f}s (p95: {float(best['p95_latency_s']):.2f}s)",
                    f"- avg cost: ${float(best['avg_cost_usd']):.4f}",
                    f"- avg total tokens: {float(best['avg_total_tokens']):.0f} (cost per 1k tokens: ${float(best['avg_cost_per_1k_tokens']):.4f})",
                    "",
                    "**Consistency:**",
                    f"- {float(best['consistency_above_threshold']) * 100:.0f}% of runs above 0.7 score",
                ]
            else:
                # Compress: all weakness signals into a single statement
                _wt = gate_threshold
                _sc = float(best['score'])
                _n = int(best['sample_count'])
                card_lines = [
                    "### Top-performing candidate (not recommended)",
                    "",
                    "**Performance summary:**",
                    f"- Best score: {_sc:.4f} (threshold: {_wt:.2f}) — all candidates below the {strategy} floor",
                    f"- {_n} sample(s); {float(best['consistency_above_threshold']) * 100:.0f}% of runs above 0.7",
                    f"- avg latency: {float(best['avg_latency_s']):.2f}s (p95: {float(best['p95_latency_s']):.2f}s), avg cost: ${float(best['avg_cost_usd']):.4f}",
                    f"- avg total tokens: {float(best['avg_total_tokens']):.0f} (cost per 1k tokens: ${float(best['avg_cost_per_1k_tokens']):.4f})",
                ]
                if runner_up_row is not None:
                    card_lines.append(f"- Runner-up: {runner_up_row['model']} ({float(runner_up_row['score']):.4f})")
            st.markdown("\n".join(card_lines), unsafe_allow_html=True)

            if runner_up_row is not None and margin < 0.05:
                st.warning(
                    f"Margin is weak: only +{margin:.4f} over second-best ({runner_up_row['model']}). "
                    "Consider collecting more samples before locking this recommendation."
                )

            if float(best["score"]) < float(quality_floor):
                st.warning(
                    f"Recommended model is below the quality floor ({float(best['score']):.4f} < {quality_floor:.2f}). "
                    "Treat this as a constrained best option, not an absolute pass."
                )

        st.caption("Interactive table with winner highlighted for recommendation scenarios.")
        st.dataframe(df, use_container_width=True, hide_index=True)

        if recommendation_available:
            st.markdown("### Why this winner")
            st.write(data.get("justification", "No justification provided."))
        else:
            st.markdown("### Performance Summary")
            st.write(data.get("justification", "No recommendation rationale provided."))

        with st.expander("API JSON"):
            st.json(data)

        if not df.empty and recommendation_available:
            st.markdown("### Trade-offs (Explicit)")
            fastest = df.sort_values("avg_latency_s", ascending=True).iloc[0]
            cheapest = df.sort_values("avg_cost_usd", ascending=True).iloc[0]
            cheapest_per_1k = df.sort_values("avg_cost_per_1k_tokens", ascending=True).iloc[0]
            top_score = df.sort_values("score", ascending=False).iloc[0]
            parts = []
            if top_score["model"] != fastest["model"] or top_score["model"] != cheapest["model"]:
                if top_score["model"] == fastest["model"] == cheapest["model"]:
                    parts.append(f"{top_score['model']} leads on score, latency, and cost.")
                else:
                    parts.append(f"Best score: {top_score['model']} ({top_score['score']:.4f})")
                    if fastest["model"] != top_score["model"]:
                        parts.append(f"fastest: {fastest['model']} ({fastest['avg_latency_s']:.3f}s)")
                    if cheapest["model"] != top_score["model"]:
                        parts.append(f"lowest cost: {cheapest['model']} (${cheapest['avg_cost_usd']:.6f})")
                    if cheapest_per_1k["model"] != top_score["model"]:
                        parts.append(
                            f"best token efficiency: {cheapest_per_1k['model']} (${cheapest_per_1k['avg_cost_per_1k_tokens']:.4f}/1k tokens)"
                        )
            else:
                parts.append(f"{top_score['model']} leads on score ({top_score['score']:.4f}), latency ({fastest['avg_latency_s']:.3f}s), and cost (${cheapest['avg_cost_usd']:.6f}).")
            st.write(" · ".join(parts))


def _render_live_degradation(base_url: str) -> None:
    st.subheader("Live Degradation")
    _render_view_layer_context("Live Degradation")

    c1, c2, c3 = st.columns(3)
    with c1:
        strategy = st.selectbox("Sort strategy", STRATEGIES, index=0, key="live_strategy")
    with c2:
        window_hours = st.number_input("Window hours", min_value=1, max_value=168, value=24)
    with c3:
        ranking_basis = st.selectbox("Ranking basis", ["window_avg", "latest"], index=0)

    c4, c5 = st.columns(2)
    with c4:
        min_samples = st.number_input("Min samples", min_value=1, max_value=1000, value=1, key="live_min")
    with c5:
        page_size = st.number_input("Page size", min_value=1, max_value=100, value=25)

    if st.button("Load Live Leaderboard", type="primary"):
        params = {
            "page": 1,
            "page_size": int(page_size),
            "sort_strategy": strategy,
            "window_hours": int(window_hours),
            "min_samples": int(min_samples),
            "ranking_basis": ranking_basis,
        }

        try:
            data = _get(f"{base_url}/leaderboard/live", params)
        except requests.RequestException as exc:
            st.error(f"Live leaderboard request failed: {exc}")
            return

        items = data.get("items") or []
        rows: List[Dict[str, Any]] = []
        degradation_count = 0

        for item in items:
            trend = item.get("trend") or {}
            direction = trend.get("direction", "unknown")
            delta = trend.get("delta_score")
            current_avg = trend.get("current_avg_score")
            previous_avg = trend.get("previous_avg_score")

            degraded = direction == "down" or (_safe_float(delta) < -0.01 if delta is not None else False)
            if degraded:
                degradation_count += 1

            rows.append(
                {
                    "degraded": "YES" if degraded else "",
                    "model": item.get("model"),
                    "score": _safe_float(item.get("scores_by_strategy", {}).get(strategy)),
                    "latest_score": _safe_float(item.get("latest_score")),
                    "sample_count": int(_safe_float(item.get("sample_count"), 0)),
                    "trend_direction": direction,
                    "delta_score": _safe_float(delta, 0.0) if delta is not None else None,
                    "current_avg": _safe_float(current_avg, 0.0) if current_avg is not None else None,
                    "previous_avg": _safe_float(previous_avg, 0.0) if previous_avg is not None else None,
                }
            )

        df = pd.DataFrame(rows)

        if degradation_count > 0:
            st.warning(f"Degradation detected for {degradation_count} model(s).")
        else:
            st.success("No explicit degradation detected in the current window.")

        st.caption("Interactive degradation table. Rows marked 'YES' in degraded are regressing.")
        st.dataframe(df, use_container_width=True, hide_index=True)

        if not df.empty:
            st.markdown("### Degradation Summary (Explicit)")
            down = df[df["trend_direction"] == "down"]
            if down.empty:
                st.write("No models are currently trending down.")
            else:
                for _, row in down.iterrows():
                    st.write(
                        f"{row['model']}: delta_score={row['delta_score']:+.4f}, "
                        f"current_avg={row['current_avg']}, previous_avg={row['previous_avg']}"
                    )


def _render_experiments(base_url: str) -> None:
    st.subheader("Batch Experiments")
    _render_view_layer_context("Experiments")

    st.caption(
        "Run multiple inputs across multiple models in one batch and compare winners per input. "
        "This writes experiment runs to experiments/logs.jsonl with source=experiment."
    )

    c1, c2 = st.columns(2)
    with c1:
        experiment_name = st.text_input("Experiment name", value="ui-batch-experiment")
    with c2:
        strategy = st.selectbox("Scoring strategy", STRATEGIES, index=0, key="experiment_strategy")

    c3, c4 = st.columns(2)
    with c3:
        models_text = st.text_input("Models (comma-separated)", value="small,default,quality", key="exp_models")
    with c4:
        runs_per_input = st.number_input("Runs per input", min_value=1, max_value=20, value=1, step=1)

    experiment_use_case = st.text_input(
        "Use case tag (optional)",
        value="",
        key="experiment_use_case",
        help="Tag all experiment runs for recommendation matching. If empty, experiment name is used.",
    )

    inputs_text = st.text_area(
        "Inputs (one prompt per line)",
        value=(
            "Explain retrieval augmented generation in one paragraph.\n"
            "Summarise why latency/cost trade-offs matter in model selection."
        ),
        height=130,
    )

    if st.button("Run Experiment", type="primary"):
        models = [m.strip() for m in models_text.split(",") if m.strip()]
        models = list(dict.fromkeys(models))
        inputs = [line.strip() for line in inputs_text.splitlines() if line.strip()]

        if not models:
            st.error("Please provide at least one model.")
            return
        if not inputs:
            st.error("Please provide at least one input prompt.")
            return

        payload = {
            "name": experiment_name.strip() or "ui-batch-experiment",
            "inputs": inputs,
            "models": models,
            "strategy": strategy,
            "runs_per_input": int(runs_per_input),
        }
        if experiment_use_case.strip():
            payload["use_case"] = experiment_use_case.strip()

        try:
            with st.spinner("Running experiment — this may take a while for large batches…"):
                data = _post(f"{base_url}/experiments/experiment", payload, timeout=300)
        except requests.RequestException as exc:
            st.error(f"Experiment request failed: {exc}")
            return

        st.success(f"Experiment completed: {data.get('name', payload['name'])}")

        summary = data.get("summary") or {}
        win_counts = summary.get("win_counts") or {}
        if win_counts:
            st.markdown("### Summary")
            summary_rows = [{"model": k, "wins": v} for k, v in win_counts.items()]
            summary_df = pd.DataFrame(summary_rows).sort_values("wins", ascending=False)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

        comparisons = data.get("comparisons") or []
        if comparisons:
            st.markdown("### Per-input Winners")
            rows: List[Dict[str, Any]] = []
            for item in comparisons:
                comp = item.get("comparison") or {}
                rows.append(
                    {
                        "input": item.get("input", "")[:120],
                        "winner": comp.get("winner") or "",
                        "tied_winners": ", ".join(comp.get("tied_winners") or []),
                        "strategy": comp.get("strategy", strategy),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("### Raw Experiment Output")
        st.json(data)


def main() -> None:
    st.title("Large Language Model Evaluation Dashboard")
    st.caption("Evaluate and monitor the performance of large language models. Benchmark different models, get recommendations, and track live performance trends over time. Incorporates standard quality metrics along with latency and cost for holistic insights.")

    base_url = st.sidebar.text_input("API base URL", value=DEFAULT_API_BASE)
    st.sidebar.markdown("Use this UI while backend is running (e.g., uvicorn on localhost:8000).")

    page = st.sidebar.radio(
        "View",
        options=["Run Task", "Compare", "Experiments", "Recommendation", "Live Degradation"],
        index=0,
    )

    with st.sidebar.expander("Architecture Layers", expanded=True):
        for layer in ["API", "Orchestration", "Evaluation", "RAG", "Experiment"]:
            active = layer in VIEW_LAYER_FLOW.get(page, [])
            st.markdown(
                (
                    f"{_layer_chip(layer, active=active)}"
                    f"<div style='margin:0.25rem 0 0.65rem 0;font-size:0.82rem;color:#9ca3af'>"
                    f"{LAYER_DESCRIPTIONS[layer]}"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )

    _render_architecture_layers(page)

    if page == "Run Task":
        _render_run_task(base_url)
    elif page == "Compare":
        _render_compare(base_url)
    elif page == "Experiments":
        _render_experiments(base_url)
    elif page == "Recommendation":
        _render_recommend(base_url)
    else:
        _render_live_degradation(base_url)


if __name__ == "__main__":
    main()
