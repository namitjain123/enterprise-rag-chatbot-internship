"""
Streamlit evaluation UI — three tabs:
  Goldens      : browse all Q&A pairs and what metric each targets
  Run Evaluation: pick questions, run RAGAS, watch live progress
  Results      : aggregate + per-question scores from the last run
"""

import json
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "eval", "eval_dataset.json")

METRIC_COLORS = {
    "faithfulness":      "#ef4444",
    "context_precision": "#3b82f6",
    "context_recall":    "#10b981",
    "response_relevancy":"#f59e0b",
}

METRIC_LABELS = {
    "faithfulness":      "Faithfulness",
    "context_precision": "Context Precision",
    "context_recall":    "Context Recall",
    "response_relevancy":"Response Relevancy",
    "answer_correctness":"Answer Correctness",
    "hit_rate":          "Hit Rate",
    "mrr":               "MRR",
}


def _badge(metric: str) -> str:
    color = METRIC_COLORS.get(metric, "#6b7280")
    label = METRIC_LABELS.get(metric, metric)
    return (
        f'<span style="background:{color};color:#fff;padding:2px 8px;'
        f'border-radius:12px;font-size:12px;font-weight:600">{label}</span>'
    )


def _load_dataset() -> list[dict]:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _score_color(score: float | None) -> str:
    if score is None:
        return "#6b7280"
    if score >= 0.8:
        return "#10b981"
    if score >= 0.5:
        return "#f59e0b"
    return "#ef4444"


# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Eval", page_icon="🧪", layout="wide")
st.title("🧪 RAG Evaluation Dashboard")

goldens_tab, run_tab, results_tab = st.tabs(["📋 Goldens", "🚀 Run Evaluation", "📊 Results"])

dataset = _load_dataset()

# ── Tab 1: Goldens ────────────────────────────────────────────────────────────
with goldens_tab:
    st.subheader("Golden Dataset — 10 Q&A Pairs with Ground Truth")
    st.caption(
        "Each golden targets a specific RAGAS metric so we can verify "
        "the evaluator catches real failure modes."
    )

    for item in dataset:
        metric = item.get("target_metric", "")
        with st.expander(f"**{item['id']}** — {item['question']}"):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("**Ground Truth**")
                st.info(item["ground_truth"])
            with col2:
                st.markdown("**Target Metric**")
                st.markdown(_badge(metric), unsafe_allow_html=True)
                st.caption(item.get("metric_note", ""))

# ── Tab 2: Run Evaluation ─────────────────────────────────────────────────────
with run_tab:
    st.subheader("Run a Question")
    st.caption("Click **Run** next to any question to evaluate just that one. "
               "The model answer and the reference answer appear right below it.")

    # Per-question reports cached across reruns, keyed by question index.
    results_by_idx: dict = st.session_state.setdefault("results_by_idx", {})

    for i, item in enumerate(dataset):
        col_q, col_btn = st.columns([6, 1])
        with col_q:
            st.markdown(f"**{item['id']}** — {item['question']}")
        with col_btn:
            run_clicked = st.button("▶ Run", key=f"run_{i}")

        if run_clicked:
            with st.spinner(f"Evaluating {item['id']} (~30–60 s)…"):
                try:
                    from app.eval.run_ragas_eval import run_evaluation

                    report = run_evaluation(indices=[i])
                    row = report["per_question"][0] if report.get("per_question") else None
                    results_by_idx[i] = {
                        "answer": row["answer"] if row else "",
                        "scores": {k: v for k, v in row.items()
                                   if k not in ("question", "answer")} if row else {},
                    }
                except Exception as exc:
                    results_by_idx[i] = {"error": str(exc)}

        # Show cached result for this question (model answer + reference answer).
        cached = results_by_idx.get(i)
        if cached:
            if "error" in cached:
                st.error(f"Evaluation failed: {cached['error']}")
            else:
                st.markdown("**🤖 Model Answer**")
                st.success(cached["answer"])
                st.markdown("**📌 Reference Answer (ground truth)**")
                st.info(item["ground_truth"])
                st.caption("See the **Results** tab for the evaluation metrics.")
        st.divider()

# ── Tab 3: Results ────────────────────────────────────────────────────────────
with results_tab:
    results_by_idx = st.session_state.get("results_by_idx", {})
    scored = {
        i: r for i, r in results_by_idx.items()
        if "error" not in r and r.get("scores")
    }

    if not scored:
        st.info("No results yet. Run a question in the **Run Evaluation** tab.")
    else:
        # ── Aggregate (mean over every question evaluated so far) ──
        all_metrics: list[str] = []
        for r in scored.values():
            for m in r["scores"]:
                if m not in all_metrics:
                    all_metrics.append(m)

        st.subheader("Aggregate Scores")
        st.caption(f"Mean over {len(scored)} evaluated question(s).")
        agg_cols = st.columns(len(all_metrics))
        for col, m in zip(agg_cols, all_metrics):
            vals = [r["scores"].get(m) for r in scored.values()
                    if isinstance(r["scores"].get(m), (int, float))]
            mean = sum(vals) / len(vals) if vals else None
            color = _score_color(mean)
            col.markdown(
                f'<div style="text-align:center;padding:12px;border-radius:8px;'
                f'background:{color}22;border:1px solid {color}">'
                f'<div style="font-size:28px;font-weight:700;color:{color}">'
                f'{"—" if mean is None else f"{mean:.3f}"}</div>'
                f'<div style="font-size:12px;color:#555;margin-top:4px">'
                f'{METRIC_LABELS.get(m, m)}</div></div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Per-question ──
        st.subheader("Per-Question Scores")
        for i in sorted(scored):
            item = dataset[i]
            r = scored[i]
            with st.expander(f"**{item['id']}** — {item['question']}"):
                st.markdown("**Answer**")
                st.write(r["answer"])
                score_cols = st.columns(len(r["scores"]))
                for col, (m, val) in zip(score_cols, r["scores"].items()):
                    color = _score_color(val)
                    col.markdown(
                        f'<div style="text-align:center;padding:8px;border-radius:6px;'
                        f'background:{color}22;border:1px solid {color}">'
                        f'<div style="font-size:22px;font-weight:700;color:{color}">'
                        f'{"—" if val is None else f"{val:.3f}"}</div>'
                        f'<div style="font-size:11px;color:#555">'
                        f'{METRIC_LABELS.get(m, m)}</div></div>',
                        unsafe_allow_html=True,
                    )


if __name__ == "__main__":
    # streamlit run app/ui/eval_ui.py
    pass
