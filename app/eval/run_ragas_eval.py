"""
Evaluate the RAG pipeline with RAGAS, using Groq as the judge LLM.

For every question in eval_dataset.json we:
  1. run the EXISTING RAG pipeline (hybrid retrieve -> rerank -> Groq answer),
  2. collect the generated answer and the retrieved contexts,
  3. score them against the human-written ground_truth using RAGAS metrics.

RAGAS normally defaults to OpenAI for its judge LLM/embeddings. We override that
with Groq (langchain-groq) and the local all-MiniLM-L6-v2 embeddings, so the only
API key required is GROQ_API_KEY.

CLI usage (run from project root, with the venv active):
    python -m app.eval.ingest_fixed_doc      # once, to index the fixed PDF
    python -m app.eval.run_ragas_eval        # run the evaluation
"""

import json
import os

import numpy as np
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.metrics import (
    AnswerCorrectness,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from app.core.config import EMBEDDING_MODEL, GROQ_MODEL, JUDGE_GROQ_API_KEY
from app.rag.retriever import generate_rag_answer

DATASET_PATH = os.path.join(os.path.dirname(__file__), "eval_dataset.json")


def _build_judge():
    """Configure RAGAS to use Groq (LLM) + local MiniLM (embeddings) instead of OpenAI."""
    if not JUDGE_GROQ_API_KEY:
        raise RuntimeError(
            "Neither JUDGE_GROQ nor GROQ_API_KEY is set. Add one to your .env before evaluating."
        )

    judge_llm = LangchainLLMWrapper(
        ChatGroq(model=GROQ_MODEL, temperature=0.0, api_key=JUDGE_GROQ_API_KEY)
    )
    hf_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    judge_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)
    # Return hf_embeddings separately so retrieval metrics can use it directly.
    return judge_llm, judge_embeddings, hf_embeddings


def _cosine_sim(a: list, b: list) -> float:
    va, vb = np.array(a), np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-10))


def _compute_retrieval_metrics(
    samples: list,
    items: list[dict],
    hf_embeddings: HuggingFaceEmbeddings,
    threshold: float = 0.50,
) -> list[dict]:
    """
    Compute Hit Rate and MRR per question using embedding cosine similarity.

    For each question:
      - Embed ground_truth and every retrieved chunk.
      - A chunk is "relevant" if cosine_sim(chunk, ground_truth) >= threshold.
      - Hit Rate = 1 if any chunk is relevant, else 0.
      - MRR      = 1 / rank_of_first_relevant_chunk  (0 if none found).

    Returns a list of dicts with keys: hit_rate, mrr.
    """
    results = []
    for sample, item in zip(samples, items):
        contexts = sample.retrieved_contexts or []
        ground_truth = item["ground_truth"]

        if not contexts:
            results.append({"hit_rate": 0.0, "mrr": 0.0})
            continue

        gt_vec = hf_embeddings.embed_query(ground_truth)
        ctx_vecs = hf_embeddings.embed_documents(contexts)

        first_hit_rank = None
        for rank, ctx_vec in enumerate(ctx_vecs, start=1):
            if _cosine_sim(gt_vec, ctx_vec) >= threshold:
                first_hit_rank = rank
                break

        results.append({
            "hit_rate": 1.0 if first_hit_rank else 0.0,
            "mrr":      1.0 / first_hit_rank if first_hit_rank else 0.0,
        })

    return results


def _load_dataset(indices: list[int] | None = None) -> list[dict]:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if indices is not None:
        data = [data[i] for i in indices if 0 <= i < len(data)]
    return data


def _collect_samples(items: list[dict]) -> list[SingleTurnSample]:
    """Run the real RAG pipeline for each question and package results for RAGAS."""
    samples = []
    for i, item in enumerate(items, start=1):
        question = item["question"]
        ground_truth = item["ground_truth"]

        result = generate_rag_answer(question)
        answer = result.get("answer", "")
        contexts = result.get("sources", [])  # the reranked chunks fed to the LLM

        print(f"[{i}/{len(items)}] {question[:70]}")

        samples.append(
            SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
                reference=ground_truth,
            )
        )
    return samples


def run_evaluation(indices: list[int] | None = None) -> dict:
    """
    Run the full RAGAS evaluation and return a JSON-friendly dict:
        {
          "aggregate": {metric: mean_score, ...},
          "per_question": [ {question, answer, ...scores}, ... ]
        }
    """
    judge_llm, judge_embeddings, hf_embeddings = _build_judge()

    metrics = [
        Faithfulness(),                       # answer grounded in retrieved context?
        ResponseRelevancy(),                  # answer addresses the question?
        LLMContextPrecisionWithReference(),   # were retrieved chunks relevant?
        LLMContextRecall(),                   # did retrieval find what the answer needed?
        AnswerCorrectness(),                  # factual correctness vs ground truth?
    ]

    items = _load_dataset(indices)
    samples = _collect_samples(items)
    dataset = EvaluationDataset(samples=samples)

    # Throttle the judge calls: Groq's free tier rate-limits RAGAS's default
    # high parallelism, which shows up as TimeoutError -> missing (NaN) scores.
    run_config = RunConfig(timeout=180, max_workers=2)

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=run_config,
    )

    df = result.to_pandas()
    metric_cols = [c for c in df.columns if c not in ("user_input", "response", "retrieved_contexts", "reference")]

    aggregate = {col: float(df[col].mean(skipna=True)) for col in metric_cols}

    retrieval_scores = _compute_retrieval_metrics(samples, items, hf_embeddings)

    per_question = []
    for (_, row), ret in zip(df.iterrows(), retrieval_scores):
        entry = {
            "question": row["user_input"],
            "answer": row["response"],
        }
        for col in metric_cols:
            val = row[col]
            entry[col] = None if val is None or (isinstance(val, float) and val != val) else float(val)
        entry["hit_rate"] = ret["hit_rate"]
        entry["mrr"] = ret["mrr"]
        per_question.append(entry)

    aggregate["hit_rate"] = float(np.mean([r["hit_rate"] for r in retrieval_scores]))
    aggregate["mrr"] = float(np.mean([r["mrr"] for r in retrieval_scores]))

    return {"aggregate": aggregate, "per_question": per_question}


def main() -> None:
    report = run_evaluation()

    print("\n================ RAGAS RESULTS (aggregate) ================")
    for metric, score in report["aggregate"].items():
        print(f"{metric:35s} {score:.4f}")

    print("\n---------------- per question ----------------")
    for row in report["per_question"]:
        print(f"\nQ: {row['question']}")
        for k, v in row.items():
            if k not in ("question", "answer"):
                print(f"   {k:33s} {v if v is None else round(v, 4)}")


if __name__ == "__main__":
    main()
