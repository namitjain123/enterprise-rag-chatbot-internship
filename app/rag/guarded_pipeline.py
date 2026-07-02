"""
Guarded RAG orchestrator — routes through NeMo Guardrails.

Every message flows through NeMo's pipeline:

    Stage 1  LLM CALL #1  self check input  (NeMo input rail)
    Stage 2  LLM CALL #2  rag_answer action (hybrid retrieve → rerank → Groq)
    Stage 3  REGEX SCAN   scan_for_pii      (NeMo output rail, no extra LLM)

A blocked message short-circuits after LLM call #1 (input rail).
Set GUARDRAILS_ENABLED=false in .env to bypass NeMo entirely (1 call, raw RAG).
"""

from app.core.config import GUARDRAILS_ENABLED
from app.rag.retriever import generate_rag_answer


def generate_guarded_answer(query: str, top_k: int | None = None) -> dict:
    """Run the full NeMo-guarded pipeline. Falls back to raw RAG if disabled."""
    if not GUARDRAILS_ENABLED:
        result = generate_rag_answer(query) if top_k is None else generate_rag_answer(query, top_k)
        result["guard"] = {"stage": "DISABLED", "llm_calls": 1, "detail": "guardrails disabled"}
        return result

    # Import lazily so NeMo (heavy) only loads when guardrails are actually enabled.
    from app.guardrails.nemo_pipeline import generate_nemo_answer
    return generate_nemo_answer(query)
