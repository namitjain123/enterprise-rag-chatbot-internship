"""
NeMo Guardrails orchestrator — four explicit stages matching the architecture:

  Stage 1  LLM CALL #1  self check input    NeMo input rail (Colang + prompts.yml)
  Stage 2  NO LLM       retrieve_context    ChromaDB hybrid retrieve → rerank
  Stage 3  LLM CALL #2  generate_answer     Groq LLM produces the final answer
  Stage 4  REGEX SCAN   scan_for_pii        NeMo output rail, no extra LLM call

Logfire spans wrap every stage so each request is fully observable:
  guardrail_pipeline
    ├── stage1_input_rail
    ├── stage2_retrieval
    ├── stage3_answer_generation
    └── stage4_pii_scan
"""

import re
import time
from pathlib import Path
from typing import Optional

import logfire
import nest_asyncio
from langchain_groq import ChatGroq
from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.actions import action

from app.core.config import GROQ_API_KEY, GUARD_MODEL, TOP_K
from app.llm.groq_client import query_groq
from app.rag.hybrid_retriever import hybrid_retrieve
from app.rag.reranker import rerank_chunks

nest_asyncio.apply()

# ── Logfire setup ─────────────────────────────────────────────────────────────
# send_to_logfire=False → structured logs go to the console only (no account needed).
# Set LOGFIRE_TOKEN in .env and remove send_to_logfire=False to stream to the cloud dashboard.
logfire.configure(
    service_name="enterprise-rag-guardrails",
    send_to_logfire="if-token-present",
)

NEMO_CONFIG_PATH = str(Path(__file__).parent / "nemo")

# PII pattern registry — name + compiled regex (name shown in logs when triggered).
_PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("SSN",         re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("credit_card", re.compile(r"\b\d{16}\b")),
    ("email",       re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("dollar",      re.compile(r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?")),
]


# ── Stage 2: ChromaDB retrieval ───────────────────────────────────────────────

@action(name="retrieve_context")
async def retrieve_context_action(query: str = "", context: Optional[dict] = None) -> str:
    if not query and context:
        query = context.get("last_user_message", "")

    with logfire.span("stage2_retrieval", query=query):
        t0 = time.perf_counter()
        chunks = hybrid_retrieve(query=query, vector_k=8, keyword_k=8)

        if not chunks:
            logfire.warn("stage2_retrieval: no chunks found", query=query)
            return ""

        best = rerank_chunks(query, chunks, top_n=TOP_K)
        elapsed = round((time.perf_counter() - t0) * 1000)

        logfire.info(
            "stage2_retrieval: done",
            query=query,
            chunks_before_rerank=len(chunks),
            chunks_after_rerank=len(best),
            latency_ms=elapsed,
        )
        return "\n\n".join(best)


# ── Stage 3: Answer generation ────────────────────────────────────────────────

@action(name="generate_answer")
async def generate_answer_action(
    query: str = "",
    retrieved_context: str = "",
    context: Optional[dict] = None,
) -> str:
    if not query and context:
        query = context.get("last_user_message", "")

    with logfire.span("stage3_answer_generation", query=query):
        if not retrieved_context:
            logfire.warn("stage3_answer_generation: empty context, returning fallback", query=query)
            return "I could not find relevant information in the indexed documents."

        t0 = time.perf_counter()
        answer = query_groq(query, retrieved_context)
        elapsed = round((time.perf_counter() - t0) * 1000)

        logfire.info(
            "stage3_answer_generation: done",
            query=query,
            context_chars=len(retrieved_context),
            answer_chars=len(answer),
            latency_ms=elapsed,
        )
        return answer


# ── Stage 4: PII scan ─────────────────────────────────────────────────────────

@action(name="scan_for_pii")
async def scan_for_pii_action(context: Optional[dict] = None) -> bool:
    response = ""
    if context:
        response = context.get("last_bot_message", "") or context.get("bot_message", "")

    with logfire.span("stage4_pii_scan"):
        triggered = [name for name, pattern in _PII_PATTERNS if pattern.search(response)]

        if triggered:
            logfire.warn(
                "stage4_pii_scan: PII detected — blocking response",
                patterns_matched=triggered,
                response_preview=response[:120],
            )
            return False   # unsafe → triggers output rail block

        logfire.info("stage4_pii_scan: clean", response_chars=len(response))
        return True        # safe → pass through


# ── Singleton LLMRails ────────────────────────────────────────────────────────

_rails: Optional[LLMRails] = None


def get_rails() -> LLMRails:
    global _rails
    if _rails is None:
        with logfire.span("nemo_rails_init"):
            config = RailsConfig.from_path(NEMO_CONFIG_PATH)
            llm = ChatGroq(
                model=GUARD_MODEL,
                api_key=GROQ_API_KEY,
                temperature=0.0,
                max_tokens=512,
            )
            rails = LLMRails(config, llm=llm)
            rails.register_action(retrieve_context_action, name="retrieve_context")
            rails.register_action(generate_answer_action,  name="generate_answer")
            rails.register_action(scan_for_pii_action,     name="scan_for_pii")
            _rails = rails
            logfire.info("nemo_rails_init: LLMRails singleton built", guard_model=GUARD_MODEL)
    return _rails


# ── Public entry point ────────────────────────────────────────────────────────

_INPUT_REFUSAL_MARKER  = "I can only answer questions about the uploaded HR policy"
_OUTPUT_REFUSAL_MARKER = "I've detected potentially sensitive personal information"


def generate_nemo_answer(query: str) -> dict:
    with logfire.span("guardrail_pipeline", query=query):
        try:
            rails = get_rails()

            with logfire.span("stage1_input_rail", query=query):
                t0 = time.perf_counter()
                raw = rails.generate(
                    messages=[{"role": "user", "content": query}]
                )
                elapsed = round((time.perf_counter() - t0) * 1000)
                response = raw["content"] if isinstance(raw, dict) else str(raw)

            # Detect which stage produced the final response.
            if _INPUT_REFUSAL_MARKER.lower() in response.lower():
                logfire.warn(
                    "guardrail_pipeline: BLOCKED at stage1",
                    query=query,
                    latency_ms=elapsed,
                    llm_calls=1,
                )
                return {
                    "answer": response,
                    "sources": [],
                    "guard": {"stage": "input_rail", "llm_calls": 1,
                              "detail": "Stage 1 — self check input blocked this request"},
                }

            if _OUTPUT_REFUSAL_MARKER.lower() in response.lower():
                logfire.warn(
                    "guardrail_pipeline: BLOCKED at stage4",
                    query=query,
                    latency_ms=elapsed,
                    llm_calls=2,
                )
                return {
                    "answer": response,
                    "sources": [],
                    "guard": {"stage": "output_rail", "llm_calls": 2,
                              "detail": "Stage 4 — PII pattern detected in response"},
                }

            logfire.info(
                "guardrail_pipeline: passed all stages",
                query=query,
                latency_ms=elapsed,
                llm_calls=2,
                answer_chars=len(response),
            )
            return {
                "answer": response,
                "sources": [],
                "guard": {"stage": "passed", "llm_calls": 2,
                          "detail": "All four stages passed"},
            }

        except Exception as exc:
            logfire.error("guardrail_pipeline: error, falling back to raw RAG", error=str(exc), query=query)
            from app.rag.retriever import generate_rag_answer
            result = generate_rag_answer(query)
            result["guard"] = {"stage": "error", "llm_calls": 1,
                               "detail": f"NeMo error, fell back to raw RAG: {exc}"}
            return result
