import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# Separate key for RAGAS judge LLM so eval runs don't exhaust the chatbot's quota.
# Falls back to GROQ_API_KEY if JUDGE_GROQ is not set.
JUDGE_GROQ_API_KEY = os.getenv("JUDGE_GROQ", "") or GROQ_API_KEY
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
# Model used by the input guardrail classifier (LLM call #1). Defaults to a fast,
# cheap model since the guard only needs to emit a short JSON verdict, not prose.
GUARD_MODEL = os.getenv("GUARD_MODEL", "llama-3.1-8b-instant")
# Master switch — set GUARDRAILS_ENABLED=false in .env to bypass the gate entirely.
GUARDRAILS_ENABLED = os.getenv("GUARDRAILS_ENABLED", "true").lower() == "true"

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_store")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "enterprise_docs")

TOP_K = int(os.getenv("TOP_K", 3))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")