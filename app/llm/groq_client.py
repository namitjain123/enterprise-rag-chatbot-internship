import os
from collections.abc import Generator

from groq import Groq

_SYSTEM_PROMPT = "You are a helpful enterprise document Q&A assistant."


def _build_prompt(query: str, context: str) -> str:
    return f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"


def _get_client() -> tuple[Groq, str]:
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set.")
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    return Groq(api_key=api_key), model


def query_groq(query: str, context: str, model_name: str | None = None) -> str:
    """Blocking call — returns the full answer string."""
    try:
        client, model = _get_client()
        resp = client.chat.completions.create(
            model=model_name or model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_prompt(query, context)},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


def query_groq_stream(query: str, context: str, model_name: str | None = None) -> Generator[str, None, None]:
    """Yields response tokens one-by-one from Groq's streaming API."""
    try:
        client, model = _get_client()
        stream = client.chat.completions.create(
            model=model_name or model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_prompt(query, context)},
            ],
            temperature=0.2,
            stream=True,
        )
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield token
    except Exception as e:
        yield f"Error: {str(e)}"
