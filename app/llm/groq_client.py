import os

from groq import Groq


# Combines user query and context, sends it to Groq LLM, returns response.
def query_groq(query, context, model_name: str | None = None):
    prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{query}

Answer:
""".strip()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Error: GROQ_API_KEY is not set. Add it to your .env or environment variables."

    model = model_name or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    try:
        client = Groq(api_key=api_key)

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful enterprise document Q&A assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        return resp.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"