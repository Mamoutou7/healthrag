# Created by Mamoutou Fofana
# Date: 10/25/2025


from ..config import LLM_PROVIDER, OPENAI_API_KEY

PROMPT_TEMPLATE = """
You are a medical assistant. Provide a concise,
evidence-based answer **using only the context provided**
from trusted sources (PubMed, DrugBank, official guidelines).
If the context does not contain the answer,
say: "I don't know based on the provided context."

Question:
{question}

Context:
{context}

Answer:"""

try:
    import openai
except ImportError:
    openai = None


def build_prompt(question: str, context: str) -> str:
    return PROMPT_TEMPLATE.format(question=question, context=context)


def generate_answer(question: str, context: str, provider=LLM_PROVIDER) -> str:
    prompt = build_prompt(question, context)

    if provider == "openai":
        if openai is None:
            return "[OpenAI module not installed]"
        try:
            openai.api_key = OPENAI_API_KEY
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                    "role": "system",
                    "content": "You are an assistant that must only use the provided context.",
                    },
                    {
                    "role": "user",
                    "content": prompt
                    },
                ],
                temperature=0.0,
                max_tokens=512,
            )
            return resp["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"[OpenAI call failed: {e}]"
    else:
        snippet = context[:1000]
        return f"(DEMO ANSWER) Based on retrieved evidence:\n\n{snippet}\n\nNote: enable LLM_PROVIDER=openai and set OPENAI_API_KEY to generate polished answers."
