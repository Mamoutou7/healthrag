# Created by Mamoutou Fofana
# Date: 10/25/2025
# Updated: 10/26/2025 – Aligned with OpenAI Python Quickstart (v1+)

import time
import logging
from typing import Optional
from dataclasses import dataclass

from openai import OpenAI, Timeout, RateLimitError, APIError
from ..config import OPENAI_API_KEY

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Template strict pour RAG
PROMPT_TEMPLATE = """You are a medical assistant.
Your role is to provide **concise, accurate, and evidence-based answers** 
using **only the information in the provided context**.

Rules:
- Use only the context below. Do **not** hallucinate or add external knowledge.
- Cite sources if possible (e.g., "According to PubMed study X...").
- If the context does not contain the answer, respond exactly: "I don't know based on the provided context."
- Be professional, clear, and concise.

Question:
{question}

Context:
{context}

Answer:"""


@dataclass
class GenerationConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 512
    top_p: float = 1.0
    retries: int = 3
    timeout: int = 30  # seconds


class OpenAIGenerator:
    """Wrapper moderne autour du client OpenAI v1+ avec retry et logs."""

    def __init__(self, config: Optional[GenerationConfig] = None):
        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is missing! "
                "Set it in environment or .env file: OPENAI_API_KEY=sk-..."
            )

        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.config = config or GenerationConfig()
        logger.info(f"OpenAIGenerator initialized – model='{self.config.model}'")

    def _call(self, prompt: str) -> str:
        """Appel OpenAI avec retry exponentiel."""
        backoff = 1
        for attempt in range(1, self.config.retries + 1):
            try:
                logger.debug(f"Calling OpenAI API (attempt {attempt})...")
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": "You are a medical assistant. Use only the provided context."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    top_p=self.config.top_p,
                    timeout=self.config.timeout,
                )
                answer = response.choices[0].message.content.strip()
                logger.info("OpenAI response received successfully.")
                return answer

            except Timeout:
                logger.warning(f"Timeout (attempt {attempt}/{self.config.retries})")
            except RateLimitError:
                logger.warning(f"Rate limit hit (attempt {attempt}/{self.config.retries})")
            except APIError as e:
                logger.error(f"OpenAI API error: {e}")
            except Exception as e:
                logger.exception(f"Unexpected error: {e}")
                raise

            # Attente avant retry
            if attempt < self.config.retries:
                logger.info(f"Retrying in {backoff}s...")
                time.sleep(backoff)
                backoff *= 2

        return "[Failed: OpenAI unreachable after retries]"

    def generate(self, question: str, context: str) -> str:
        """Génère une réponse à partir du contexte uniquement."""
        if not isinstance(question, str) or not question.strip():
            return "Error: question must be a non-empty string."
        if not isinstance(context, str):
            return "I don't know based on the provided context."

        q = question.strip()
        c = context.strip() or " "

        prompt = PROMPT_TEMPLATE.format(question=q, context=c)
        logger.debug(f"Prompt sent to LLM:\n{'='*50}\n{prompt}\n{'='*50}")

        return self._call(prompt)


# Fonction publique – appelée par l'API
def generate_answer(question: str, context: str, **kwargs) -> str:
    """
    Génère une réponse RAG avec LLM.
    Args:
        question: Question utilisateur
        context: Contexte récupéré (FAISS + KG)
        **kwargs: model, temperature, etc.
    """
    logger.info("=== generate_answer STARTED ===")
    logger.info(f"Question: {question}")
    logger.info(f"Context length: {len(context)} characters")

    # Filtrer les kwargs valides
    cfg_kwargs = {
        k: v for k, v in kwargs.items()
        if k in {"model", "temperature", "max_tokens", "top_p", "retries", "timeout"}
    }

    config = GenerationConfig(**cfg_kwargs)

    try:
        generator = OpenAIGenerator(config)
        logger.info("OpenAIGenerator ready")

    except Exception as e:
        logger.error(f"Failed to initialize OpenAIGenerator: {e}", exc_info=True)
        return "I don't know based on the provided context."

    try:
        answer = generator.generate(question, context)
        print(answer)
        logger.info(f"Answer generated ({len(answer)} chars): {answer[:120]}{'...' if len(answer) > 120 else ''}")
        return answer
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        return "I don't know based on the provided context."

if __name__ == "__main__":
    q = "What is the Metastasis?"
    c = "Remdesivir is a nucleotide analog prodrug that inhibits viral RNA polymerase."
    print(generate_answer(q, c))