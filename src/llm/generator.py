# Created by Mamoutou Fofana
# Date: 10/25/2025


import os
import time
import logging
from typing import Optional
from dataclasses import dataclass

import openai
from ..config import LLM_PROVIDER, OPENAI_API_KEY

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

Answer:
"""


@dataclass
class GenerationConfig:
    """Configuration for response generation"""
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.0
    max_tokens: int = 512
    top_p: float = 1.0
    retries: int = 3
    timeout: int = 30              # seconds




class OpenAIGenerator:
    """Thin wrapper around the new OpenAI v1 client with retry & safety."""

    def __init__(self, config:Optional[GenerationConfig] = None):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY must be set in the environment or config.")

        openai.api_key = OPENAI_API_KEY
        self.config = config or GenerationConfig()
        logger.info(f"OpenAIGenerator initialized - model={self.config.model}")


    def _call(self, prompt: str) -> str:
        backoff = 1
        for attempt in range(1, self.config.retries + 1):
            try:
                resp = openai.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": "You are a medical assistant. Use only the provided context."},
                        {"role": "user",   "content": prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    top_p=self.config.top_p,
                    timeout=self.config.timeout,
                )
                return resp.choices[0].message.content.strip()

            except openai.Timeout:
                log.warning(f"OpenAI timeout (attempt {attempt}/{self.config.retries})")
            except openai.RateLimitError:
                log.warning(f"OpenAI rate-limit (attempt {attempt}/{self.config.retries})")
            except openai.APIError as e:
                log.error(f"OpenAI API error: {e}")
            except Exception as e:
                log.exception(f"Unexpected OpenAI error: {e}")
                raise

            if attempt < self.config.retries:
                time.sleep(backoff)
                backoff *= 2

        return "[OpenAI call failed after retries]"


    def generate(self, question: str, context: str) -> str:
            """
            Generate an answer using **only** the supplied context.

            Args:
                question: User question (string)
                context:  Retrieved evidence (string)

            Returns:
                Final answer (or the “I don't know…”).
            """
            if not isinstance(question, str) or not question.strip():
                return "Error: question must be a non-empty string."
            if not isinstance(context, str):
                return "I don't know based on the provided context."

            q = question.strip()
            c = context.strip() or " "   # avoid empty context → hallucination

            prompt = PROMPT_TEMPLATE.format(question=q, context=c)
            return self._call(prompt)


def generate_answer(question: str, context: str, **kwargs) -> str:
    """
    Drop-in replacement for the original function.

    Extra kwargs are forwarded to ``GenerationConfig`` (model, temperature, …).
    """
    cfg_kwargs = {k: v for k, v in kwargs.items() if hasattr(GenerationConfig, k)}
    config = GenerationConfig(**cfg_kwargs)
    gen = OpenAIGenerator(config)
    return gen.generate(question, context)




if __name__ == "__main__":
    q = "What is the mechanism of action of Remdesivir?"
    c = "Remdesivir is a nucleotide analog prodrug that inhibits viral RNA polymerase."
    print(generate_answer(q, c))