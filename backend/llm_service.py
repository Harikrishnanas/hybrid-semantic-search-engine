"""
LLM Service — Ollama-backed
===========================
Uses a local Ollama instance (qwen2.5:3b) for all LLM calls.
"""

import logging
import requests
from typing import List, Dict, Any, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"


def _call_llm(prompt: str) -> str:
    """Call local Ollama instance."""
    try:
        resp = requests.post(
            OLLAMA_ENDPOINT,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()
    except Exception as e:
        logger.error(f"Ollama call failed: {e}")
        return "I was unable to generate an answer at this time."


# ─────────────────────────────────────────────────────────────────────────── #


class LLMService:

    @staticmethod
    def generate_answer(query: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Answer a query strictly from provided document chunks.
        Returns the string 'NOT_FOUND' if the context doesn't contain the answer.
        """
        context = "\n\n---\n\n".join(chunk["text"] for chunk in chunks)
        prompt = f"""You are a precise document analysis assistant.

Answer the following question ONLY using the provided document context below.
Be concise and factual. Do NOT add external knowledge.

If the answer cannot be found in the context, reply EXACTLY with the single word:
NOT_FOUND

Question: {query}

Document Context:
{context}

Answer:"""
        return _call_llm(prompt)

    @staticmethod
    def generate_general_answer(query: str) -> str:
        """Answer using AI general knowledge (no document context)."""
        prompt = f"""You are a knowledgeable AI assistant. 
Answer the following question clearly and concisely using your general knowledge.

Question: {query}

Answer:"""
        return _call_llm(prompt)

    @staticmethod
    def generate_related_questions(
        query: str, context: str
    ) -> List[str]:
        """
        Generate 3–5 related questions based on the query and document context.
        Returns a list of question strings.
        """
        prompt = f"""Based on the following question and document context, generate 4 related follow-up questions that a researcher might ask next. 
Return ONLY the questions, one per line, without numbering or bullet points.

Original Question: {query}

Context:
{context[:1500]}

Related Questions (4, one per line):"""
        try:
            raw = _call_llm(prompt)
            lines = [
                line.strip().lstrip("•-*0123456789.) ")
                for line in raw.strip().splitlines()
                if line.strip() and "?" in line
            ]
            return lines[:5]
        except Exception:
            return []

    @staticmethod
    def generate_hybrid_answer(
        query: str,
        chunks: List[Dict[str, Any]],
        coverage_score: float,
    ) -> Tuple[str, str]:
        """
        Generate an answer that may blend document evidence with AI knowledge
        depending on the coverage score.

        Returns (answer_text, source_type) where source_type is 'doc'|'hybrid'|'ai'.
        """
        if coverage_score >= 40:
            answer = LLMService.generate_answer(query, chunks)
            if answer == "NOT_FOUND" or not answer.strip():
                answer = LLMService.generate_general_answer(query)
                return answer, "ai"
            return answer, "doc" if coverage_score >= 70 else "hybrid"
        else:
            answer = LLMService.generate_general_answer(query)
            return answer, "ai"