"""
LLM Service — Ollama-backed
===========================
Uses a local Ollama instance (qwen2.5:3b) for all LLM calls.

Answer Modes:
  DOCUMENT  — retrieved evidence strongly answers the query.
  HYBRID    — evidence is partial; combined with LLM knowledge.
  KNOWLEDGE — evidence is weak/absent; pure LLM knowledge.
"""

import logging
import requests
import json
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"


def _call_llm(prompt: str, json_format: bool = False) -> str:
    """Call local Ollama instance."""
    try:
        payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
        if json_format:
            payload["format"] = "json"

        resp = requests.post(
            OLLAMA_ENDPOINT,
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()
    except Exception as e:
        logger.error(f"Ollama call failed: {e}")
        return "{}" if json_format else "I was unable to generate an answer at this time."


# ─────────────────────────────────────────────────────────────────────────── #


class LLMService:

    # ── QUERY DECOMPOSITION ───────────────────────────────────────────────── #

    @staticmethod
    def decompose_query(query: str) -> list:
        """
        Decompose a multi-concept user query into individual concepts.
        Returns a list of concept strings (e.g. ["Artificial Intelligence", "Agriculture"]).
        Falls back to [query] if decomposition fails or returns only one concept.
        """
        prompt = f"""Extract all independent concepts requested by the user.
Return JSON only. No explanation. No extra text.

Example:
Question: What is AI and what is Agriculture?
Output: {{"concepts": ["Artificial Intelligence", "Agriculture"]}}

Example:
Question: Explain machine learning
Output: {{"concepts": ["Machine Learning"]}}

Question: {query}
Output:"""
        try:
            raw = _call_llm(prompt, json_format=True)
            parsed = json.loads(raw)
            concepts = parsed.get("concepts", [])
            if isinstance(concepts, list) and len(concepts) >= 1:
                return [c.strip() for c in concepts if c.strip()]
        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}")
        return [query]

    # ── DOCUMENT MODE ────────────────────────────────────────────────────── #

    @staticmethod
    def generate_document_answer(
        query: str, context: str, is_definition_query: bool = False
    ) -> str:
        """
        DOCUMENT MODE: Answer strictly from retrieved document evidence.
        Used when evidence strongly answers the query (high rerank score).
        Returns 'NOT_FOUND' if the context does not contain the answer.
        """
        if is_definition_query:
            prompt = f"""You are a document search assistant.

Answer the definition question using ONLY the provided document evidence.

Output Format:
[Topic Name]
[Definition]

Rules:
- Answer each concept separately.
- Do not merge concepts.
- Do not replace concepts with related concepts (e.g. if asked about Agriculture, do NOT answer with Precision Farming or Smart Agriculture).
- Use only information from the document evidence below.
- Do NOT add extra commentary, trends, or applications unless explicitly asked.
- If the document does not contain the definition, respond with exactly: NOT_FOUND

Question: {query}

Document Evidence:
{context}

Answer:"""
        else:
            prompt = f"""You are a document search assistant.

Answer the question using ONLY the provided document evidence.

Rules:
- Answer each concept separately.
- Do not merge concepts.
- Do not replace concepts with related concepts (e.g. if asked about Agriculture, do NOT answer with Precision Farming or Smart Agriculture).
- If the answer is not in the evidence, respond with exactly: NOT_FOUND

Question: {query}

Document Evidence:
{context}

Answer:"""
        return _call_llm(prompt)

    # ── HYBRID MODE ──────────────────────────────────────────────────────── #

    @staticmethod
    def generate_hybrid_answer(
        query: str, context: str, is_definition_query: bool = False
    ) -> str:
        """
        HYBRID MODE: Combine document evidence with LLM general knowledge.
        Used when retrieved evidence is relevant but incomplete.
        For definition queries, LLM provides a base definition first,
        then enriches it with document-specific details.
        """
        if is_definition_query:
            prompt = f"""You are a knowledgeable AI assistant with access to document evidence.

Answer the definition question. Follow this approach:
1. Provide a clear, accurate definition using your general knowledge.
2. If the document evidence below adds relevant context, enrich the definition with it.

Output Format:
[Topic Name]
[Definition — based on general knowledge, enriched with document evidence where relevant]

Rules:
- Answer each concept separately.
- Do not merge concepts.
- Do not replace concepts with related concepts (e.g. if asked about Agriculture, do NOT answer with Precision Farming or Smart Agriculture).

Question: {query}

Document Evidence:
{context}

Answer:"""
        else:
            prompt = f"""You are a helpful AI assistant with access to document evidence.

Answer the question by combining the provided document evidence with your general knowledge.

Rules:
- Answer each concept separately.
- Do not merge concepts.
- Do not replace concepts with related concepts (e.g. if asked about Agriculture, do NOT answer with Precision Farming or Smart Agriculture).
- Use the document evidence as the primary source where it is relevant.
- Supplement with your general knowledge where the documents are incomplete.
- Be accurate and complete.

Question: {query}

Document Evidence:
{context}

Answer:"""
        return _call_llm(prompt)

    # ── KNOWLEDGE MODE ───────────────────────────────────────────────────── #

    @staticmethod
    def generate_knowledge_answer(
        query: str, is_definition_query: bool = False
    ) -> str:
        """
        KNOWLEDGE MODE: Answer using only LLM general knowledge.
        Used when retrieval confidence is low or no documents are uploaded.
        """
        if is_definition_query:
            prompt = f"""You are a knowledgeable AI assistant.

Answer the definition question clearly and accurately using your general knowledge.

Output Format:
[Topic Name]
[Definition]

If multiple topics are asked, provide each separately in the same format.

Question: {query}

Answer:"""
        else:
            prompt = f"""You are a knowledgeable AI assistant.
Answer the following question clearly and concisely using your general knowledge.

Question: {query}

Answer:"""
        return _call_llm(prompt)

    # ── DOCUMENT INTELLIGENCE ─────────────────────────────────────────────── #

    @staticmethod
    def extract_document_intelligence(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extracts document intelligence (summary and topics) using Ollama JSON format.
        """
        sample_chunks = []
        char_count = 0
        taken = 0
        step = max(1, len(chunks) // 15)
        for i in range(0, len(chunks), step):
            if taken >= 30:
                break
            text = chunks[i]["text"]
            if char_count + len(text) < 30000:
                char_count += len(text)
                sample_chunks.append(text)
                taken += 1

        context = "\n\n---\n\n".join(sample_chunks)

        prompt = f"""You are a high-level document intelligence AI.

Analyze the provided document content and extract its core subject and key terminology.

Topic Rules:
- Extract exactly 30 distinct noun phrases that represent specific concepts or terminology discussed in this document.
- Ensure these are concrete semantic noun phrases (e.g. "Precision Farming", "Agricultural Modernization", "Climate Smart Agriculture").
- Do NOT use vague single words (like "Future", "Technology").
- Return only the raw noun phrases.

Summary Rules:
- Generate a concise summary of this document.
- Include: Main subject, Important concepts, Applications, Key conclusions.
- Keep the summary to 3-5 sentences.

Return JSON only using the exact format:
{{
  "summary": "...",
  "topics": [
    "...",
    "..."
  ]
}}

Document Content:
{context}
"""
        raw_json = _call_llm(prompt, json_format=True)
        try:
            parsed = json.loads(raw_json)
            if "summary" not in parsed:
                parsed["summary"] = ""
            if "topics" not in parsed:
                parsed["topics"] = []
            return parsed
        except Exception as e:
            logger.error(f"Failed to parse LLM JSON: {e}")
            return {"summary": "", "topics": []}

    # ── RELATED QUESTIONS ─────────────────────────────────────────────────── #

    @staticmethod
    def generate_related_questions(
        query: str, context: str
    ) -> List[str]:
        """
        Generate 3–5 related questions based on the query and document context.
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