from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx
import ollama


class GemmaUnavailableError(RuntimeError):
    pass


class FastGemma:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("GEMMA_FAST_MODEL", "gemma4:e2b")
        self.host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        self.client = ollama.Client(host=self.host, timeout=int(os.getenv("GEMMA_FAST_TIMEOUT", "240")))
        self.keep_alive = os.getenv("GEMMA_KEEP_ALIVE", "30m")
        self.fast_options = {
            "temperature": 0.1,
            "num_predict": int(os.getenv("GEMMA_FAST_NUM_PREDICT", "600")),
            "num_ctx": int(os.getenv("GEMMA_FAST_NUM_CTX", "2048")),
            "top_p": 0.9,
        }
        self.reasoning_options = {
            "temperature": 0.2,
            "num_predict": int(os.getenv("GEMMA_REASONING_NUM_PREDICT", "1000")),
            "num_ctx": int(os.getenv("GEMMA_REASONING_NUM_CTX", "4096")),
        }
        if os.getenv("GEMMA_NUM_THREAD"):
            self.fast_options["num_thread"] = int(os.environ["GEMMA_NUM_THREAD"])
            self.reasoning_options["num_thread"] = int(os.environ["GEMMA_NUM_THREAD"])

    def status(self) -> dict[str, Any]:
        try:
            response = httpx.get(f"{self.host.rstrip('/')}/api/tags", timeout=5)
            response.raise_for_status()
            models = [model.get("name") for model in response.json().get("models", [])]
            return {
                "status": "reachable",
                "model": self.model,
                "host": self.host,
                "keep_alive": self.keep_alive,
                "model_present": self.model in models,
                "available_models": models,
            }
        except Exception as exc:
            return {
                "status": "unreachable",
                "model": self.model,
                "host": self.host,
                "keep_alive": self.keep_alive,
                "model_present": False,
                "available_models": [],
                "error": str(exc),
                "hint": "Start Ollama locally or set OLLAMA_HOST to your friend's GPU Ollama URL before starting the fast backend.",
            }

    def structured(self, prompt: str, fast: bool = True) -> dict[str, Any]:
        options = self.fast_options if fast else self.reasoning_options
        response = self._generate(prompt, options)
        return self._parse(response.get("response", ""))

    def text(self, prompt: str) -> str:
        response = self._generate(prompt, self.fast_options)
        return response.get("response", "").strip()

    def warmup(self) -> dict[str, Any]:
        response = self.text("What is 2+2? Answer only the number.")
        return {"model": self.model, "host": self.host, "response": response, "ok": response.strip() == "4"}

    def summarize(self, abstract: str, conclusion: str) -> dict[str, Any]:
        prompt = f"""Analyze this scientific paper. Be concise.

ABSTRACT: {abstract[:800]}
CONCLUSION: {conclusion[:600]}

OUTPUT VALID JSON ONLY:
{{
  "tldr": "One sentence.",
  "contribution": "What is new.",
  "key_findings": ["finding 1", "finding 2", "finding 3"],
  "limitations": ["limit 1"],
  "future_directions": ["direction 1"]
}}"""
        return self.structured(prompt)

    def extract_entities(self, text: str) -> dict[str, Any]:
        prompt = f"""Extract scientific entities from this text.

TEXT: {text[:1200]}

OUTPUT VALID JSON ONLY:
{{
  "entities": {{
    "DISEASE": [],
    "PROTEIN": [],
    "GENE": [],
    "CHEMICAL": [],
    "METHOD": [],
    "CONCEPT": []
  }},
  "relationships": [
    {{"source": "A", "relation": "inhibits|treats|causes|correlates_with|uses|improves", "target": "B", "confidence": 0.8}}
  ]
}}"""
        return self.structured(prompt)

    def generate_hypotheses(self, query: str, chunks: list[str], num: int = 3) -> dict[str, Any]:
        context = "\n---\n".join(chunks[:5])[:2500]
        prompt = f"""You are an AI research scientist. Generate {num} novel hypotheses.

QUERY: {query}

EVIDENCE FROM PAPERS:
{context}

Rules:
- Each hypothesis must NOT be stated in the evidence above
- Must be testable
- Must be grounded in the evidence

OUTPUT VALID JSON ONLY:
{{
  "hypotheses": [
    {{
      "id": 1,
      "hypothesis": "Clear statement.",
      "reasoning": "Why this follows from the evidence.",
      "confidence": 0.75,
      "novelty": 0.85,
      "experiment": "How to test this.",
      "evidence_used": ["brief quote from context"]
    }}
  ],
  "key_insight": "One cross-domain insight from combining papers."
}}"""
        return self.structured(prompt, fast=False)

    def detect_contradiction(self, text_a: str, text_b: str, title_a: str, title_b: str) -> dict[str, Any]:
        prompt = f"""Compare these two research excerpts.

PAPER A "{title_a[:60]}":
{text_a[:900]}

PAPER B "{title_b[:60]}":
{text_b[:900]}

OUTPUT VALID JSON ONLY:
{{
  "has_contradiction": false,
  "severity": "LOW|MEDIUM|HIGH",
  "paper_a_claim": "...",
  "paper_b_claim": "...",
  "explanation": "...",
  "resolution": "..."
}}"""
        return self.structured(prompt)

    def find_gaps(self, context: str, topic: str) -> list[str]:
        prompt = f"""What is NOT studied in this research on "{topic}"?

CONTEXT: {context[:1500]}

OUTPUT VALID JSON ONLY:
{{"gaps": ["gap 1", "gap 2", "gap 3", "gap 4", "gap 5"]}}"""
        result = self.structured(prompt)
        return result.get("gaps", [])

    def _generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.client.generate(
                model=self.model,
                prompt=prompt,
                options=options,
                keep_alive=self.keep_alive,
            )
        except Exception as exc:
            raise GemmaUnavailableError(
                f"Could not reach Ollama at {self.host} for model {self.model}. "
                "Start Ollama, pull the model, or set OLLAMA_HOST to the correct GPU server."
            ) from exc

    def _parse(self, raw: str) -> dict[str, Any]:
        cleaned = re.sub(r"```json|```", "", raw).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
        return {"raw": raw, "error": "parse_failed"}
