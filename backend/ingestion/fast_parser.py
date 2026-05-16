from __future__ import annotations

import re
from typing import Any

import fitz
import httpx


HIGH_VALUE_SECTIONS = {
    "abstract": 1.0,
    "conclusion": 0.95,
    "results": 0.90,
    "discussion": 0.80,
    "introduction": 0.60,
    "methods": 0.50,
}


class FastPaperParser:
    max_pages = 20

    def parse_bytes(self, file_bytes: bytes) -> dict[str, Any]:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        try:
            return self._extract(doc)
        finally:
            doc.close()

    def parse_path(self, path: str) -> dict[str, Any]:
        doc = fitz.open(path)
        try:
            return self._extract(doc)
        finally:
            doc.close()

    async def parse_arxiv(self, arxiv_id: str) -> dict[str, Any]:
        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url)
            response.raise_for_status()
        parsed = self.parse_bytes(response.content)
        parsed["arxiv_id"] = arxiv_id
        return parsed

    def _extract(self, doc: fitz.Document) -> dict[str, Any]:
        result: dict[str, Any] = {
            "title": self._get_title(doc[0]) if len(doc) else "Untitled paper",
            "abstract": "",
            "sections": {},
            "full_text_preview": "",
            "page_count": len(doc),
        }
        current = "introduction"
        preview_parts: list[str] = []

        for page_num, page in enumerate(doc):
            if page_num >= self.max_pages:
                break
            text = page.get_text("text", flags=4)
            for raw_line in text.splitlines():
                line = re.sub(r"\s+", " ", raw_line).strip()
                if not line or len(line) < 3:
                    continue
                section = self._detect_section(line)
                if section:
                    current = section
                    result["sections"].setdefault(current, [])
                    continue
                result["sections"].setdefault(current, []).append(line)
                if page_num < 3:
                    preview_parts.append(line)

        flattened: dict[str, str] = {}
        for key, parts in result["sections"].items():
            flattened[key] = " ".join(parts)
        result["sections"] = flattened
        result["abstract"] = flattened.get("abstract", "")
        result["full_text_preview"] = " ".join(preview_parts)[:3000]
        if not result["abstract"]:
            result["abstract"] = self._guess_abstract(result["full_text_preview"])
        return result

    def _get_title(self, first_page: fitz.Page) -> str:
        blocks = first_page.get_text("dict", flags=4).get("blocks", [])
        best_text = ""
        best_size = 0.0
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = re.sub(r"\s+", " ", span.get("text", "")).strip()
                    size = float(span.get("size", 0))
                    if size > best_size and len(text) > 8:
                        best_size = size
                        best_text = text
        return best_text or "Untitled paper"

    def _detect_section(self, line: str) -> str | None:
        if len(line) > 70:
            return None
        normalized = re.sub(r"^[0-9IVXivx.\s-]+", "", line).lower().rstrip(".:").strip()
        aliases = {"methodology": "methods", "method": "methods", "conclusions": "conclusion"}
        normalized = aliases.get(normalized, normalized)
        if normalized in HIGH_VALUE_SECTIONS:
            return normalized
        return None

    def _guess_abstract(self, preview: str) -> str:
        match = re.search(r"abstract\s+(.*?)(introduction|keywords|1\s+introduction)", preview, re.I | re.S)
        if match:
            return match.group(1).strip()[:1500]
        return preview[:1000]
