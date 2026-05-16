from __future__ import annotations


SECTION_PRIORITY = {
    "abstract": 1.0,
    "conclusion": 0.95,
    "results": 0.90,
    "discussion": 0.80,
    "introduction": 0.60,
    "methods": 0.50,
}
MAX_WORDS = 300
OVERLAP = 30


class FastChunker:
    def chunk(self, sections: dict[str, str], paper_id: int) -> list[dict]:
        chunks: list[dict] = []
        ordered_sections = sorted(
            sections.items(),
            key=lambda item: -SECTION_PRIORITY.get(item[0], 0.5),
        )
        for section, text in ordered_sections:
            if not text or len(text.strip()) < 50:
                continue
            importance = SECTION_PRIORITY.get(section, 0.5)
            words = text.split()
            if len(words) <= MAX_WORDS:
                chunks.append(self._chunk(paper_id, section, 0, text, importance))
                continue
            start = 0
            sub_index = 0
            while start < len(words):
                end = min(start + MAX_WORDS, len(words))
                chunks.append(
                    self._chunk(paper_id, section, sub_index, " ".join(words[start:end]), importance)
                )
                sub_index += 1
                if end == len(words):
                    break
                start += MAX_WORDS - OVERLAP
        return sorted(chunks, key=lambda chunk: -chunk["importance"])

    def _chunk(self, paper_id: int, section: str, sub_index: int, text: str, importance: float) -> dict:
        safe_section = section.replace(" ", "_")
        return {
            "paper_id": paper_id,
            "section": section,
            "sub_index": sub_index,
            "content": text.strip(),
            "importance": importance,
            "chroma_id": f"p{paper_id}_{safe_section}_{sub_index}",
        }
