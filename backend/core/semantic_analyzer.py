from __future__ import annotations

from collections import Counter, defaultdict
import re
from typing import Any


DOMAIN_TERMS = {
    "METHOD": [
        "deep learning",
        "machine learning",
        "convolutional neural network",
        "cnn",
        "transformer",
        "attention",
        "classification",
        "segmentation",
        "object detection",
        "lesion detection",
        "endoscopy",
        "colonoscopy",
        "gastroscopy",
        "computer-aided diagnosis",
        "cad",
        "systematic review",
        "meta-analysis",
    ],
    "DISEASE": [
        "cancer",
        "colorectal cancer",
        "gastric cancer",
        "esophageal cancer",
        "adenoma",
        "polyp",
        "lesion",
        "ulcer",
        "barrett",
        "neoplasia",
    ],
    "CONCEPT": [
        "sensitivity",
        "specificity",
        "accuracy",
        "precision",
        "recall",
        "auc",
        "dataset",
        "annotation",
        "real-time detection",
        "clinical workflow",
        "generalization",
        "external validation",
        "explainability",
        "false positives",
        "false negatives",
    ],
}

RELATION_PATTERNS = [
    (r"(?P<src>deep learning|machine learning|cnn|transformer).{0,80}(?P<tgt>lesion detection|polyp detection|classification|segmentation)", "improves"),
    (r"(?P<src>endoscopy|colonoscopy|gastroscopy).{0,80}(?P<tgt>lesion|polyp|adenoma|cancer)", "detects"),
    (r"(?P<src>dataset|annotation|external validation).{0,100}(?P<tgt>generalization|accuracy|performance)", "influences"),
    (r"(?P<src>false positives|false negatives).{0,80}(?P<tgt>clinical workflow|accuracy|sensitivity|specificity)", "affects"),
]


def analyze_paper(parsed: dict[str, Any], chunks: list[dict]) -> dict[str, Any]:
    text = _paper_text(parsed, chunks)
    entities = extract_entities(text)
    relationships = extract_relationships(text, entities)
    gaps = infer_research_gaps(text, entities)
    themes = rank_themes(text, entities)
    summary = deterministic_summary(parsed, themes, gaps)
    return {
        "entities": entities,
        "relationships": relationships,
        "research_gaps": gaps,
        "themes": themes,
        "deterministic_summary": summary,
    }


def merge_entity_data(gemma_data: dict[str, Any] | None, deterministic: dict[str, Any]) -> dict[str, Any]:
    merged_entities: dict[str, list[str]] = defaultdict(list)
    for source in [deterministic.get("entities", {}), (gemma_data or {}).get("entities", {})]:
        for entity_type, names in (source or {}).items():
            for name in names or []:
                _append_unique(merged_entities[entity_type], _clean_label(str(name)))

    relationships = []
    seen = set()
    for rel in deterministic.get("relationships", []) + ((gemma_data or {}).get("relationships", []) or []):
        source = _clean_label(str(rel.get("source", "")))
        target = _clean_label(str(rel.get("target", "")))
        relation = str(rel.get("relation", "relates_to")).strip() or "relates_to"
        if not source or not target or source.lower() == target.lower():
            continue
        key = (source.lower(), relation, target.lower())
        if key in seen:
            continue
        seen.add(key)
        relationships.append(
            {
                "source": source,
                "relation": relation,
                "target": target,
                "confidence": float(rel.get("confidence", 0.65)),
                "evidence": rel.get("evidence", "Semantic co-occurrence in high-value paper sections."),
            }
        )
    return {"entities": dict(merged_entities), "relationships": relationships}


def infer_cross_paper_connections(papers_meta: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    connections = []
    ids = list(papers_meta)
    for index, paper_a_id in enumerate(ids):
        for paper_b_id in ids[index + 1 :]:
            paper_a = papers_meta[paper_a_id]
            paper_b = papers_meta[paper_b_id]
            entities_a = _entity_set(paper_a.get("entities", {}))
            entities_b = _entity_set(paper_b.get("entities", {}))
            shared = sorted(entities_a & entities_b)
            if not shared:
                continue
            themes_a = {theme["term"].lower() for theme in paper_a.get("themes", [])}
            themes_b = {theme["term"].lower() for theme in paper_b.get("themes", [])}
            shared_themes = sorted(themes_a & themes_b)
            score = round(min(1.0, (len(shared) * 0.12) + (len(shared_themes) * 0.08)), 3)
            connections.append(
                {
                    "paper_a_id": paper_a_id,
                    "paper_a_title": paper_a.get("title", ""),
                    "paper_b_id": paper_b_id,
                    "paper_b_title": paper_b.get("title", ""),
                    "shared_entities": shared[:12],
                    "shared_themes": shared_themes[:8],
                    "connection_score": score,
                    "discovery_angle": _connection_angle(shared, shared_themes),
                }
            )
    return sorted(connections, key=lambda item: item["connection_score"], reverse=True)


def build_landscape(papers_meta: dict[int, dict[str, Any]], topic: str) -> dict[str, Any]:
    all_gaps = []
    theme_counter: Counter[str] = Counter()
    entity_counter: Counter[str] = Counter()
    for paper in papers_meta.values():
        all_gaps.extend(paper.get("research_gaps", []))
        for theme in paper.get("themes", []):
            theme_counter[theme["term"]] += theme.get("count", 1)
        for entity in _entity_set(paper.get("entities", {})):
            entity_counter[entity] += 1
    return {
        "topic": topic,
        "paper_count": len(papers_meta),
        "dominant_themes": [{"term": term, "count": count} for term, count in theme_counter.most_common(10)],
        "recurring_entities": [{"entity": term, "paper_count": count} for term, count in entity_counter.most_common(12)],
        "open_questions": _dedupe(all_gaps)[:10],
        "cross_paper_connections": infer_cross_paper_connections(papers_meta)[:10],
        "scientific_discovery_summary": _landscape_summary(topic, theme_counter, all_gaps),
    }


def extract_entities(text: str) -> dict[str, list[str]]:
    lowered = text.lower()
    entities: dict[str, list[str]] = {key: [] for key in ["DISEASE", "PROTEIN", "GENE", "CHEMICAL", "METHOD", "CONCEPT"]}
    for entity_type, terms in DOMAIN_TERMS.items():
        for term in terms:
            if term in lowered:
                _append_unique(entities[entity_type], _clean_label(term))

    for acronym in re.findall(r"\b[A-Z][A-Z0-9-]{2,}\b", text):
        if acronym not in {"PDF", "HTTP", "JSON"}:
            _append_unique(entities["CONCEPT"], acronym)

    for phrase in re.findall(r"\b(?:[A-Z][a-z]+(?:-[A-Z][a-z]+)?\s+){1,3}(?:Network|Model|Detection|Endoscopy|Cancer|Review|Dataset)\b", text):
        _append_unique(entities["CONCEPT"], _clean_label(phrase))

    for metric in re.findall(r"\b(?:accuracy|sensitivity|specificity|auc|precision|recall)\s+(?:of\s+)?\d+(?:\.\d+)?%?", lowered):
        _append_unique(entities["CONCEPT"], metric)

    return {key: values[:16] for key, values in entities.items() if values}


def extract_relationships(text: str, entities: dict[str, list[str]]) -> list[dict[str, Any]]:
    lowered = text.lower()
    relationships = []
    for pattern, relation in RELATION_PATTERNS:
        for match in re.finditer(pattern, lowered, re.I | re.S):
            relationships.append(
                {
                    "source": _clean_label(match.group("src")),
                    "relation": relation,
                    "target": _clean_label(match.group("tgt")),
                    "confidence": 0.72,
                    "evidence": _evidence_window(text, match.start(), match.end()),
                }
            )

    methods = entities.get("METHOD", [])[:5]
    diseases = entities.get("DISEASE", [])[:5]
    concepts = entities.get("CONCEPT", [])[:5]
    for method in methods:
        for disease in diseases:
            if method.lower() in lowered and disease.lower() in lowered:
                relationships.append(
                    {
                        "source": method,
                        "relation": "applied_to",
                        "target": disease,
                        "confidence": 0.62,
                        "evidence": "Both concepts appear in high-value sections of the same paper.",
                    }
                )
        for concept in concepts:
            if method.lower() in lowered and concept.lower() in lowered:
                relationships.append(
                    {
                        "source": method,
                        "relation": "evaluated_by",
                        "target": concept,
                        "confidence": 0.58,
                        "evidence": "Method and evaluation concept co-occur in high-value sections.",
                    }
                )
    return _dedupe_relationships(relationships)[:30]


def infer_research_gaps(text: str, entities: dict[str, list[str]]) -> list[str]:
    lowered = text.lower()
    gaps = []
    if "external validation" not in lowered:
        gaps.append("Need external validation across independent clinical sites and device settings.")
    if "prospective" not in lowered:
        gaps.append("Need prospective studies that measure real-time clinical impact.")
    if "explain" not in lowered and "interpret" not in lowered:
        gaps.append("Need explainability analysis to make model decisions auditable for clinicians.")
    if "dataset" in lowered and ("bias" not in lowered and "generalization" not in lowered):
        gaps.append("Need dataset bias and generalization analysis across populations.")
    if "false positive" not in lowered:
        gaps.append("Need explicit false-positive analysis to estimate clinical workflow burden.")
    methods = entities.get("METHOD", [])
    diseases = entities.get("DISEASE", [])
    if methods and diseases:
        gaps.append(f"Opportunity to compare {methods[0]} performance across multiple lesion and cancer subtypes.")
    return _dedupe(gaps)[:8]


def rank_themes(text: str, entities: dict[str, list[str]]) -> list[dict[str, Any]]:
    lowered = text.lower()
    candidates = [term for terms in DOMAIN_TERMS.values() for term in terms]
    candidates.extend(entity for values in entities.values() for entity in values)
    counts = Counter()
    for term in candidates:
        count = lowered.count(term.lower())
        if count:
            counts[_clean_label(term)] += count
    return [{"term": term, "count": count} for term, count in counts.most_common(12)]


def deterministic_summary(parsed: dict[str, Any], themes: list[dict[str, Any]], gaps: list[str]) -> dict[str, Any]:
    theme_text = ", ".join(theme["term"] for theme in themes[:4]) or "the paper's main methods and findings"
    return {
        "tldr": f"This paper centers on {theme_text}.",
        "contribution": "Fast semantic analysis identified methods, clinical targets, evaluation concepts, and research gaps from high-value sections.",
        "key_findings": [f"Prominent theme: {theme['term']}" for theme in themes[:3]],
        "limitations": gaps[:2],
        "future_directions": gaps[2:5],
        "source": "deterministic_fast_semantics",
    }


def _paper_text(parsed: dict[str, Any], chunks: list[dict]) -> str:
    high_value = [parsed.get("title", ""), parsed.get("abstract", "")]
    for chunk in chunks[:10]:
        high_value.append(chunk.get("content", ""))
    return " ".join(high_value)


def _entity_set(entities: dict[str, list[str]]) -> set[str]:
    return {str(name).strip().lower() for names in entities.values() for name in names if str(name).strip()}


def _append_unique(values: list[str], value: str) -> None:
    if value and value.lower() not in {item.lower() for item in values}:
        values.append(value)


def _clean_label(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip(" .,:;()[]{}")).strip()


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        key = value.lower()
        if key not in seen:
            seen.add(key)
            output.append(value)
    return output


def _dedupe_relationships(relationships: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    output = []
    for rel in relationships:
        key = (rel["source"].lower(), rel["relation"], rel["target"].lower())
        if key not in seen:
            seen.add(key)
            output.append(rel)
    return output


def _evidence_window(text: str, start: int, end: int) -> str:
    return re.sub(r"\s+", " ", text[max(0, start - 120) : min(len(text), end + 120)]).strip()


def _connection_angle(shared: list[str], shared_themes: list[str]) -> str:
    anchor = ", ".join((shared_themes or shared)[:3])
    return f"These papers may be connected through shared scientific concepts: {anchor}."


def _landscape_summary(topic: str, theme_counter: Counter[str], gaps: list[str]) -> str:
    themes = ", ".join(term for term, _ in theme_counter.most_common(3)) or topic
    gap = gaps[0] if gaps else "More comparative evidence is needed."
    return f"For {topic}, the current mini-landscape is dominated by {themes}. Key discovery gap: {gap}"
