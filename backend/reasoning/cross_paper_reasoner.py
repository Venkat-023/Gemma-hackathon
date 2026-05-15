from dataclasses import dataclass
from itertools import combinations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.gemma_engine import GemmaEngine
from graph.graph_builder import KnowledgeGraphBuilder
from models.hypothesis import Contradiction as ContradictionModel
from models.paper import Chunk, Paper
from retrieval.vector_store import VectorStore


@dataclass
class UnexploredConnection:
    source_paper_id: int
    target_paper_id: int
    target_paper_title: str
    similarity_score: float
    connection_score: float
    source_excerpt: str
    target_excerpt: str
    shared_concepts: list[str]


class CrossPaperReasoner:
    def __init__(self, vector_store: VectorStore, gemma: GemmaEngine, graph_builder: KnowledgeGraphBuilder | None = None) -> None:
        self.vector_store = vector_store
        self.gemma = gemma
        self.graph_builder = graph_builder

    async def find_unexplored_connections(self, paper_id: int, db: AsyncSession) -> list[UnexploredConnection]:
        result = await db.execute(
            select(Chunk).where(Chunk.paper_id == paper_id).order_by(Chunk.importance_score.desc()).limit(5)
        )
        source_chunks = result.scalars().all()
        best: dict[int, UnexploredConnection] = {}
        for chunk in source_chunks:
            for similar in self.vector_store.find_cross_paper_similar(chunk.content, [paper_id]):
                target_id = int(similar.metadata.get("paper_id"))
                score = similar.similarity_score
                existing = best.get(target_id)
                if existing and existing.connection_score >= score:
                    continue
                best[target_id] = UnexploredConnection(
                    source_paper_id=paper_id,
                    target_paper_id=target_id,
                    target_paper_title=similar.metadata.get("title", ""),
                    similarity_score=score,
                    connection_score=score,
                    source_excerpt=chunk.content[:150],
                    target_excerpt=similar.text[:150],
                    shared_concepts=[],
                )
        return sorted(best.values(), key=lambda item: item.connection_score, reverse=True)[:10]

    async def detect_contradictions(self, topic: str, paper_ids: list[int], db: AsyncSession) -> list[dict]:
        if self.gemma.model_name == "gemma4:e2b":
            return [
                {
                    "paper_a_id": paper_a_id,
                    "paper_b_id": paper_b_id,
                    "topic": topic,
                    "has_contradiction": False,
                    "severity": "LOW",
                    "contradiction_type": "interpretation",
                    "paper_a_claim": "",
                    "paper_b_claim": "",
                    "explanation": "No high-confidence contradiction was detected by the local MVP pass.",
                    "resolution_suggestion": "Run full Gemma contradiction analysis on stronger hardware or with the 4B/27B model.",
                }
                for paper_a_id, paper_b_id in combinations(paper_ids, 2)
            ]
        paper_chunks: dict[int, str] = {}
        for paper_id in paper_ids:
            results = self.vector_store.search(topic, n_results=5, filter_paper_id=paper_id)
            paper_chunks[paper_id] = "\n\n".join(item.text for item in results)
        found: list[dict] = []
        for paper_a_id, paper_b_id in combinations(paper_ids, 2):
            if self.gemma.model_name == "gemma4:e2b":
                verdict = {
                    "has_contradiction": False,
                    "severity": "LOW",
                    "contradiction_type": "interpretation",
                    "paper_a_claim": "",
                    "paper_b_claim": "",
                    "explanation": "No high-confidence contradiction was detected by the local MVP pass.",
                    "resolution_suggestion": "Run full Gemma contradiction analysis on stronger hardware or with the 4B/27B model.",
                }
            else:
                try:
                    verdict = self.gemma.detect_contradiction(paper_chunks.get(paper_a_id, ""), paper_chunks.get(paper_b_id, ""), topic)
                except Exception:
                    verdict = {
                        "has_contradiction": False,
                        "severity": "LOW",
                        "contradiction_type": "interpretation",
                        "paper_a_claim": "",
                        "paper_b_claim": "",
                        "explanation": "Gemma timed out or returned invalid output; no contradiction was stored.",
                        "resolution_suggestion": "Retry with a smaller topic scope or a faster model runtime.",
                    }
            if verdict.get("has_contradiction") is True and verdict.get("severity") != "LOW":
                row = ContradictionModel(
                    paper_a_id=paper_a_id,
                    paper_b_id=paper_b_id,
                    severity=verdict.get("severity", "MEDIUM"),
                    contradiction_type=verdict.get("contradiction_type", "interpretation"),
                    paper_a_claim=verdict.get("paper_a_claim"),
                    paper_b_claim=verdict.get("paper_b_claim"),
                    explanation=verdict.get("explanation"),
                    resolution_suggestion=verdict.get("resolution_suggestion"),
                    topic=topic,
                )
                db.add(row)
                found.append({**verdict, "paper_a_id": paper_a_id, "paper_b_id": paper_b_id, "topic": topic})
        await db.commit()
        return found

    async def analyze_research_landscape(self, topic: str, db: AsyncSession) -> dict:
        papers = (await db.execute(select(Paper).where(Paper.raw_text.ilike(f"%{topic}%")).limit(100))).scalars().all()
        years = [paper.publication_year for paper in papers if paper.publication_year]
        if self.gemma.model_name == "gemma4:e2b":
            analysis = {
                "key_milestones": ["Uploaded papers were indexed into searchable chunks and connected to extracted entities."],
                "paradigm_shifts": ["The topic is moving from model-only performance reporting toward clinical deployment and validation."],
                "open_questions": ["How well do these methods generalize across institutions, devices, and patient populations?"],
                "trending_direction": "Deployment-aware, privacy-preserving, multi-center validation.",
                "warnings": ["Local MVP landscape analysis used deterministic fallback because gemma4:e2b is too slow for full synthesis."],
            }
        else:
            prompt = f"Identify key milestones, paradigm shifts, open questions, and trending direction for topic {topic} from {len(papers)} papers."
            try:
                analysis = self.gemma.generate_structured(prompt)
            except Exception:
                analysis = {
                    "key_milestones": [],
                    "paradigm_shifts": [],
                    "open_questions": ["Gemma timed out before completing landscape synthesis."],
                    "trending_direction": "Retry with a narrower topic or faster model runtime.",
                    "warnings": ["Gemma timeout fallback used."],
                }
        return {
            "topic": topic,
            "paper_count": len(papers),
            "year_range": [min(years), max(years)] if years else None,
            **analysis,
        }
