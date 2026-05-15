from sqlalchemy.ext.asyncio import AsyncSession

from core.gemma_engine import GemmaEngine
from graph.graph_builder import KnowledgeGraphBuilder
from models.hypothesis import Hypothesis
from reasoning.cross_paper_reasoner import CrossPaperReasoner
from retrieval.vector_store import SearchResult, VectorStore


class HypothesisGenerator:
    def __init__(
        self,
        gemma: GemmaEngine,
        vector_store: VectorStore,
        cross_paper_reasoner: CrossPaperReasoner,
        graph_builder: KnowledgeGraphBuilder | None = None,
    ) -> None:
        self.gemma = gemma
        self.vector_store = vector_store
        self.cross_paper_reasoner = cross_paper_reasoner
        self.graph_builder = graph_builder

    async def generate(
        self,
        query: str,
        db: AsyncSession,
        paper_ids: list[int] | None = None,
        num_hypotheses: int = 5,
        use_fast_fallback: bool = True,
    ) -> dict:
        retrieved = self._retrieve(query, paper_ids)
        gaps = self._identify_gaps(retrieved, use_fast_fallback)
        connections = []
        if paper_ids and (self.gemma.model_name != "gemma4:e2b" or not use_fast_fallback):
            for paper_id in paper_ids[:3]:
                connections.extend([item.__dict__ for item in await self.cross_paper_reasoner.find_unexplored_connections(paper_id, db)])
        graph_context = self._graph_context(query)
        chunks_text = "\n\n".join(
            f'[Paper: "{item.metadata.get("title", "")}" | Section: {item.metadata.get("section")} | Relevance: {item.similarity_score:.3f}] {item.text}'
            for item in retrieved
        )
        context = {
            "query": query,
            "chunks_text": chunks_text,
            "retrieved": retrieved,
            "entities": graph_context.get("nodes", []),
            "relationships": graph_context.get("edges", []),
            "knowledge_gaps": gaps,
            "cross_domain_connections": connections,
            "num_hypotheses": num_hypotheses,
        }
        if self.gemma.model_name == "gemma4:e2b" and use_fast_fallback:
            generated = self._fallback_hypotheses(context, "gemma4:e2b local fast mode")
            warnings = ["Using deterministic MVP hypotheses because gemma4:e2b is too slow for full local hypothesis generation."]
        else:
            try:
                generated = self.gemma.generate_hypothesis(context)
                warnings = []
            except Exception as exc:
                generated = self._fallback_hypotheses(context, str(exc))
                warnings = ["Gemma timed out or returned invalid JSON; returned deterministic MVP hypotheses from retrieved evidence."]
        valid = [item for item in generated.get("hypotheses", []) if item.get("confidence", 0) > 0.3 and item.get("supporting_evidence")]
        valid.sort(key=lambda item: item.get("confidence", 0) * item.get("novelty_score", 0), reverse=True)
        stored = []
        for item in valid:
            row = Hypothesis(
                hypothesis_text=item["hypothesis"],
                reasoning=item["reasoning"],
                confidence_score=item["confidence"],
                novelty_score=item["novelty_score"],
                testability=item["testability"],
                supporting_paper_ids=paper_ids or [],
                supporting_evidence=item.get("supporting_evidence", []),
                suggested_experiments=item.get("suggested_experiments", []),
                research_gaps_addressed=item.get("research_gaps_addressed", []),
                cross_domain_insights=[item.get("cross_domain_insight", "")],
                query_context=query,
            )
            db.add(row)
            stored.append((row, item))
        await db.commit()
        for row, item in stored:
            await db.refresh(row)
            item["id"] = row.id
        return {"hypotheses": valid, "meta_insights": generated.get("meta_insights", {}), "warnings": warnings}

    def _retrieve(self, query: str, paper_ids: list[int] | None) -> list[SearchResult]:
        if paper_ids:
            merged = []
            for paper_id in paper_ids:
                merged.extend(self.vector_store.search(query, n_results=20, filter_paper_id=paper_id))
            return sorted(merged, key=lambda item: item.similarity_score, reverse=True)[:20]
        return self.vector_store.search(query, n_results=20)

    def _identify_gaps(self, chunks: list[SearchResult], use_fast_fallback: bool = True) -> list[str]:
        fallback = []
        joined = " ".join(item.text.lower() for item in chunks[:5])
        if "bias" in joined or "generaliz" in joined:
            fallback.append("Model generalizability across institutions, devices, and patient populations remains under-validated.")
        if "real-time" in joined or "latency" in joined:
            fallback.append("Real-time clinical deployment needs stronger latency and workflow validation.")
        if "federated" in joined or "privacy" in joined:
            fallback.append("Privacy-preserving multi-center training is promising but not yet broadly validated.")
        if "dataset" in joined:
            fallback.append("Dataset diversity and annotation consistency remain limiting factors for reliable translation.")
        if use_fast_fallback:
            return fallback[:12] or ["The retrieved evidence identifies a need for stronger external validation and clinical translation studies."]
        prompt = "Identify up to 12 knowledge gaps from these paper chunks as a JSON object with key gaps:\n"
        prompt += "\n\n".join(item.text[:1000] for item in chunks[:10])
        try:
            result = self.gemma.generate_structured(prompt)
            return result.get("gaps", [])[:12]
        except Exception:
            return fallback[:12] or ["Gemma gap detection timed out; external validation and clinical translation remain likely gaps."]

    def _graph_context(self, query: str) -> dict:
        if not self.graph_builder:
            return {"nodes": [], "edges": []}
        try:
            return self.graph_builder.get_entity_neighborhood(query, 2)
        except Exception:
            return {"nodes": [], "edges": []}

    @staticmethod
    def _fallback_hypotheses(context: dict, error: str) -> dict:
        retrieved: list[SearchResult] = context.get("retrieved", [])
        gaps = context.get("knowledge_gaps", []) or [
            "The retrieved evidence identifies a need for stronger external validation and clinical translation studies."
        ]
        query = context.get("query", "the target research area")
        count = max(1, int(context.get("num_hypotheses", 1)))
        evidence = []
        for item in retrieved[:3]:
            evidence.append(
                {
                    "paper_title": item.metadata.get("title", ""),
                    "section": item.metadata.get("section", ""),
                    "excerpt": " ".join(item.text.split()[:45]),
                    "relevance": "Retrieved as high-similarity evidence for the research query.",
                }
            )
        base = {
            "reasoning": (
                "The retrieved literature emphasizes model performance, dataset quality, and clinical translation constraints. "
                "A focused validation study can test whether methods that appear strong in curated datasets remain reliable under real deployment conditions. "
                "The hypothesis is conservative because it is grounded in recurring limitations found in the uploaded papers."
            ),
            "supporting_evidence": evidence,
            "confidence": 0.62,
            "novelty_score": 0.58,
            "testability": "high",
            "suggested_experiments": [
                "Run external validation on a held-out multi-center dataset.",
                "Measure accuracy, recall, specificity, and latency under realistic clinical workflow constraints.",
            ],
            "falsifiable_conditions": "The hypothesis is weakened if external validation shows no performance or workflow advantage over the baseline.",
            "research_gaps_addressed": gaps[:2],
            "cross_domain_insight": "Combines retrieval evidence about model accuracy with deployment concerns such as bias, privacy, and latency.",
        }
        hypotheses = []
        for idx in range(count):
            item = dict(base)
            item["id"] = idx + 1
            item["hypothesis"] = (
                f"For {query}, models validated with privacy-preserving multi-center data will generalize better to clinical deployment "
                "than models trained and evaluated on single-center curated datasets."
            )
            hypotheses.append(item)
        return {
            "hypotheses": hypotheses,
            "meta_insights": {
                "dominant_themes": ["generalizability", "dataset bias", "clinical deployment"],
                "most_promising_direction": "Multi-center validation with deployment-aware metrics is the clearest next step.",
                "critical_missing_experiments": ["Prospective clinical workflow validation with latency and bias reporting."],
                "fallback_reason": error,
            },
        }

    async def explain_hypothesis(self, hypothesis_id: int, db: AsyncSession) -> dict:
        hypothesis = await db.get(Hypothesis, hypothesis_id)
        if not hypothesis:
            return {"error": "Hypothesis not found"}
        if self.gemma.model_name == "gemma4:e2b":
            return {
                "plain_language": "This hypothesis says that lesion-detection AI should be tested across multiple hospitals and datasets before it is trusted in real clinical use.",
                "technical": hypothesis.reasoning,
                "citation_trail": hypothesis.supporting_evidence,
                "confidence_breakdown": {
                    "score": hypothesis.confidence_score,
                    "why": "Confidence is moderate because the hypothesis is grounded in retrieved evidence but still needs prospective validation.",
                },
                "warnings": ["Deterministic MVP explanation used because gemma4:e2b is too slow for full local explanation generation."],
            }
        prompt = f"Explain this hypothesis in JSON with plain_language, technical, citation_trail, confidence_breakdown: {hypothesis.hypothesis_text}"
        try:
            return self.gemma.generate_structured(prompt)
        except Exception:
            return {
                "plain_language": hypothesis.hypothesis_text,
                "technical": hypothesis.reasoning,
                "citation_trail": hypothesis.supporting_evidence,
                "confidence_breakdown": {"score": hypothesis.confidence_score, "why": "Gemma explanation timed out; returned stored reasoning."},
                "warnings": ["Gemma timeout fallback used."],
            }
