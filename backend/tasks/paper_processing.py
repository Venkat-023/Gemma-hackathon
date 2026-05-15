import logging
import time
import traceback
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from celery import Celery
from sqlalchemy import select

from core.config import get_settings
from graph.graph_builder import KnowledgeGraphBuilder
from ingestion.chunker import SemanticChunker
from ingestion.metadata_extractor import MetadataExtractor
from ingestion.pdf_parser import ScientificPDFParser
from models.database import AsyncSessionLocal
from models.entity import Entity, EntityRelationship, PaperEntity
from models.paper import Chunk as ChunkModel
from models.paper import Paper, ProcessingStatus
from reasoning.entity_extractor import ScientificEntityExtractor
from reasoning.relationship_mapper import RelationshipMapper
from retrieval.vector_store import VectorStore

logger = logging.getLogger("scientific_discovery.tasks")
settings = get_settings()
celery_app = Celery("scientific_discovery", broker=settings.celery_broker_url, backend=settings.celery_result_backend)


@celery_app.task(name="tasks.paper_processing.process_paper", bind=True, max_retries=2)
def process_paper(self, paper_id: int) -> dict:
    import asyncio

    return asyncio.run(_process_paper_async(paper_id))


async def _process_paper_async(paper_id: int) -> dict:
    timings: dict[str, float] = {}
    async with AsyncSessionLocal() as db:
        paper = await db.get(Paper, paper_id)
        if not paper:
            raise ValueError(f"Paper {paper_id} not found")
        try:
            await _set_status(db, paper, ProcessingStatus.processing)

            started = time.perf_counter()
            parser = ScientificPDFParser()
            raw = parser.parse_from_arxiv(paper.arxiv_id) if paper.arxiv_id else parser.parse_pdf(paper.pdf_path or "")
            timings["parse_pdf"] = time.perf_counter() - started

            started = time.perf_counter()
            metadata = MetadataExtractor().extract(raw)
            if metadata.get("doi"):
                duplicate_doi = (
                    await db.execute(select(Paper).where(Paper.doi == metadata["doi"], Paper.id != paper.id))
                ).scalar_one_or_none()
                if duplicate_doi:
                    metadata["doi"] = None
            for key, value in metadata.items():
                if value is not None and hasattr(paper, key):
                    setattr(paper, key, value)
            if not paper.pdf_path and paper.arxiv_id:
                paper.pdf_path = str(Path("/tmp") / f"{paper.arxiv_id}.pdf")
            await db.commit()
            timings["metadata"] = time.perf_counter() - started

            started = time.perf_counter()
            chunks = SemanticChunker().chunk_paper(raw)
            ids = VectorStore().add_chunks(
                paper.id,
                chunks,
                {"title": paper.title, "year": paper.publication_year, "authors": paper.authors or [], "arxiv_id": paper.arxiv_id},
            )
            for chunk, embedding_id in zip(chunks, ids, strict=False):
                db.add(
                    ChunkModel(
                        paper_id=paper.id,
                        section=chunk.section,
                        content=chunk.content,
                        sub_index=chunk.sub_index,
                        importance_score=chunk.importance_score,
                        chroma_embedding_id=embedding_id,
                        word_count=chunk.word_count,
                    )
                )
            await db.commit()
            timings["chunk_embed_store"] = time.perf_counter() - started

            started = time.perf_counter()
            extraction = ScientificEntityExtractor().extract(chunks)
            entity_name_to_id: dict[str, int] = {}
            for entity_type, names in extraction.entities.items():
                for name in names:
                    normalized = name.lower().strip()
                    existing = (
                        await db.execute(select(Entity).where(Entity.normalized_name == normalized, Entity.entity_type == entity_type))
                    ).scalar_one_or_none()
                    if existing:
                        existing.paper_count += 1
                        entity = existing
                    else:
                        entity = Entity(name=name, normalized_name=normalized, entity_type=entity_type, aliases=[])
                        db.add(entity)
                        await db.flush()
                    entity_name_to_id[name] = entity.id
            frequencies = Counter(word.lower().strip() for names in extraction.entities.values() for word in names)
            for name, entity_id in entity_name_to_id.items():
                await db.merge(PaperEntity(paper_id=paper.id, entity_id=entity_id, frequency=frequencies[name.lower().strip()] or 1))
            mapper = RelationshipMapper()
            for rel in extraction.relationships:
                source_id = entity_name_to_id.get(rel.get("source"))
                target_id = entity_name_to_id.get(rel.get("target"))
                if source_id and target_id:
                    db.add(
                        EntityRelationship(
                            source_entity_id=source_id,
                            target_entity_id=target_id,
                            relationship_type=mapper.normalize(rel.get("relation", "")),
                            confidence=float(rel.get("confidence", 0.5)),
                            evidence_text=rel.get("evidence"),
                            paper_id=paper.id,
                        )
                    )
            await db.commit()
            timings["entities"] = time.perf_counter() - started

            started = time.perf_counter()
            try:
                graph = KnowledgeGraphBuilder()
                graph.upsert_paper(paper.id, paper.title, paper.publication_year, paper.arxiv_id)
                for entity_type, names in extraction.entities.items():
                    for name in names:
                        graph.upsert_entity(name, entity_type)
                        graph.link_paper_entity(paper.id, name, frequencies[name.lower().strip()] or 1)
                for rel in extraction.relationships:
                    graph.create_relationship(
                        rel.get("source"),
                        rel.get("target"),
                        mapper.normalize(rel.get("relation", "")),
                        float(rel.get("confidence", 0.5)),
                        paper.id,
                        rel.get("evidence", ""),
                    )
            except Exception:
                logger.exception("neo4j_graph_update_failed", extra={"paper_id": paper.id})
            timings["graph"] = time.perf_counter() - started

            paper.processing_status = ProcessingStatus.completed
            paper.processed_at = datetime.now(UTC)
            await db.commit()
            logger.info("paper_processed", extra={"paper_id": paper.id, "timings": timings})
            return {"paper_id": paper.id, "status": "completed", "timings": timings}
        except Exception:
            traceback.print_exc()
            await db.rollback()
            paper.processing_status = ProcessingStatus.failed
            await db.commit()
            logger.error("paper_processing_failed", extra={"paper_id": paper.id, "traceback": traceback.format_exc()})
            return {"paper_id": paper.id, "status": "failed", "error": traceback.format_exc()}


async def _set_status(db, paper: Paper, status: ProcessingStatus) -> None:
    paper.processing_status = status
    await db.commit()
