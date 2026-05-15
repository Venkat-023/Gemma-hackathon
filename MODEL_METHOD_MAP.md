# Model and Method Map

This document explains which model, library, or algorithm is responsible for each process in **Scientific Discovery Copilot**. It is meant to be a quick technical guide for understanding how the backend works internally.

## High-Level Pipeline

```text
PDF Upload
  -> PDF Parsing
  -> Metadata Extraction
  -> Section Detection
  -> Semantic Chunking
  -> Embedding Generation
  -> ChromaDB Vector Storage
  -> Entity Extraction
  -> Knowledge Graph Construction
  -> Search / Analysis / Hypothesis Generation
  -> Streamlit UI Display
```

## Component Responsibility Table

| Process | Model / Method / Library | File(s) | Current Role |
| --- | --- | --- | --- |
| API backend | FastAPI + Uvicorn | `backend/api/main.py` | Serves all backend endpoints and Swagger UI. |
| UI frontend | Streamlit | `frontend/app.py` | Lets users upload papers and test backend functions. |
| PDF parsing | PyMuPDF | `backend/ingestion/pdf_parser.py` | Primary parser for PDF text, page blocks, spans, font sizes, and sections. |
| PDF fallback parsing | pdfplumber | `backend/ingestion/pdf_parser.py` | Used if PyMuPDF parsing fails or returns poor text/title output. |
| Title detection | Font-size heuristic | `backend/ingestion/pdf_parser.py` | Finds the largest meaningful first-page text span. |
| Section detection | Rule-based header matching | `backend/ingestion/pdf_parser.py` | Detects abstract, introduction, methods, results, discussion, conclusion, future work, references, etc. |
| Metadata extraction | Rule-based regex + parsed PDF fields | `backend/ingestion/metadata_extractor.py` | Extracts title, authors, DOI, year, journal-style metadata where possible. |
| arXiv ingestion | arXiv PDF URL + arXiv API XML | `backend/ingestion/arxiv_fetcher.py`, `backend/ingestion/pdf_parser.py` | Downloads arXiv PDFs and merges API metadata. |
| PubMed ingestion | BioPython Entrez | `backend/ingestion/pubmed_fetcher.py` | Fetches PubMed metadata path. |
| Semantic chunking | spaCy sentencizer + sentence-transformer similarity | `backend/ingestion/chunker.py` | Splits paper sections into meaningful chunks. |
| Chunk splitting model | `all-MiniLM-L6-v2` | `backend/ingestion/chunker.py` | Lightweight sentence embedding model used only for chunk boundary detection. |
| Chunk importance scoring | Rule-based scoring | `backend/ingestion/chunker.py` | Scores chunks using section type, novelty words, percentages, p-values, and contrast signals. |
| Main embeddings | `BAAI/bge-large-en-v1.5` | `backend/retrieval/vector_store.py` | Converts paper chunks and search queries into vector embeddings. |
| Vector database | ChromaDB persistent local store | `backend/retrieval/vector_store.py` | Stores chunk embeddings and performs vector similarity search. |
| Vector similarity | Cosine similarity through ChromaDB | `backend/retrieval/vector_store.py` | Retrieves semantically similar chunks. |
| Cross-paper similarity | BGE embeddings + ChromaDB search | `backend/retrieval/vector_store.py`, `backend/reasoning/cross_paper_reasoner.py` | Finds related chunks from different papers. |
| Relational storage | PostgreSQL 16 | `backend/models/*.py` | Stores papers, chunks, entities, relationships, hypotheses, and contradictions. |
| ORM | SQLAlchemy async + asyncpg | `backend/models/database.py` | Handles database sessions and model persistence. |
| Entity extraction primary path | spaCy / SciSpaCy | `backend/reasoning/entity_extractor.py` | Extracts raw scientific entities from high-importance chunks. |
| UMLS linking path | SciSpaCy UMLS linker | `backend/reasoning/entity_extractor.py` | Adds biomedical linking capability when model/linker is available. |
| Entity refinement | Gemma 4 through Ollama | `backend/reasoning/entity_extractor.py`, `backend/core/gemma_engine.py` | Intended to validate entities, add missed entities, and extract relationships. |
| Entity fallback | Deterministic scientific keyword extraction | `backend/reasoning/entity_extractor.py` | Keeps graph construction useful when local Gemma is too slow. |
| Relationship normalization | Rule-based mapper | `backend/reasoning/relationship_mapper.py` | Normalizes relationship labels such as activates, inhibits, causes, treats, similar_to, part_of. |
| Knowledge graph | Neo4j Community Edition | `backend/graph/graph_builder.py` | Stores paper/entity nodes and relationships. |
| Graph queries | Cypher | `backend/graph/graph_queries.py`, `backend/graph/graph_builder.py` | Fetches paper graphs, entity neighborhoods, and cross-paper paths. |
| Hypothesis generation | Gemma 4 through Ollama | `backend/core/hypothesis_generator.py`, `backend/core/gemma_engine.py` | Generates evidence-grounded scientific hypotheses. |
| Hypothesis MVP fallback | Deterministic evidence-grounded template | `backend/core/hypothesis_generator.py` | Used when `gemma4:e2b` is too slow and fast fallback is enabled. |
| Knowledge gap detection | Gemma 4 or deterministic text signals | `backend/core/hypothesis_generator.py` | Finds gaps such as bias, generalizability, privacy, real-time deployment, and dataset limits. |
| Hypothesis storage | PostgreSQL | `backend/core/hypothesis_generator.py`, `backend/models/hypothesis.py` | Persists generated hypotheses, scores, evidence, and experiments. |
| Hypothesis explanation | Gemma 4 or deterministic explanation fallback | `backend/core/hypothesis_generator.py` | Explains hypotheses in plain language and technical form. |
| Contradiction detection | Gemma 4 through Ollama | `backend/reasoning/cross_paper_reasoner.py`, `backend/core/gemma_engine.py` | Intended to compare paper claims and detect disagreements. |
| Contradiction MVP fallback | Conservative deterministic result | `backend/reasoning/cross_paper_reasoner.py` | Returns no high-confidence contradiction in local E2B MVP mode. |
| Landscape analysis | Gemma 4 or deterministic topic summary | `backend/reasoning/cross_paper_reasoner.py` | Summarizes milestones, open questions, and trend direction. |
| Multi-agent debate | Gemma 4 through Ollama | `backend/reasoning/multi_agent_debate.py` | Intended to run expert-style critique rounds. |
| Debate MVP fallback | Deterministic agent transcript + verdict | `backend/reasoning/multi_agent_debate.py` | Returns a quick domain/methodology/devil-advocate/synthesizer result in E2B mode. |
| Async paper processing | Celery + Redis | `backend/tasks/paper_processing.py` | Handles full ingestion pipeline in background. |
| Queue broker | Redis | `docker-compose.yml` | Stores Celery jobs and results. |
| Local model runtime | Ollama | `backend/core/gemma_engine.py`, `docker-compose.yml` | Runs Gemma models locally through HTTP API. |

## LLM Model Responsibilities

### Current Local Model

```text
gemma4:e2b
```

Used for:

- entity refinement when enabled
- relationship extraction when enabled
- hypothesis generation when full mode is selected
- contradiction reasoning when full mode is selected
- hypothesis explanation when full mode is selected
- multi-agent debate when full mode is selected
- research landscape synthesis when full mode is selected

Current behavior:

- `gemma4:e2b` is installed and reachable through Ollama.
- It is usable for small calls.
- It is slow for long scientific prompts.
- The app therefore supports fast MVP fallback mode.
- Default paper ingestion does not call Gemma; it uses spaCy/SciSpaCy plus deterministic entity rules for speed on CPU-only machines.

### Planned Upgrade Model

```text
gemma4:e4b
```

Why this is planned:

- `gemma4:4b` is not a valid Ollama tag.
- `gemma4:e4b` is the valid 4B-class Gemma 4 tag.
- It may improve quality, but it may also be slower depending on available CPU/GPU.
- We will test it only after the current Streamlit flow is confirmed.

### Larger Reasoning Option

```text
gemma4:26b
```

Intended future role:

- higher-quality hypothesis generation
- contradiction detection
- multi-agent debate
- complex research landscape analysis

Expected requirement:

- significantly stronger hardware than the current local MVP setup.

## Embedding Model Responsibilities

### Main Retrieval Embedding Model

```text
BAAI/bge-large-en-v1.5
```

Used by:

- `backend/retrieval/vector_store.py`

Responsibilities:

- embed paper chunks
- embed user search queries
- support semantic search
- support cross-paper similarity
- support unexplored connection discovery

Output is stored in:

```text
backend/data/chroma_db
```

### Chunking Embedding Model

```text
all-MiniLM-L6-v2
```

Used by:

- `backend/ingestion/chunker.py`

Responsibilities:

- embed individual sentences
- compute consecutive sentence similarity
- detect semantic split points
- avoid cutting sections at arbitrary word counts

Why a smaller model is used here:

- chunking needs many small sentence comparisons
- speed matters more than deep retrieval quality
- main retrieval uses the stronger BGE model later

## Non-LLM Methods

### PDF Layout Understanding

Method:

- font-size analysis
- page span extraction
- section header rules
- fallback text extraction

Used for:

- title extraction
- abstract extraction
- section segmentation
- reference extraction

### Importance Scoring

Method:

- rule-based scoring

Signals:

- section type
- novelty language
- numerical result patterns
- statistical language
- contrast words

Example high-value sections:

- abstract
- results
- conclusion
- future work

### Entity Fallback

Method:

- deterministic scientific keyword matching
- acronym/concept pattern detection

Purpose:

- keeps graph construction alive even when local Gemma is too slow
- works especially well for AI/medical-imaging papers where terms like deep learning, endoscopy, lesion, federated learning, and clinical translation are important
- is now the default ingestion path on CPU-only local runs

### Hypothesis MVP Fallback

Method:

- retrieved-evidence template
- knowledge-gap signals
- confidence/novelty/testability defaults

Purpose:

- lets the UI and backend show a useful hypothesis without waiting several minutes
- can be disabled from Streamlit by unchecking **Use fast MVP fallback**

## Process-by-Process Explanation

### Uploading a PDF

Handled by:

- FastAPI upload endpoint
- file save logic
- PostgreSQL paper row creation
- Celery processing task

Models used:

- none during the upload request itself

### Parsing a Paper

Handled by:

- PyMuPDF
- pdfplumber fallback
- metadata extraction rules

Models used:

- none

### Chunking a Paper

Handled by:

- spaCy sentence splitting
- `all-MiniLM-L6-v2` sentence embeddings
- cosine similarity between adjacent sentences
- section-aware chunk rules

### Embedding Chunks

Handled by:

- `BAAI/bge-large-en-v1.5`
- ChromaDB

### Searching Papers

Handled by:

- `BAAI/bge-large-en-v1.5` for query embedding
- ChromaDB for vector similarity

Gemma is not used for normal semantic search.

### Extracting Entities

Handled by:

- spaCy / SciSpaCy
- optional Gemma refinement
- deterministic keyword fallback

### Building the Graph

Handled by:

- Neo4j
- Cypher queries
- entity and paper records from PostgreSQL

Gemma is not directly required for graph writes unless relationship extraction uses Gemma.

### Generating Hypotheses

Handled by:

- ChromaDB retrieval
- knowledge-gap detection
- optional graph context
- Gemma 4 full generation or deterministic fallback

Primary model:

```text
gemma4:e2b currently, gemma4:e4b planned
```

### Detecting Contradictions

Handled by:

- relevant chunk retrieval
- pairwise paper comparison
- Gemma 4 full reasoning or local fallback

Primary model:

```text
Gemma 4 through Ollama
```

### Multi-Agent Debate

Handled by:

- Gemma 4 agent turns in full mode
- deterministic debate fallback in local E2B MVP mode

Agents:

- domain expert
- methodology critic
- devil advocate
- cross-domain linker
- synthesizer

## Current Timing Expectations

Observed locally:

| Operation | Expected Local Behavior |
| --- | --- |
| Backend health | usually under 1 second |
| Warm semantic search | about 1 second or less |
| First semantic search after restart | can be slower due to model warmup |
| PDF upload request | usually quick |
| Full paper processing | can take minutes depending on PDF length and model warmup |
| Hypothesis with fast fallback | usually seconds after retrieval warms up |
| Hypothesis with full Gemma E2B | can take several minutes or timeout |
| Multi-agent full Gemma debate | expected to be very slow locally |

## Where to Change Models

Backend local runner:

```text
backend/run_local_uvicorn.ps1
```

Environment variables:

```powershell
$env:GEMMA_REASONING_MODEL = "gemma4:e2b"
$env:GEMMA_LIGHT_MODEL = "gemma4:e2b"
```

To test E4B later, change these to:

```powershell
$env:GEMMA_REASONING_MODEL = "gemma4:e4b"
$env:GEMMA_LIGHT_MODEL = "gemma4:e4b"
```

Only do this after pulling:

```powershell
docker exec scientific-discovery-copilot-ollama-1 ollama pull gemma4:e4b
```

## Summary

The current system uses:

- **PyMuPDF/pdfplumber** for paper reading
- **rule-based methods** for metadata and section detection
- **all-MiniLM-L6-v2** for chunk splitting
- **BAAI/bge-large-en-v1.5** for semantic embeddings
- **ChromaDB** for vector retrieval
- **spaCy/SciSpaCy + fallback rules** for entity extraction
- **Neo4j** for graph storage
- **Gemma 4 via Ollama** for scientific reasoning tasks
- **deterministic fallback methods** to keep the MVP responsive on local hardware
- **Streamlit** for manual UI testing
