# Scientific Discovery Copilot

A multi-module scientific discovery backend and lightweight Streamlit test UI for ingesting research papers, extracting scientific structure, building a knowledge graph, finding cross-paper connections, detecting contradictions, and generating explainable research hypotheses with Gemma through Ollama.

This project is built for the Gemma Impact Challenge as a discovery engine, not a chatbot or plain RAG demo.

## What It Does

Scientific Discovery Copilot turns PDF papers into connected scientific evidence:

- Parses uploaded research PDFs with PyMuPDF, with pdfplumber fallback support.
- Extracts metadata, abstract, sections, references, and page text.
- Splits papers into semantic chunks with importance scoring.
- Stores chunks in Postgres and indexes embeddings in ChromaDB.
- Uses BAAI/bge-large-en-v1.5 embeddings for semantic retrieval.
- Extracts scientific entities and relationships.
- Builds paper/entity relationship graphs in Neo4j.
- Finds unexplored cross-paper connections.
- Detects contradictions across selected papers.
- Generates testable hypotheses from retrieved evidence, graph context, research gaps, and cross-domain links.
- Provides a Streamlit UI for uploading papers, polling processing status, testing Gemma, searching, graph inspection, hypothesis generation, and analysis.

## Current MVP Status

The current build is optimized for local CPU development and hackathon validation.

Working now:

- FastAPI backend with OpenAPI/Swagger.
- Streamlit frontend in `frontend/`.
- Local Uvicorn runner for Windows.
- Docker Compose services for Postgres, Redis, Neo4j, Ollama, backend, and Celery worker.
- Startup Gemma warm-up endpoint and UI status display.
- Sidebar Gemma chat check for validating whether the configured model is responding.
- PDF upload endpoint and paper processing pipeline.
- Paper listing, details, and processing status polling.
- Semantic search endpoint.
- Knowledge graph endpoints.
- Hypothesis generation endpoint with fast MVP fallback and full Gemma mode.
- Cross-paper analysis endpoints for contradictions, unexplored connections, and landscape analysis.
- Project documentation for model/method responsibilities in `MODEL_METHOD_MAP.md`.

GPU note:

- `docker-compose.yml` is now GPU-ready for Ollama with `gpus: all`.
- The GPU host must have NVIDIA drivers, Docker, and NVIDIA Container Toolkit installed.
- On a GPU machine, Ollama should report non-zero `size_vram` from `GET /api/ps` after the model loads.

Known local CPU behavior:

- First Gemma/Ollama load can take 1-2 minutes.
- Once loaded, short checks can respond in a few seconds.
- Longer scientific answers can still take 60-120 seconds without GPU.
- The app now keeps Gemma warm with `GEMMA_KEEP_ALIVE=30m`.

## Architecture

```text
scientific-discovery-copilot/
├── backend/
│   ├── api/                 FastAPI app, dependencies, routes
│   ├── core/                settings, Gemma engine, embeddings, hypothesis orchestration
│   ├── ingestion/           PDF parsing, metadata, chunking, arXiv/PubMed fetchers
│   ├── retrieval/           ChromaDB vector store and semantic search
│   ├── reasoning/           entity extraction, contradiction detection, cross-paper reasoning
│   ├── graph/               Neo4j graph builder, queries, exporter
│   ├── models/              SQLAlchemy ORM models
│   ├── schemas/             Pydantic request/response schemas
│   ├── tasks/               Celery paper-processing task
│   ├── tests/               pytest test suite
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── frontend/
│   ├── app.py               Streamlit MVP UI
│   ├── requirements.txt
│   └── README.md
├── docker-compose.yml
├── MODEL_METHOD_MAP.md
├── .gitignore
└── README.md
```

## Core Technology Stack

- Backend: FastAPI, Uvicorn, Python 3.11+
- LLM runtime: Gemma via Ollama
- Embeddings: sentence-transformers, `BAAI/bge-large-en-v1.5`
- Vector DB: ChromaDB persistent local store
- Graph DB: Neo4j Community 5.x
- Relational DB: PostgreSQL 16 with SQLAlchemy async and asyncpg
- Queue/cache: Celery, Redis
- PDF parsing: PyMuPDF primary, pdfplumber fallback
- UI: Streamlit
- Containers: Docker and Docker Compose
- Tests/linting: pytest, pytest-asyncio, ruff, mypy

## GPU Ollama Setup

The recommended fast setup is to run Ollama on an NVIDIA GPU machine and point the backend to it.

### Option A: Run Ollama GPU with Docker Compose

On the GPU machine:

1. Install NVIDIA drivers.
2. Install Docker Desktop or Docker Engine.
3. Install NVIDIA Container Toolkit.
4. Clone this repo.
5. Start Ollama:

```powershell
docker compose up -d ollama
```

6. Pull the model:

```powershell
docker exec -it scientific-discovery-copilot-ollama-1 ollama pull gemma4:e2b
```

7. Verify GPU use after a request:

```powershell
Invoke-RestMethod http://localhost:11434/api/ps
```

Expected: `size_vram` should be greater than `0`.

### Option B: Use Your Friend's GPU Ollama Server

If your friend is already running Ollama on a GPU server, use their Ollama URL in `backend/.env`:

```env
OLLAMA_HOST=http://<FRIEND_GPU_OLLAMA_HOST>:11434
GEMMA_REASONING_MODEL=gemma4:e2b
GEMMA_LIGHT_MODEL=gemma4:e2b
GEMMA_KEEP_ALIVE=30m
GEMMA_NUM_THREAD=
```

Important: do not expose Ollama openly to the public internet. Prefer a private network, firewall allowlist, VPN, or SSH tunnel.

## Using Your Friend's Gemma/Ollama Server

If your friend has Gemma running on a faster machine or GPU server, you do not need to run Gemma locally. Point the backend to their Ollama host.

1. Ask your friend for their Ollama URL, for example:

```text
http://192.168.1.50:11434
```

or for a cloud GPU server:

```text
http://<gpu-server-ip>:11434
```

2. Copy the backend environment template:

```powershell
cd C:\Users\admin\Desktop\Gemma4\scientific-discovery-copilot\backend
Copy-Item .env.example .env
```

3. Edit `backend/.env`:

```env
OLLAMA_HOST=http://<FRIEND_OLLAMA_HOST>:11434
GEMMA_REASONING_MODEL=gemma4:e2b
GEMMA_LIGHT_MODEL=gemma4:e2b
GEMMA_KEEP_ALIVE=30m
GEMMA_NUM_THREAD=
```

4. If running with the Windows local script, also update `backend/run_local_uvicorn.ps1` or set the environment variable before launch:

```powershell
$env:OLLAMA_HOST = "http://<FRIEND_OLLAMA_HOST>:11434"
```

Important: Ollama should not be exposed openly on the public internet for long-running use. Prefer a private LAN, firewall rule, VPN, or SSH tunnel.

## Local Development Setup

### 1. Start infrastructure

Run Postgres, Redis, Neo4j, and GPU Ollama:

```powershell
cd C:\Users\admin\Desktop\Gemma4\scientific-discovery-copilot
docker compose up -d postgres redis neo4j ollama
```

This expects an NVIDIA GPU host with NVIDIA Container Toolkit because the Ollama service uses `gpus: all`.

If you are using your friend's remote Gemma/Ollama server, skip the local Ollama service:

```powershell
docker compose up -d postgres redis neo4j
```

### 2. Configure backend

```powershell
cd C:\Users\admin\Desktop\Gemma4\scientific-discovery-copilot\backend
Copy-Item .env.example .env
```

For local GPU Docker Ollama:

```env
OLLAMA_HOST=http://localhost:11434
```

For friend's remote Ollama:

```env
OLLAMA_HOST=http://<FRIEND_OLLAMA_HOST>:11434
```

### 3. Install backend dependencies

```powershell
cd C:\Users\admin\Desktop\Gemma4\scientific-discovery-copilot\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 4. Run backend

```powershell
cd C:\Users\admin\Desktop\Gemma4\scientific-discovery-copilot
.\backend\run_local_uvicorn.ps1
```

Backend URLs:

- Health: `http://127.0.0.1:8000/health`
- Swagger UI: `http://127.0.0.1:8000/docs`
- API base: `http://127.0.0.1:8000/api/v1`

When the backend starts, it begins warming Gemma. Check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/v1/agents/model-status
```

Expected loaded state:

```json
{
  "status": "loaded",
  "message": "Gemma model loaded successfully and is ready for backend functions."
}
```

### 5. Run frontend

```powershell
cd C:\Users\admin\Desktop\Gemma4\scientific-discovery-copilot\frontend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The Streamlit sidebar lets you:

- Check backend health.
- See Gemma model status.
- Manually warm the model.
- Send a small Gemma chat prompt.
- Upload papers and test the workflow.

## Docker Setup

Full stack:

```powershell
cd C:\Users\admin\Desktop\Gemma4\scientific-discovery-copilot
docker compose up --build
```

The full stack expects GPU-enabled Docker because the `ollama` service requests `gpus: all`.

If using a remote Ollama/Gemma server, edit `backend/.env` before starting:

```env
OLLAMA_HOST=http://<FRIEND_OLLAMA_HOST>:11434
```

Then run:

```powershell
docker compose up --build backend celery_worker postgres redis neo4j
```

## Main API Endpoints

Papers:

- `POST /api/v1/papers/upload`
- `POST /api/v1/papers/arxiv`
- `GET /api/v1/papers/`
- `GET /api/v1/papers/{paper_id}`
- `GET /api/v1/papers/{paper_id}/status`

Search:

- `POST /api/v1/search`

Hypotheses:

- `POST /api/v1/hypothesis/generate`
- `POST /api/v1/hypothesis/{hypothesis_id}/explain`
- `GET /api/v1/hypotheses/`
- `POST /api/v1/hypotheses/{id}/upvote`

Graph:

- `GET /api/v1/graph/{paper_id}`
- `GET /api/v1/graph/entity/{entity_name}`

Analysis:

- `POST /api/v1/analysis/contradictions`
- `POST /api/v1/analysis/connections`
- `POST /api/v1/analysis/landscape`

Gemma checks:

- `GET /api/v1/agents/model-status`
- `POST /api/v1/agents/model-warmup`
- `POST /api/v1/agents/chat`

## What Each Model/Method Does

See `MODEL_METHOD_MAP.md` for the detailed map. Short version:

- Gemma via Ollama: hypothesis generation, contradiction analysis, gap analysis, debate, explanation, optional entity refinement.
- BAAI/bge-large-en-v1.5: production semantic embeddings for ChromaDB search.
- all-MiniLM-L6-v2: lightweight sentence embeddings for semantic chunk splitting.
- PyMuPDF/pdfplumber: PDF text and structure extraction.
- ChromaDB: vector search over paper chunks.
- Neo4j: entity and paper knowledge graph.
- Postgres: paper, chunk, entity, relationship, contradiction, and hypothesis records.
- Redis/Celery: asynchronous paper processing.

## Planned WOW Features

The MVP already demonstrates the discovery workflow. Planned additions for the final product:

- GPU-backed Gemma inference for much faster full hypothesis generation.
- Streaming SSE/WebSocket responses for long Gemma tasks.
- Visual Cytoscape/D3 graph explorer in the frontend.
- Cross-domain bridge scoring between papers that do not cite each other.
- Contradiction resolution planner that proposes experiments to resolve conflicting claims.
- Multi-agent debate UI showing domain expert, methodology critic, devil's advocate, cross-domain linker, and synthesizer turns.
- Research landscape timeline with milestones, paradigm shifts, and open questions.
- Batch paper ingestion from arXiv/PubMed.
- Better scientific NER with scispaCy UMLS linking enabled in production mode.
- Citation trail visualization for every generated hypothesis.

## Git Hygiene

The repository intentionally ignores:

- `.env` files
- Python virtual environments
- local Docker volumes in `data/`
- uploads and PDFs
- Chroma/Postgres/Neo4j/Ollama runtime data
- Hugging Face/model caches
- Python cache folders
- logs and PID files

This keeps Git clean and lets collaborators pull source code without downloading local databases or model files.

## Quick Handoff Checklist

For a friend pulling this repository:

1. Clone the repo.
2. Copy `backend/.env.example` to `backend/.env`.
3. Set `OLLAMA_HOST` to their local or remote Ollama server.
4. Start Postgres, Redis, and Neo4j with Docker Compose.
5. Run the backend with `backend/run_local_uvicorn.ps1` or Docker.
6. Open Swagger at `http://127.0.0.1:8000/docs`.
7. Run Streamlit from `frontend/`.
8. Confirm the sidebar says `Model loaded successfully`.
