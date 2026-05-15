from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time
from typing import Any

import requests
import streamlit as st


st.set_page_config(
    page_title="Scientific Discovery Copilot",
    page_icon="SC",
    layout="wide",
)


def api_base() -> str:
    return st.session_state.get("api_base", "http://127.0.0.1:8000/api/v1").rstrip("/")


def call_api(method: str, path: str, **kwargs: Any) -> tuple[dict[str, Any] | list[Any] | None, str | None]:
    url = f"{api_base()}{path}"
    try:
        response = requests.request(method, url, timeout=kwargs.pop("timeout", 90), **kwargs)
        if not response.ok:
            return None, f"{response.status_code}: {response.text}"
        if not response.text:
            return {}, None
        return response.json(), None
    except requests.RequestException as exc:
        return None, str(exc)


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, remainder = divmod(seconds, 60)
    if minutes:
        return f"{minutes}m {remainder}s"
    return f"{remainder}s"


def timed_api_call(
    method: str,
    path: str,
    label: str,
    estimate_seconds: int,
    timeout: int,
    **kwargs: Any,
) -> tuple[dict[str, Any] | list[Any] | None, str | None, float]:
    started = time.perf_counter()
    status = st.empty()
    progress = st.progress(0.0, text=f"{label}: starting...")
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(call_api, method, path, timeout=timeout, **kwargs)
        while not future.done():
            elapsed = time.perf_counter() - started
            remaining = max(0, estimate_seconds - int(elapsed))
            progress.progress(
                min(0.95, elapsed / max(estimate_seconds, 1)),
                text=f"{label}: elapsed {format_seconds(elapsed)} | estimated remaining {format_seconds(remaining)}",
            )
            if elapsed > estimate_seconds:
                status.warning(
                    f"{label} is taking longer than the estimate. Still waiting; elapsed {format_seconds(elapsed)}."
                )
            time.sleep(1)
        data, error = future.result()
    elapsed = time.perf_counter() - started
    progress.progress(1.0, text=f"{label}: completed in {format_seconds(elapsed)}")
    return data, error, elapsed


def poll_paper_processing(paper_id: int, estimate_seconds: int = 240) -> None:
    started = time.perf_counter()
    progress = st.progress(0.0, text="Paper processing: waiting for backend status...")
    status_box = st.empty()
    for _ in range(180):
        data, error = call_api("GET", f"/papers/{paper_id}/status", timeout=30)
        elapsed = time.perf_counter() - started
        remaining = max(0, estimate_seconds - int(elapsed))
        if error:
            st.error(error)
            break
        status = data.get("status", "unknown")
        chunks = data.get("chunks_created", 0)
        entities = data.get("entities_extracted", 0)
        graph_ready = data.get("graph_built", False)
        progress.progress(
            min(0.98, elapsed / estimate_seconds),
            text=(
                f"Paper processing: {status} | elapsed {format_seconds(elapsed)} | "
                f"estimated remaining {format_seconds(remaining)}"
            ),
        )
        status_box.info(
            f"Status: {status} | chunks: {chunks} | entities: {entities} | graph ready: {graph_ready}"
        )
        if elapsed > estimate_seconds:
            st.warning(
                f"Paper processing is taking longer than the estimate. Still waiting; elapsed {format_seconds(elapsed)}."
            )
        if status in {"completed", "failed"}:
            progress.progress(1.0, text=f"Paper processing {status} in {format_seconds(elapsed)}")
            status_box.json(data)
            break
        time.sleep(5)


def render_status_badge(status: str) -> None:
    color = {
        "completed": "green",
        "processing": "orange",
        "pending": "blue",
        "failed": "red",
    }.get(status, "gray")
    st.markdown(f":{color}[{status}]")


def list_papers() -> list[dict[str, Any]]:
    data, error = call_api("GET", "/papers/", params={"limit": 100, "offset": 0}, timeout=30)
    if error:
        st.warning(f"Could not load papers: {error}")
        return []
    return data if isinstance(data, list) else data.get("value", []) if isinstance(data, dict) else []


def paper_selector(papers: list[dict[str, Any]], label: str, multi: bool = True) -> list[int]:
    options = {f'{paper["id"]}: {paper.get("title") or "Untitled"}': paper["id"] for paper in papers}
    if not options:
        st.info("No papers available yet. Upload a PDF first.")
        return []
    if multi:
        selected = st.multiselect(label, list(options.keys()), default=list(options.keys())[:1])
        return [options[item] for item in selected]
    selected = st.selectbox(label, list(options.keys()))
    return [options[selected]]


with st.sidebar:
    st.title("Discovery Copilot")
    st.text_input("Backend API base", key="api_base", value="http://127.0.0.1:8000/api/v1")
    health_url = api_base().replace("/api/v1", "/health")
    if st.button("Check backend", use_container_width=True):
        try:
            health = requests.get(health_url, timeout=10).json()
            st.success(f"Backend OK: {health}")
        except requests.RequestException as exc:
            st.error(f"Backend not reachable: {exc}")
    st.caption("Use this UI to upload papers and smoke-test backend modules before switching models.")

    st.divider()
    st.subheader("Gemma Model")
    model_status, model_error = call_api("GET", "/agents/model-status", timeout=10)
    if model_error:
        st.error(f"Model status unavailable: {model_error}")
    elif isinstance(model_status, dict):
        status = model_status.get("status", "unknown")
        model_name = model_status.get("model") or "unknown"
        duration = model_status.get("duration_seconds")
        if status == "loaded":
            st.success(f"Model loaded successfully: {model_name}")
        elif status == "loading":
            st.info(f"Loading model: {model_name}. First CPU load can take 1-2 minutes.")
        elif status == "warning":
            st.warning(f"Model loaded with warning: {model_status.get('message')}")
        elif status == "failed":
            st.error(f"Model warm-up failed: {model_status.get('detail') or model_status.get('message')}")
        else:
            st.warning(model_status.get("message", "Model warm-up has not started."))
        if duration:
            st.caption(f"Warm-up time: {duration}s")
    warm_col1, warm_col2 = st.columns(2)
    with warm_col1:
        if st.button("Warm up", use_container_width=True):
            data, error = call_api("POST", "/agents/model-warmup", timeout=30)
            if error:
                st.error(error)
            else:
                st.json(data)
    with warm_col2:
        if st.button("Refresh", use_container_width=True):
            st.rerun()

    st.divider()
    st.subheader("Gemma Chat Check")
    if "gemma_chat_history" not in st.session_state:
        st.session_state["gemma_chat_history"] = []
    chat_prompt = st.text_area(
        "Message Gemma",
        value="Reply with one sentence: are you working?",
        height=90,
        key="gemma_chat_prompt",
    )
    chat_col1, chat_col2 = st.columns(2)
    with chat_col1:
        send_chat = st.button("Send", use_container_width=True)
    with chat_col2:
        clear_chat = st.button("Clear", use_container_width=True)
    if clear_chat:
        st.session_state["gemma_chat_history"] = []
    if send_chat and chat_prompt.strip():
        payload = {"message": chat_prompt.strip(), "temperature": 0.25, "max_tokens": 512, "timeout_seconds": 360}
        data, error, elapsed = timed_api_call(
            "POST",
            "/agents/chat",
            "Gemma chat",
            estimate_seconds=120,
            timeout=380,
            json=payload,
        )
        if error:
            st.error(error)
        else:
            st.session_state["gemma_chat_history"].append(
                {
                    "user": chat_prompt.strip(),
                    "assistant": data.get("response", ""),
                    "model": data.get("model", ""),
                    "duration_seconds": data.get("duration_seconds", elapsed),
                }
            )
    for item in reversed(st.session_state["gemma_chat_history"][-3:]):
        st.markdown(f"**You:** {item['user']}")
        st.markdown(f"**Gemma ({item['model']}, {item['duration_seconds']}s):** {item['assistant']}")


st.title("Scientific Discovery Copilot")
st.caption("MVP test console for paper ingestion, retrieval, graph, hypotheses, and analysis.")

tabs = st.tabs(["Upload", "Papers", "Search", "Graph", "Hypotheses", "Analysis"])


with tabs[0]:
    st.subheader("Upload Research Paper")
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])
    auto_wait = st.checkbox("Wait and show processing time after upload", value=True)
    if uploaded and st.button("Upload and process", type="primary"):
        files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
        data, error, _ = timed_api_call(
            "POST",
            "/papers/upload",
            "PDF upload",
            estimate_seconds=20,
            timeout=120,
            files=files,
        )
        if error:
            st.error(error)
        else:
            st.success("Upload accepted")
            st.json(data)
            st.session_state["last_uploaded_paper_id"] = data.get("paper_id")
            if auto_wait and data.get("paper_id"):
                poll_paper_processing(data["paper_id"])

    last_id = st.session_state.get("last_uploaded_paper_id")
    if last_id:
        st.divider()
        st.write(f"Last uploaded paper: `{last_id}`")
        if st.button("Poll latest status"):
            data, error = call_api("GET", f"/papers/{last_id}/status", timeout=30)
            if error:
                st.error(error)
            else:
                st.json(data)
        if st.button("Auto-poll until complete"):
            poll_paper_processing(last_id)


with tabs[1]:
    st.subheader("Indexed Papers")
    papers = list_papers()
    if papers:
        for paper in papers:
            with st.container(border=True):
                left, right = st.columns([4, 1])
                with left:
                    st.markdown(f"**{paper['id']}. {paper.get('title') or 'Untitled'}**")
                    authors = ", ".join(paper.get("authors") or [])
                    st.caption(f"{paper.get('publication_year') or 'Unknown year'} | {authors or 'No authors parsed'}")
                with right:
                    render_status_badge(paper.get("processing_status", "unknown"))
                if st.button("Open details", key=f"detail-{paper['id']}"):
                    data, error, _ = timed_api_call(
                        "GET",
                        f"/papers/{paper['id']}",
                        "Paper details",
                        estimate_seconds=5,
                        timeout=30,
                    )
                    if error:
                        st.error(error)
                    else:
                        st.json(data)


with tabs[2]:
    st.subheader("Semantic Search")
    papers = list_papers()
    selected_ids = paper_selector(papers, "Limit to papers", multi=True)
    query = st.text_input("Search query", value="deep learning lesion detection endoscopy")
    n_results = st.slider("Results", 1, 20, 5)
    if st.button("Run search", type="primary"):
        payload = {"query": query, "paper_ids": selected_ids or None, "n_results": n_results}
        data, error, elapsed = timed_api_call(
            "POST",
            "/search",
            "Semantic search",
            estimate_seconds=20,
            timeout=90,
            json=payload,
        )
        if error:
            st.error(error)
        else:
            st.metric("Query time", f"{data.get('query_time_ms', 0)} ms")
            st.caption(f"Total UI wait: {elapsed:.1f}s")
            for item in data.get("results", []):
                with st.container(border=True):
                    st.markdown(f"**{item['id']}** | similarity `{item['similarity_score']:.3f}`")
                    st.caption(item.get("metadata", {}))
                    st.write(item.get("text", "")[:1200])


with tabs[3]:
    st.subheader("Knowledge Graph")
    papers = list_papers()
    selected_ids = paper_selector(papers, "Paper for graph", multi=False)
    if selected_ids and st.button("Load graph"):
        data, error, _ = timed_api_call(
            "GET",
            f"/graph/{selected_ids[0]}",
            "Knowledge graph",
            estimate_seconds=10,
            timeout=60,
        )
        if error:
            st.error(error)
        else:
            nodes = data.get("nodes", [])
            edges = data.get("edges", [])
            col1, col2 = st.columns(2)
            col1.metric("Nodes", len(nodes))
            col2.metric("Edges", len(edges))
            st.write("Nodes")
            st.dataframe(nodes, use_container_width=True)
            if edges:
                st.write("Edges")
                st.dataframe(edges, use_container_width=True)

    entity_name = st.text_input("Entity neighborhood", value="Deep Learning")
    if st.button("Load entity neighborhood"):
        data, error, _ = timed_api_call(
            "GET",
            f"/graph/entity/{entity_name}",
            "Entity neighborhood",
            estimate_seconds=10,
            timeout=60,
        )
        if error:
            st.error(error)
        else:
            st.json(data)


with tabs[4]:
    st.subheader("Hypothesis Generation")
    papers = list_papers()
    selected_ids = paper_selector(papers, "Use papers", multi=True)
    h_query = st.text_input("Research query", value="deep learning lesion detection endoscopy")
    num_h = st.slider("Hypotheses", 1, 5, 1)
    use_fast_fallback = st.checkbox(
        "Use fast MVP fallback",
        value=True,
        help="Turn this off to wait for real Gemma generation. On this machine it can take several minutes or timeout.",
    )
    if not use_fast_fallback:
        st.info(
            "Full Gemma mode is enabled. Keep this tab open; the timer below will keep updating while the backend works."
        )
    if st.button("Generate hypothesis", type="primary"):
        payload = {
            "query": h_query,
            "paper_ids": selected_ids or None,
            "num_hypotheses": num_h,
            "use_fast_fallback": use_fast_fallback,
        }
        estimate = 25 if use_fast_fallback else 240
        timeout = 120 if use_fast_fallback else 900
        data, error, elapsed = timed_api_call(
            "POST",
            "/hypothesis/generate",
            "Hypothesis generation",
            estimate_seconds=estimate,
            timeout=timeout,
            json=payload,
        )
        if error:
            st.error(error)
        else:
            st.caption(f"Completed in {elapsed:.1f}s")
            for warning in data.get("warnings", []):
                st.warning(warning)
            for item in data.get("hypotheses", []):
                with st.container(border=True):
                    st.markdown(f"**Hypothesis #{item.get('id', '?')}**")
                    st.write(item.get("hypothesis"))
                    st.progress(float(item.get("confidence", 0)), text=f"Confidence {item.get('confidence', 0):.2f}")
                    st.write(item.get("reasoning"))
                    st.write("Suggested experiments")
                    st.write(item.get("suggested_experiments", []))

    st.divider()
    st.write("Stored hypotheses")
    if st.button("Refresh stored hypotheses"):
        data, error, _ = timed_api_call(
            "GET",
            "/hypotheses/",
            "Stored hypotheses",
            estimate_seconds=5,
            timeout=30,
            params={"limit": 20, "offset": 0},
        )
        if error:
            st.error(error)
        else:
            st.dataframe(data, use_container_width=True)


with tabs[5]:
    st.subheader("Cross-Paper Analysis")
    papers = list_papers()
    selected_ids = paper_selector(papers, "Papers for contradiction check", multi=True)
    topic = st.text_input("Analysis topic", value="deep learning lesion detection")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Contradictions", use_container_width=True):
            if len(selected_ids) < 2:
                st.warning("Select at least two papers.")
            else:
                payload = {"topic": topic, "paper_ids": selected_ids}
                data, error, _ = timed_api_call(
                    "POST",
                    "/analysis/contradictions",
                    "Contradiction analysis",
                    estimate_seconds=120,
                    timeout=600,
                    json=payload,
                )
                if error:
                    st.error(error)
                else:
                    st.json(data)
    with col2:
        if st.button("Connections", use_container_width=True):
            if not selected_ids:
                st.warning("Select one paper.")
            else:
                payload = {"paper_id": selected_ids[0]}
                data, error, _ = timed_api_call(
                    "POST",
                    "/analysis/connections",
                    "Connection discovery",
                    estimate_seconds=60,
                    timeout=300,
                    json=payload,
                )
                if error:
                    st.error(error)
                else:
                    st.json(data)
    with col3:
        if st.button("Landscape", use_container_width=True):
            payload = {"topic": topic}
            data, error, _ = timed_api_call(
                "POST",
                "/analysis/landscape",
                "Landscape analysis",
                estimate_seconds=90,
                timeout=600,
                json=payload,
            )
            if error:
                st.error(error)
            else:
                st.json(data)
