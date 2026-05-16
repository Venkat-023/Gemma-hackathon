from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from html import escape
import math
import time
from typing import Any

import requests
import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(page_title="Fast Discovery Pipeline", page_icon="FD", layout="wide")


def api_base() -> str:
    return st.session_state.get("fast_api_base", "http://127.0.0.1:8011").rstrip("/")


def format_seconds(seconds: float) -> str:
    seconds = int(max(0, seconds))
    minutes, remainder = divmod(seconds, 60)
    return f"{minutes}m {remainder}s" if minutes else f"{remainder}s"


def call_api(method: str, path: str, timeout: int = 120, **kwargs: Any) -> tuple[Any, str | None]:
    try:
        response = requests.request(method, f"{api_base()}{path}", timeout=timeout, **kwargs)
        if not response.ok:
            return None, f"{response.status_code}: {response.text}"
        return response.json() if response.text else {}, None
    except requests.RequestException as exc:
        return None, str(exc)


def timed_call(
    method: str,
    path: str,
    label: str,
    estimate_seconds: int,
    timeout: int,
    **kwargs: Any,
) -> tuple[Any, str | None, float]:
    started = time.perf_counter()
    progress = st.progress(0.0, text=f"{label}: starting...")
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(call_api, method, path, timeout=timeout, **kwargs)
        while not future.done():
            elapsed = time.perf_counter() - started
            remaining = max(0, estimate_seconds - elapsed)
            progress.progress(
                min(0.95, elapsed / max(estimate_seconds, 1)),
                text=f"{label}: elapsed {format_seconds(elapsed)} | estimated remaining {format_seconds(remaining)}",
            )
            time.sleep(1)
        data, error = future.result()
    elapsed = time.perf_counter() - started
    progress.progress(1.0, text=f"{label}: completed in {format_seconds(elapsed)}")
    return data, error, elapsed


def safe_list(payload: Any, key: str) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        value = payload.get(key, [])
        return value if isinstance(value, list) else []
    return []


def filter_rows(rows: list[dict[str, Any]], text: str) -> list[dict[str, Any]]:
    query = text.strip().lower()
    if not query:
        return rows
    return [row for row in rows if query in " ".join(str(value).lower() for value in row.values())]


def filter_graph_elements(elements: dict[str, Any], node_types: list[str], bridge_only: bool) -> dict[str, Any]:
    nodes = elements.get("elements", {}).get("nodes", [])
    edges = elements.get("elements", {}).get("edges", [])
    if node_types:
        nodes = [node for node in nodes if node.get("data", {}).get("type") in node_types]
    if bridge_only:
        nodes = [
            node
            for node in nodes
            if node.get("data", {}).get("type") == "PAPER" or node.get("data", {}).get("is_bridge")
        ]
    node_ids = {node.get("data", {}).get("id") for node in nodes}
    edges = [
        edge
        for edge in edges
        if edge.get("data", {}).get("source") in node_ids and edge.get("data", {}).get("target") in node_ids
    ]
    return {"elements": {"nodes": nodes, "edges": edges}, "stats": elements.get("stats", {})}


def render_graph(elements: dict[str, Any], height: int = 620) -> None:
    nodes = elements.get("elements", {}).get("nodes", [])
    edges = elements.get("elements", {}).get("edges", [])
    if not nodes:
        st.info("No graph nodes available yet. Upload a paper first.")
        return

    width = 980
    center_x, center_y = width / 2, height / 2
    paper_nodes = [node for node in nodes if node.get("data", {}).get("type") == "PAPER"]
    other_nodes = [node for node in nodes if node not in paper_nodes]
    positions: dict[str, tuple[float, float]] = {}

    for index, node in enumerate(paper_nodes):
        node_id = node["data"]["id"]
        y = 110 + index * max(90, min(140, (height - 220) / max(len(paper_nodes), 1)))
        positions[node_id] = (150, min(height - 80, y))

    radius_x = 330
    radius_y = max(170, height / 2 - 110)
    for index, node in enumerate(other_nodes):
        angle = (2 * math.pi * index) / max(len(other_nodes), 1)
        node_id = node["data"]["id"]
        positions[node_id] = (
            center_x + math.cos(angle) * radius_x,
            center_y + math.sin(angle) * radius_y,
        )

    edge_svg = []
    for edge in edges:
        data = edge.get("data", {})
        source = data.get("source")
        target = data.get("target")
        if source not in positions or target not in positions:
            continue
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        relation = escape(str(data.get("relation", ""))[:28])
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        edge_svg.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            'stroke="#B8C1CC" stroke-width="1.4" opacity="0.72" />'
            f'<text x="{mid_x:.1f}" y="{mid_y:.1f}" fill="#64748B" font-size="10" '
            f'text-anchor="middle">{relation}</text>'
        )

    node_svg = []
    legend_types = {}
    for node in nodes:
        data = node.get("data", {})
        node_id = data.get("id")
        if node_id not in positions:
            continue
        x, y = positions[node_id]
        node_type = data.get("type", "UNKNOWN")
        label = escape(str(data.get("label", node_id))[:32])
        color = data.get("color", "#64748B")
        radius = 20 if node_type != "PAPER" else 30
        legend_types[node_type] = color
        node_svg.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" '
            'stroke="#FFFFFF" stroke-width="2"><title>'
            f'{escape(str(data.get("label", node_id)))} ({escape(node_type)})</title></circle>'
            f'<text x="{x:.1f}" y="{y + radius + 15:.1f}" text-anchor="middle" '
            'fill="#0F172A" font-size="11" font-weight="600">'
            f'{label}</text>'
        )

    legend = []
    for index, (node_type, color) in enumerate(sorted(legend_types.items())):
        x = 24 + index * 145
        legend.append(
            f'<rect x="{x}" y="18" width="12" height="12" rx="3" fill="{color}" />'
            f'<text x="{x + 18}" y="29" fill="#334155" font-size="12">{escape(node_type)}</text>'
        )

    html = f"""
    <div style="width:100%; overflow:auto; border:1px solid #E2E8F0; border-radius:8px; background:#F8FAFC;">
      <svg viewBox="0 0 {width} {height}" width="100%" height="{height}" xmlns="http://www.w3.org/2000/svg">
        <rect width="{width}" height="{height}" fill="#F8FAFC"/>
        {''.join(legend)}
        <g>{''.join(edge_svg)}</g>
        <g>{''.join(node_svg)}</g>
      </svg>
    </div>
    """
    components.html(html, height=height + 20, scrolling=True)


def render_paper_connections(
    connections: list[dict[str, Any]],
    height: int = 520,
    max_connections: int = 12,
    max_concepts: int = 20,
    concept_filter: str = "",
) -> None:
    if not connections:
        st.info("No cross-paper connections yet. Upload at least two papers with shared concepts.")
        return
    width = 980
    query = concept_filter.strip().lower()
    if query:
        connections = [
            item
            for item in connections
            if query in " ".join(item.get("shared_entities", []) + item.get("shared_themes", [])).lower()
        ]
    top_connections = connections[:max_connections]
    paper_ids: list[Any] = []
    title_by_id = {}
    concept_to_papers: dict[str, set[Any]] = {}
    for item in top_connections:
        title_by_id[item.get("paper_a_id")] = item.get("paper_a_title", f"Paper {item.get('paper_a_id')}")
        title_by_id[item.get("paper_b_id")] = item.get("paper_b_title", f"Paper {item.get('paper_b_id')}")
        for key in ["paper_a_id", "paper_b_id"]:
            if item.get(key) not in paper_ids:
                paper_ids.append(item.get(key))
        concepts = item.get("shared_entities", [])[:8] or item.get("shared_themes", [])[:8]
        for concept in concepts:
            label = str(concept).strip()
            if not label:
                continue
            concept_to_papers.setdefault(label, set()).update([item.get("paper_a_id"), item.get("paper_b_id")])

    ranked_concepts = sorted(concept_to_papers, key=lambda concept: (-len(concept_to_papers[concept]), concept))[
        :max_concepts
    ]
    positions = {}
    for index, paper_id in enumerate(paper_ids):
        angle = (2 * math.pi * index) / max(len(paper_ids), 1)
        positions[f"P:{paper_id}"] = (
            width / 2 + math.cos(angle) * 330,
            height / 2 + math.sin(angle) * 160,
        )
    concept_positions = {}
    for index, concept in enumerate(ranked_concepts):
        angle = (2 * math.pi * index) / max(len(ranked_concepts), 1)
        concept_positions[concept] = (
            width / 2 + math.cos(angle) * 150,
            height / 2 + math.sin(angle) * 95,
        )
        positions[f"C:{concept}"] = concept_positions[concept]

    edges = []
    seen_edges = set()
    for concept in ranked_concepts:
        concept_node = f"C:{concept}"
        if concept_node not in positions:
            continue
        x2, y2 = positions[concept_node]
        for paper_id in concept_to_papers[concept]:
            paper_node = f"P:{paper_id}"
            if paper_node not in positions:
                continue
            key = (paper_node, concept_node)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            x1, y1 = positions[paper_node]
            weight = min(1.0, 0.35 + len(concept_to_papers[concept]) * 0.12)
            edges.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="#2563EB" stroke-width="{1.0 + weight * 3:.1f}" opacity="0.48" />'
            )

    concept_nodes = []
    for concept, (x, y) in concept_positions.items():
        connected = len(concept_to_papers[concept])
        radius = min(28, 14 + connected * 4)
        label = escape(concept[:26])
        concept_nodes.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="#F59E0B" '
            'stroke="#FFFFFF" stroke-width="2"><title>'
            f'{escape(concept)} connects {connected} papers</title></circle>'
            f'<text x="{x:.1f}" y="{y + radius + 13:.1f}" text-anchor="middle" '
            f'fill="#78350F" font-size="11" font-weight="700">{label}</text>'
        )

    paper_nodes = []
    for paper_id in paper_ids:
        paper_node = f"P:{paper_id}"
        if paper_node not in positions:
            continue
        x, y = positions[paper_node]
        label = escape(str(title_by_id.get(paper_id, f"Paper {paper_id}"))[:34])
        connected_concepts = sum(1 for concept in ranked_concepts if paper_id in concept_to_papers[concept])
        paper_nodes.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="34" fill="#4A90E2" stroke="#FFFFFF" stroke-width="2" />'
            f'<text x="{x:.1f}" y="{y - 3:.1f}" text-anchor="middle" fill="#FFFFFF" '
            f'font-size="12" font-weight="700">P{paper_id}</text>'
            f'<text x="{x:.1f}" y="{y + 52:.1f}" text-anchor="middle" fill="#0F172A" '
            f'font-size="12" font-weight="700">{label}</text>'
            f'<text x="{x:.1f}" y="{y + 68:.1f}" text-anchor="middle" fill="#475569" '
            f'font-size="10">{connected_concepts} shared concepts</text>'
        )

    direct_links = []
    for item in top_connections:
        a_node = f"P:{item.get('paper_a_id')}"
        b_node = f"P:{item.get('paper_b_id')}"
        if a_node not in positions or b_node not in positions:
            continue
        x1, y1 = positions[a_node]
        x2, y2 = positions[b_node]
        score = float(item.get("connection_score", 0))
        direct_links.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="#94A3B8" stroke-width="{1 + score * 2:.1f}" opacity="0.18" stroke-dasharray="6 7" />'
        )

    legend = (
        '<rect x="24" y="44" width="12" height="12" rx="3" fill="#4A90E2" />'
        '<text x="42" y="55" fill="#334155" font-size="12">Paper</text>'
        '<rect x="112" y="44" width="12" height="12" rx="3" fill="#F59E0B" />'
        '<text x="130" y="55" fill="#334155" font-size="12">Shared concept</text>'
        '<line x1="250" y1="50" x2="292" y2="50" stroke="#94A3B8" stroke-width="2" opacity="0.35" stroke-dasharray="6 7" />'
        '<text x="300" y="55" fill="#334155" font-size="12">Direct paper similarity</text>'
    )

    html = f"""
    <div style="width:100%; overflow:auto; border:1px solid #E2E8F0; border-radius:8px; background:#F8FAFC;">
      <svg viewBox="0 0 {width} {height}" width="100%" height="{height}" xmlns="http://www.w3.org/2000/svg">
        <rect width="{width}" height="{height}" fill="#F8FAFC"/>
        <text x="24" y="28" fill="#334155" font-size="14" font-weight="700">Inter-paper concept graph</text>
        {legend}
        <g>{''.join(direct_links)}</g>
        <g>{''.join(edges)}</g>
        <g>{''.join(concept_nodes)}</g>
        <g>{''.join(paper_nodes)}</g>
      </svg>
    </div>
    """
    components.html(html, height=height + 20, scrolling=True)


with st.sidebar:
    st.title("Fast Pipeline")
    st.text_input("Fast API base", key="fast_api_base", value="http://127.0.0.1:8011")
    health_data, health_error = call_api("GET", "/health", timeout=5)
    if health_error:
        st.error("Backend offline")
        st.caption(health_error)
    else:
        st.success("Backend online")
        h1, h2 = st.columns(2)
        h1.metric("Papers", health_data.get("papers", 0))
        h2.metric("Chunks", health_data.get("chunks_indexed", 0))
        st.caption(f"Retrieval: {health_data.get('retrieval_mode', 'unknown')}")
    if st.button("Check API", use_container_width=True):
        data, error = call_api("GET", "/health", timeout=10)
        if error:
            st.error(error)
        else:
            st.success("Fast backend OK")
            st.json(data)
    if st.button("Warm Gemma", use_container_width=True):
        data, error, _ = timed_call("POST", "/warmup", "Gemma warmup", 90, 240)
        if error:
            st.error(error)
        else:
            st.json(data)


st.title("Scientific Discovery Copilot - Fast Pipeline")
st.caption("Separate speed-first duplicate for testing faster paper ingestion and demo workflows.")

tabs = st.tabs(["Upload", "Search", "Hypotheses", "Discovery", "Graph", "Papers"])

with tabs[0]:
    st.subheader("Fast Upload")
    uploaded = st.file_uploader("Choose a research PDF", type=["pdf"])
    if uploaded and st.button("Upload and process fast", type="primary"):
        files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
        data, error, elapsed = timed_call("POST", "/upload", "Fast paper processing", 60, 600, files=files)
        if error:
            st.error(error)
        else:
            st.success(f"Processed in {elapsed:.1f}s")
            timings = data.get("timings", {})
            if timings:
                st.write("Timings")
                st.json(timings)
            st.write("Summary")
            st.json(data.get("summary", {}))
            st.write("Entities")
            st.json(data.get("entities", {}))
            st.write("Semantic relationships")
            st.dataframe(data.get("relationships", []), use_container_width=True)
            st.write("Themes")
            st.dataframe(data.get("themes", []), use_container_width=True)
            st.write("Research gaps")
            st.write(data.get("research_gaps", []))
            graph_data, graph_error = call_api("GET", f"/graph/{data.get('paper_id')}", timeout=30)
            if graph_error:
                st.warning(f"Graph preview unavailable: {graph_error}")
            else:
                st.write("Graph preview")
                render_graph(graph_data, height=520)

with tabs[1]:
    st.subheader("Search")
    query = st.text_input("Search query", value="deep learning lesion detection")
    if st.button("Search", type="primary"):
        data, error, elapsed = timed_call("POST", "/search", "Fast search", 10, 60, json={"query": query})
        if error:
            st.error(error)
        else:
            st.caption(f"UI wait: {elapsed:.1f}s")
            for result in data.get("results", []):
                with st.container(border=True):
                    st.markdown(f"**{result['metadata'].get('title', '')}**")
                    st.caption(f"{result['metadata'].get('section', '')} | score {result['similarity']}")
                    st.write(result["text"][:1200])

with tabs[2]:
    st.subheader("Hypotheses")
    h_query = st.text_input("Hypothesis query", value="deep learning lesion detection")
    count = st.slider("Number of hypotheses", 1, 5, 3)
    if st.button("Generate", type="primary"):
        payload = {"query": h_query, "num_hypotheses": count}
        data, error, elapsed = timed_call("POST", "/hypotheses", "Fast hypothesis generation", 90, 600, json=payload)
        if error:
            st.error(error)
        else:
            st.caption(f"Completed in {elapsed:.1f}s")
            st.json(data)

with tabs[3]:
    st.subheader("Scientific Discovery")
    topic = st.text_input("Discovery topic", value="deep learning lesion detection")
    control_a, control_b, control_c = st.columns([1.2, 1, 1])
    with control_a:
        view = st.selectbox(
            "Discovery view",
            ["Landscape", "Concept Bridges", "Research Gaps", "Hub Concepts", "Concept Clusters", "Inter-Paper Concepts"],
        )
    with control_b:
        row_filter = st.text_input("Filter concepts", value="")
    with control_c:
        top_n = st.slider("Top results", 5, 50, 15)

    if st.button("Refresh discovery dashboard", type="primary", use_container_width=True):
        st.session_state["fast_discovery_refresh"] = time.time()

    if view == "Landscape":
        data, error, elapsed = timed_call("POST", "/discovery", "Landscape analysis", 10, 60, json={"query": topic})
        if error:
            st.error(error)
        else:
            st.caption(f"Completed in {elapsed:.1f}s")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Papers", data.get("paper_count", 0))
            col_b.metric("Open questions", len(data.get("open_questions", [])))
            col_c.metric("Connections", len(data.get("cross_paper_connections", [])))
            st.info(data.get("scientific_discovery_summary", ""))
            st.write("Dominant themes")
            st.dataframe(filter_rows(data.get("dominant_themes", []), row_filter)[:top_n], use_container_width=True)
            st.write("Open questions")
            st.write(data.get("open_questions", [])[:top_n])

    elif view == "Inter-Paper Concepts":
        data, error = call_api("GET", "/connections", timeout=30)
        if error:
            st.error(error)
        else:
            connections = filter_rows(data.get("connections", []), row_filter)
            max_concepts = st.slider("Concept nodes", 4, 40, 18)
            render_paper_connections(
                connections,
                max_connections=top_n,
                max_concepts=max_concepts,
                concept_filter=row_filter,
            )
            st.write("Connection details")
            st.dataframe(connections[:top_n], use_container_width=True)

    elif view == "Concept Bridges":
        data, error = call_api("GET", f"/graph/bridges?top_n={top_n}", timeout=30)
        if error:
            st.error(error)
        else:
            rows = filter_rows(safe_list(data, "bridges"), row_filter)
            st.metric("Bridge concepts", len(rows))
            st.dataframe(rows, use_container_width=True)

    elif view == "Research Gaps":
        data, error = call_api("GET", f"/graph/gaps?top_n={top_n}", timeout=30)
        if error:
            st.error(error)
        else:
            rows = filter_rows(safe_list(data, "gaps"), row_filter)
            st.metric("Gap candidates", len(rows))
            st.dataframe(rows, use_container_width=True)

    elif view == "Hub Concepts":
        data, error = call_api("GET", f"/graph/hubs?top_n={top_n}", timeout=30)
        if error:
            st.error(error)
        else:
            rows = filter_rows(safe_list(data, "hubs"), row_filter)
            st.metric("Hub concepts", len(rows))
            st.dataframe(rows, use_container_width=True)

    elif view == "Concept Clusters":
        data, error = call_api("GET", "/graph/clusters", timeout=30)
        if error:
            st.error(error)
        else:
            rows = filter_rows(safe_list(data, "clusters"), row_filter)[:top_n]
            st.metric("Concept clusters", len(rows))
            st.dataframe(rows, use_container_width=True)

with tabs[4]:
    st.subheader("Graph")
    papers_payload, papers_error = call_api("GET", "/papers", timeout=30)
    papers = safe_list(papers_payload, "papers") if not papers_error else []
    paper_options = {"All papers": 0}
    for paper in papers:
        paper_options[f"{paper.get('paper_id')}: {paper.get('title', 'Untitled')}"] = paper.get("paper_id")

    graph_controls = st.columns([1.4, 1.4, 1, 1])
    with graph_controls[0]:
        selected_label = st.selectbox("Graph scope", list(paper_options.keys()))
        selected_paper_id = paper_options[selected_label]
    with graph_controls[1]:
        node_types = st.multiselect(
            "Node types",
            ["PAPER", "METHOD", "DISEASE", "CONCEPT", "GENE", "PROTEIN", "CHEMICAL", "CLUSTER"],
            default=[],
        )
    with graph_controls[2]:
        bridge_only = st.toggle("Bridge concepts only", value=False)
    with graph_controls[3]:
        graph_height = st.slider("Height", 420, 820, 620, step=40)

    action_a, action_b, action_c = st.columns(3)
    with action_a:
        load_graph = st.button("Render graph", type="primary", use_container_width=True)
    with action_b:
        load_connections = st.button("Render inter-paper concepts", use_container_width=True)
    with action_c:
        load_export = st.button("Analytics export", use_container_width=True)

    if load_graph:
        path = "/graph" if selected_paper_id == 0 else f"/graph/{selected_paper_id}"
        data, error = call_api("GET", path, timeout=30)
        if error:
            st.error(error)
        else:
            filtered = filter_graph_elements(data, node_types, bridge_only)
            stats = data.get("stats", {})
            m1, m2, m3 = st.columns(3)
            m1.metric("Nodes", len(filtered.get("elements", {}).get("nodes", [])))
            m2.metric("Edges", len(filtered.get("elements", {}).get("edges", [])))
            m3.metric("Bridge concepts", stats.get("bridge_concepts", 0))
            render_graph(filtered, height=graph_height)
            with st.expander("Raw graph JSON"):
                st.json(filtered)

    if load_connections:
        data, error = call_api("GET", "/connections", timeout=30)
        if error:
            st.error(error)
        else:
            concept_filter = st.text_input("Connection concept filter", key="graph_connection_filter")
            max_concepts = st.slider("Max concept nodes", 4, 40, 20, key="graph_max_concepts")
            render_paper_connections(
                data.get("connections", []),
                height=graph_height,
                max_concepts=max_concepts,
                concept_filter=concept_filter,
            )
            with st.expander("Connection table"):
                st.dataframe(filter_rows(data.get("connections", []), concept_filter), use_container_width=True)

    if load_export:
        data, error = call_api("GET", "/graph/export", timeout=30)
        if error:
            st.error(error)
        else:
            stats = data.get("stats", {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Papers", stats.get("total_papers", 0))
            c2.metric("Nodes", stats.get("total_nodes", 0))
            c3.metric("Edges", stats.get("total_edges", 0))
            c4.metric("Bridge concepts", stats.get("bridge_concepts", 0))
            with st.expander("Graph analytics JSON"):
                st.json(data)

with tabs[5]:
    st.subheader("Papers")
    if st.button("Refresh papers"):
        data, error = call_api("GET", "/papers", timeout=30)
        if error:
            st.error(error)
        else:
            st.json(data)
