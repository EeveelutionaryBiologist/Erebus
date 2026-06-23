#!/usr/bin/env python3
"""
Interactive graph visualizer for the Erebus knowledge graph.

Reads KnowledgeGraph/knowledge_graph.json (or temporal_graph.json with --temporal),
enriches nodes with SQLite group data, and writes a self-contained HTML file
rendered with vis.js via pyvis.

Usage:
    python scripts/visualize_graph.py [--temporal] [--output graph.html] [--open]

Requires: pip install pyvis
"""

import argparse
import json
import os
import sqlite3
import webbrowser
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph

try:
    from pyvis.network import Network
except ImportError:
    print("ERROR: pyvis is not installed.  Run:  pip install pyvis")
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# Paths (relative to repo root — script is in scripts/)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
KG_PATH = REPO_ROOT / "KnowledgeGraph" / "knowledge_graph.json"
TG_PATH = REPO_ROOT / "KnowledgeGraph" / "temporal_graph.json"
DB_PATH = REPO_ROOT / "DB" / "metadata.db"

# Group → color mapping (extend as needed)
GROUP_COLORS = {
    "family":      "#e76f51",
    "friends":     "#2a9d8f",
    "colleagues":  "#457b9d",
    "work":        "#457b9d",
    "places":      "#8ecae6",
    "education":   "#a8dadc",
    "pets":        "#f4a261",
    "hobbies":     "#b7e4c7",
    "health":      "#d62828",
    "finance":     "#ffb703",
    "misc":        "#adb5bd",
}
DEFAULT_COLOR = "#adb5bd"
TEMPORAL_EDGE_COLORS = {
    "PRECEDED_BY":    "#e63946",
    "CONCURRENT_WITH": "#457b9d",
}


# ---------------------------------------------------------------------------
# Load graph
# ---------------------------------------------------------------------------

def load_graph(path: Path, is_temporal: bool) -> nx.MultiDiGraph:
    if not path.exists():
        print(f"ERROR: Graph file not found: {path}")
        raise SystemExit(1)
    with open(path) as f:
        data = json.load(f)
    G = json_graph.node_link_graph(data)
    if not isinstance(G, nx.MultiDiGraph):
        G = nx.MultiDiGraph(G)
    node_count = G.number_of_nodes()
    edge_count = G.number_of_edges()
    label = "temporal graph" if is_temporal else "knowledge graph"
    print(f"Loaded {label}: {node_count} nodes, {edge_count} edges")
    return G


# ---------------------------------------------------------------------------
# Load SQLite metadata (entity groups + hit counts)
# ---------------------------------------------------------------------------

def load_entity_metadata(db_path: Path) -> dict[str, dict]:
    """Returns {entity_id: {name, hit_count, groups: [str]}}"""
    if not db_path.exists():
        return {}
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # Entity base data
        entities: dict[str, dict] = {}
        for row in cur.execute("SELECT id, canonical_name, hit_count FROM entities"):
            entities[row["id"]] = {
                "name": row["canonical_name"],
                "hit_count": row["hit_count"],
                "groups": [],
            }

        # Group assignments
        for row in cur.execute("""
            SELECT eg.entity_id, g.name
            FROM entity_groups eg
            JOIN groups g ON g.id = eg.group_id
        """):
            if row["entity_id"] in entities:
                entities[row["entity_id"]]["groups"].append(row["name"])

        conn.close()
        return entities
    except sqlite3.Error as e:
        print(f"Warning: could not read SQLite metadata: {e}")
        return {}


def load_fact_metadata(db_path: Path) -> dict[str, dict]:
    """Returns {fact_id: {text, temporal_status, valid_period}} for temporal graph."""
    if not db_path.exists():
        return {}
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        facts = {}
        for row in cur.execute(
            "SELECT id, content, temporal_status, valid_period FROM atomic_facts"
        ):
            facts[row["id"]] = {
                "text": row["content"],
                "temporal_status": row["temporal_status"] or "current",
                "valid_period": row["valid_period"],
            }
        conn.close()
        return facts
    except sqlite3.Error as e:
        print(f"Warning: could not read fact metadata: {e}")
        return {}


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def group_to_color(groups: list[str]) -> str:
    for g in groups:
        key = g.lower()
        if key in GROUP_COLORS:
            return GROUP_COLORS[key]
        for k in GROUP_COLORS:
            if k in key or key in k:
                return GROUP_COLORS[k]
    return DEFAULT_COLOR


def hit_count_to_size(hit_count: int) -> int:
    return max(10, min(50, 10 + hit_count * 4))


def temporal_status_to_color(status: str) -> str:
    return {"current": "#2a9d8f", "historical": "#e76f51", "uncertain": "#ffb703"}.get(
        status, DEFAULT_COLOR
    )


# ---------------------------------------------------------------------------
# Build pyvis network — entity (knowledge) graph
# ---------------------------------------------------------------------------

def build_entity_network(G: nx.MultiDiGraph, entity_meta: dict) -> Network:
    net = Network(
        height="90vh",
        width="100%",
        directed=True,
        bgcolor="#1a1a2e",
        font_color="#e0e0e0",
        notebook=False,
    )
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=150)

    for node_id in G.nodes:
        meta = entity_meta.get(node_id, {})
        name = meta.get("name") or G.nodes[node_id].get("name", node_id[:8])
        hit_count = meta.get("hit_count", 0)
        groups = meta.get("groups", [])
        color = group_to_color(groups)
        size = hit_count_to_size(hit_count)
        group_str = ", ".join(groups) if groups else "ungrouped"
        tooltip = f"<b>{name}</b><br>Groups: {group_str}<br>Hit count: {hit_count}"
        net.add_node(node_id, label=name, color=color, size=size, title=tooltip)

    for src, dst, data in G.edges(data=True):
        relation = data.get("relation", "?")
        fact_ids = data.get("source_fact_ids", [])
        tooltip = f"{relation}<br>Backed by {len(fact_ids)} fact(s)"
        net.add_edge(src, dst, label=relation, title=tooltip, color="#8888aa", arrows="to")

    return net


# ---------------------------------------------------------------------------
# Build pyvis network — temporal graph
# ---------------------------------------------------------------------------

def build_temporal_network(G: nx.MultiDiGraph, fact_meta: dict) -> Network:
    net = Network(
        height="90vh",
        width="100%",
        directed=True,
        bgcolor="#1a1a2e",
        font_color="#e0e0e0",
        notebook=False,
    )
    net.barnes_hut(gravity=-5000, central_gravity=0.2, spring_length=200)

    for node_id in G.nodes:
        meta = fact_meta.get(node_id, {})
        text = meta.get("text") or G.nodes[node_id].get("name", node_id[:8])
        status = meta.get("temporal_status", "current")
        period = meta.get("valid_period")
        color = temporal_status_to_color(status)
        label = text[:40] + "…" if len(text) > 40 else text
        period_str = f"<br>Period: {period}" if period else ""
        tooltip = f"<b>[{status}]</b>{period_str}<br>{text}"
        net.add_node(node_id, label=label, color=color, size=16, title=tooltip)

    for src, dst, data in G.edges(data=True):
        relation = data.get("relation", "?")
        color = TEMPORAL_EDGE_COLORS.get(relation, "#8888aa")
        net.add_edge(src, dst, label=relation, title=relation, color=color, arrows="to")

    return net


# ---------------------------------------------------------------------------
# Inject a legend into the HTML
# ---------------------------------------------------------------------------

def inject_legend(html: str, is_temporal: bool) -> str:
    if is_temporal:
        items = [
            ("#2a9d8f", "current"),
            ("#e76f51", "historical"),
            ("#ffb703", "uncertain"),
            ("#e63946", "PRECEDED_BY edge"),
            ("#457b9d", "CONCURRENT_WITH edge"),
        ]
    else:
        items = [(color, name) for name, color in GROUP_COLORS.items()]

    legend_html = """
<div id="legend" style="position:fixed;top:12px;left:12px;background:#2a2a4a;
     border:1px solid #555;border-radius:8px;padding:12px 16px;z-index:999;
     font-family:sans-serif;font-size:13px;color:#ddd;min-width:140px">
  <b style="display:block;margin-bottom:8px;">Legend</b>
"""
    for color, label in items:
        legend_html += (
            f'  <div style="display:flex;align-items:center;gap:8px;margin:4px 0">'
            f'<span style="width:14px;height:14px;border-radius:50%;'
            f'background:{color};flex-shrink:0"></span>{label}</div>\n'
        )
    legend_html += "</div>\n"

    return html.replace("</body>", legend_html + "</body>", 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Erebus knowledge graph visualizer")
    parser.add_argument(
        "--temporal", action="store_true", help="Visualize temporal graph instead of entity graph"
    )
    parser.add_argument("--output", default="", help="Output HTML file (default: graph.html or temporal_graph.html)")
    parser.add_argument("--open", action="store_true", help="Open the HTML file in the default browser")
    args = parser.parse_args()

    is_temporal = args.temporal
    graph_path = TG_PATH if is_temporal else KG_PATH
    default_out = "temporal_graph.html" if is_temporal else "graph.html"
    out_path = Path(args.output or default_out)

    G = load_graph(graph_path, is_temporal)

    if G.number_of_nodes() == 0:
        print("Graph is empty — run the server and ingest some data first.")
        return

    db_meta: dict
    if is_temporal:
        db_meta = load_fact_metadata(DB_PATH)
        net = build_temporal_network(G, db_meta)
    else:
        db_meta = load_entity_metadata(DB_PATH)
        net = build_entity_network(G, db_meta)

    # Write raw HTML from pyvis then inject legend
    tmp_path = out_path.with_suffix(".tmp.html")
    net.write_html(str(tmp_path))
    raw_html = tmp_path.read_text()
    tmp_path.unlink()

    final_html = inject_legend(raw_html, is_temporal)
    out_path.write_text(final_html)

    abs_path = out_path.resolve()
    print(f"Wrote {abs_path}")

    if args.open:
        webbrowser.open(abs_path.as_uri())


if __name__ == "__main__":
    main()
