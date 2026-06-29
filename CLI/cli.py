#!/usr/bin/env python3
"""
Interactive CLI for the Erebus memory server.

Usage:
    python CLI/cli.py [--url http://localhost:8000]

Commands inside the loop:
    context  <query>             Fast-path retrieval (no LLM, current facts only)
    search   <query>             Deep search with entity extraction + KG traversal
    chain    <query|fact_id>     Temporal predecessor chain (UUID → by ID, else → query)
    add      <text>              Add a text snippet to memory
    learn    <file>              Ingest a text file via /memory/learn
    consolidate                  Run the memory consolidation pass
    visualize [--temporal]       Export graph.html and open in browser
    all      [raw|fact|entity]   Dump memory records (truncated to 50)
    clear                        Wipe all memory (asks for confirmation)
    help                         Show this message
    exit / quit                  Exit
    
"""

import argparse
import re
import shlex
import sys
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------------

BASE_URL = "http://localhost:8000"
POLL_INTERVAL = 5.0   # seconds between HTTP task-status requests
SPINNER_INTERVAL = 0.2  # seconds between visual spinner updates

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
)

HELP_TEXT = """\
Commands:
  context  <query>             Fast-path retrieval (no LLM, current facts only)
  search   <query>             Deep search with entity extraction + KG traversal
  chain    <query|fact_id>     Temporal predecessor chain (UUID → by ID, else → query)
  add      <text>              Add a text snippet to memory
  learn    <file>              Ingest a text file via /memory/learn
  consolidate                  Run the memory consolidation pass
  visualize [--temporal]       Export graph.html and open in browser
  all      [raw|fact|entity]   Dump memory records (first 50)
  clear                        Wipe all memory (asks for confirmation)
  help / ?                     Show this message
  exit / quit                  Exit"""


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get(path: str, **kwargs) -> dict:
    resp = requests.get(f"{BASE_URL}{path}", timeout=15, **kwargs)
    resp.raise_for_status()
    return resp.json()


def _post(path: str, body: dict | None = None, timeout: int = 30) -> dict:
    resp = requests.post(f"{BASE_URL}{path}", json=body or {}, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _delete(path: str) -> dict:
    resp = requests.delete(f"{BASE_URL}{path}", timeout=15)
    resp.raise_for_status()
    return resp.json()


def _wait_for_task(task_id: str, label: str = "Working") -> dict:
    """Poll GET /memory/task/{task_id} until complete, showing a spinner.

    HTTP requests are made every POLL_INTERVAL seconds; the spinner updates
    every SPINNER_INTERVAL seconds so the server log isn't flooded.
    """
    spinner = ["|", "/", "-", "\\"]
    tick = 0
    last_poll = 0.0
    data: dict = {}
    while True:
        now = time.monotonic()
        if now - last_poll >= POLL_INTERVAL:
            data = _get(f"/memory/task/{task_id}")
            last_poll = now
            status = data.get("status")
            if status in ("completed", "failed"):
                print(f"\r{label}... done.     ")
                return data
        print(f"\r{label}... {spinner[tick % 4]}", end="", flush=True)
        tick += 1
        time.sleep(SPINNER_INTERVAL)


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def context(query: str) -> None:
    data = _post("/memory/context", {"query": query, "top_k": 5})
    results = data.get("results", [])
    if not results:
        print("  (no results)")
        return
    print(f"  {len(results)} result(s):")
    for r in results:
        hits = r.get("hit_count", 0)
        print(f"    [{hits:>3} hits]  {r['text']}")
    ctx = data.get("relational_context", "").strip()
    if ctx:
        print("\n  Relational context:")
        for line in ctx.splitlines():
            print(f"    {line}")


def search(query: str) -> None:
    data = _post("/memory/search", {"query": query, "top_k": 5})
    results = data.get("results", [])
    if not results:
        print("  (no results)")
        return
    print(f"  {len(results)} result(s):")
    for r in results:
        status = r.get("temporal_status", "?")
        hits = r.get("hit_count", 0)
        print(f"    [{status:>9}] [{hits:>3} hits]  {r['text']}")
    ctx = data.get("relational_context", "").strip()
    if ctx:
        print("\n  Relational context:")
        for line in ctx.splitlines():
            print(f"    {line}")
    groups = data.get("entity_groups", {})
    if groups:
        print("\n  Entity groups:")
        for ent, grps in groups.items():
            print(f"    {ent}: {', '.join(grps)}")
    temporal = data.get("temporal_context", [])
    if temporal:
        print("\n  Temporal context:")
        for entry in temporal:
            print(f"    ▶ {entry['current_fact']}")
            for pred in entry.get("preceded_by", []):
                print(f"      └─ {pred['fact']}")
                for conc in pred.get("concurrent_with", []):
                    print(f"           ∥  {conc}")


def chain(query_or_id: str) -> None:
    body: dict = {"max_depth": 10}
    if UUID_RE.match(query_or_id):
        body["fact_id"] = query_or_id
    else:
        body["query"] = query_or_id
    data = _post("/memory/temporal/chain", body)
    root = data.get("root_fact", {})
    period = f" [{root['valid_period']}]" if root.get("valid_period") else ""
    print(f"  Root [{root.get('temporal_status', '?')}]{period}: {root.get('text', '?')}")
    entries = data.get("chain", [])
    if not entries:
        print("  (no predecessor history)")
        return
    for entry in entries:
        indent = "  " + "  " * entry["hop"]
        period = f" [{entry['valid_period']}]" if entry.get("valid_period") else ""
        print(f"{indent}└─ hop {entry['hop']}{period}: {entry['text']}")
        for conc in entry.get("concurrent_with", []):
            period_c = f" [{conc['valid_period']}]" if conc.get("valid_period") else ""
            print(f"{indent}     ∥  {conc['text']}{period_c}")


def add(text: str) -> None:
    data = _post("/memory/add", {"text": text})
    task = _wait_for_task(data["task_id"], "Adding")
    result = task.get("result") or {}
    if task.get("status") == "failed":
        print(f"  Error: {task.get('error')}")
        return
    print(f"  Facts added:   {result.get('facts_added', '?')}")
    print(f"  Triples added: {result.get('triples_added', '?')}")


def learn(filepath: str) -> None:
    path = Path(filepath).expanduser()
    if not path.exists():
        print(f"  File not found: {filepath}")
        return
    text = path.read_text(encoding="utf-8")
    data = _post("/memory/learn", {"text": text}, timeout=300)
    task = _wait_for_task(data["task_id"], f"Learning '{path.name}'")
    result = task.get("result") or {}
    if task.get("status") == "failed":
        print(f"  Error: {task.get('error')}")
        return
    print(f"  Chunks:  {result.get('chunks_succeeded', '?')}/{result.get('chunks_total', '?')}")
    print(f"  Facts:   {result.get('facts_added', '?')}")
    print(f"  Triples: {result.get('triples_added', '?')}")
    errors = result.get("errors") or []
    if errors:
        print(f"  Errors ({len(errors)}):")
        for e in errors[:5]:
            print(f"    chunk {e.get('chunk_index')}: {e.get('error')}")


def consolidate() -> None:
    data = _post("/memory/consolidate", timeout=600)
    task = _wait_for_task(data["task_id"], "Consolidating")
    result = task.get("result") or {}
    if task.get("status") == "failed":
        print(f"  Error: {task.get('error')}")
        return
    report = result.get("report", {})
    print(f"  pruned:              {report.get('pruned', 0)}")
    print(f"  merged:              {report.get('merged', 0)}")
    print(f"  split:               {report.get('split', 0)}")
    print(f"  superseded:          {report.get('superseded', 0)}")
    print(f"  reclassified:        {report.get('reclassified', 0)}")
    print(f"  predicates_normed:   {report.get('predicates_normalized', 0)}")
    resolved = report.get("resolved_entities", [])
    if resolved:
        print(f"  resolved: {len(resolved)} compound entity/entities")
        for r in resolved:
            print(f"    {r['compound']}  →[{r['predicate']}]→  {r['contained']}")
    merged_ents = report.get("merged_entities", [])
    if merged_ents:
        print(f"  merged entities: {len(merged_ents)}")
        for m in merged_ents:
            print(f"    '{m['eliminated']}' → '{m['canonical']}'")
    cleaned = report.get("cleaned_nodes", [])
    if cleaned:
        print(f"  cleaned nodes: {len(cleaned)}")
        for c in cleaned:
            print(f"    [{c['type']}] '{c['name']}'")
    flagged = report.get("flagged", [])
    if flagged:
        print(f"  flagged:  {len(flagged)} item(s) for review")


def visualize(flags: list[str]) -> None:
    import subprocess
    scripts_dir = Path(__file__).parent.parent / "scripts"
    cmd = [sys.executable, str(scripts_dir / "visualize_graph.py"), "--open"]
    if "--temporal" in flags or "-t" in flags:
        cmd.append("--temporal")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"  Visualization failed: {e}")
    except FileNotFoundError:
        print(f"  Script not found: {scripts_dir / 'visualize_graph.py'}")


def all_records(record_type: str | None = None) -> None:
    params = {}
    if record_type:
        params["type"] = record_type
    records = _get("/memory/all", params=params).get("results", [])
    if not records:
        print("  (empty)")
        return
    shown = records[:50]
    for r in shown:
        rtype = r.get("record_type", "?")
        text = r.get("text") or r.get("canonical_name") or "?"
        hits = r.get("hit_count", 0)
        ts = r.get("temporal_status", "")
        status_str = f"[{ts}] " if ts else ""
        groups = r.get("groups", [])
        group_str = f" ({', '.join(groups)})" if groups else ""
        print(f"  [{rtype:>6}] {status_str}[{hits:>3} hits]{group_str}  {text[:100]}")
    if len(records) > 50:
        print(f"  ... and {len(records) - 50} more.")


def clear_memory() -> None:
    answer = input("  Wipe all memory? Type 'yes' to confirm: ").strip()
    if answer.lower() != "yes":
        print("  Aborted.")
        return
    _delete("/memory/clear")
    print("  Memory cleared.")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    global BASE_URL

    parser = argparse.ArgumentParser(description="Erebus interactive CLI")
    parser.add_argument("--url", default=BASE_URL, help="Erebus server base URL")
    args = parser.parse_args()
    BASE_URL = args.url.rstrip("/")

    # Connection check
    try:
        _get("/memory/all", params={"type": "fact"})
    except requests.ConnectionError:
        print(f"Cannot connect to Erebus at {BASE_URL}. Is the server running?")
        sys.exit(1)
    except Exception:
        pass  # server up even if query returns an error

    print(f"Connected to Erebus at {BASE_URL}")
    print("Type 'help' for available commands.\n")

    while True:
        try:
            line = input("erebus> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        try:
            parts = shlex.split(line)
        except ValueError as e:
            print(f"  Parse error: {e}")
            continue

        cmd, rest = parts[0].lower(), parts[1:]

        try:
            match cmd:
                case "context":
                    if not rest:
                        print("  Usage: context <query>")
                    else:
                        context(" ".join(rest))

                case "search":
                    if not rest:
                        print("  Usage: search <query>")
                    else:
                        search(" ".join(rest))

                case "chain":
                    if not rest:
                        print("  Usage: chain <query|fact_id>")
                    else:
                        chain(" ".join(rest))

                case "add":
                    if not rest:
                        print("  Usage: add <text>")
                    else:
                        add(" ".join(rest))

                case "learn":
                    if not rest:
                        print("  Usage: learn <file>")
                    else:
                        learn(rest[0])

                case "consolidate":
                    consolidate()

                case "visualize":
                    visualize(rest)

                case "all":
                    all_records(rest[0] if rest else None)

                case "clear":
                    clear_memory()

                case "help" | "?":
                    print(HELP_TEXT)

                case "exit" | "quit" | "q":
                    break

                case _:
                    print(f"  Unknown command '{cmd}'. Type 'help' for commands.")

        except requests.ConnectionError:
            print(f"  Connection lost — is the server still at {BASE_URL}?")
        except requests.HTTPError as e:
            print(f"  HTTP {e.response.status_code}: {e.response.text[:200]}")
        except Exception as e:
            print(f"  Error: {e}")

    print("Bye.")


if __name__ == "__main__":
    main()
