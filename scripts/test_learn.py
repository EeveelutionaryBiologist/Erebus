#!/usr/bin/env python3
"""
Live integration harness for the Erebus memory server.

Usage:
    python scripts/test_learn.py [--url http://localhost:8000] [--clear] [--consolidate] [--query "..."]

Requires: pip install requests
"""


import argparse
import sys
import time

import requests

# ---------------------------------------------------------------------------
# Test story — exercises chunking, context-hint propagation, entity grouping,
# temporal tagging, and supersession keywords across multiple chunk boundaries.
# ---------------------------------------------------------------------------
TEST_STORY = """
Alice Mercer grew up in Portland, Oregon, where she lived with her parents, James and
Ruth Mercer, and her younger brother, Tom. During her childhood she was close to her
grandmother, Eleanor Mercer, who used to tell her stories about the family's history in
Ireland. Eleanor passed away in 2005, and Alice has kept a journal of those stories ever since.

Alice attended Reed College from 2008 to 2012, where she studied biochemistry. During college
she worked part-time at a local café called Brew & Bloom, which she describes as formative for
her love of community spaces. She used to compete in amateur cycling races during that period,
though she no longer races competitively; she still cycles recreationally on the weekends.

After graduating, Alice moved to San Francisco and joined a biotech startup called NovaStem
from 2013 to 2017. She was a research associate there and co-authored two papers on stem cell
differentiation. She formed a close working friendship with her colleague Dr. Priya Nair,
who was her lab partner for most of that period. Alice used to live in the Mission District
during the NovaStem years, but she has since moved to the Sunset District.

In 2018 Alice left NovaStem and co-founded her own company, CellBridge Therapeutics, where
she is currently the Chief Science Officer. CellBridge is headquartered in South San Francisco.
Her co-founder is Marcus Webb, a former venture capitalist who formerly worked at Andreessen
Horowitz before transitioning into biotech entrepreneurship.

Alice's partner is Jordan Kim, a high school art teacher in San Francisco. They have been
together since 2016 and have a dog named Miso. Alice and Jordan are planning to get married
in 2027. Outside of work, Alice volunteers at the Pacific Science Center on weekends and
is an avid amateur photographer. She used to paint watercolors regularly but has not done
so since 2020 when her schedule became too demanding.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def poll_task(base_url: str, task_id: str, label: str = "task") -> dict:
    print(f"  Polling {label} [{task_id}] ", end="", flush=True)
    while True:
        r = requests.get(f"{base_url}/memory/task/{task_id}", timeout=10)
        r.raise_for_status()
        data = r.json()
        status = data["status"]
        if status in ("completed", "failed"):
            print(f" {status}")
            return data
        print(".", end="", flush=True)
        time.sleep(2)


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step_clear(base_url: str) -> None:
    print_section("CLEAR")
    r = requests.delete(f"{base_url}/memory/clear", timeout=10)
    r.raise_for_status()
    print(f"  {r.json().get('message', 'done')}")


def step_learn(base_url: str) -> None:
    print_section("LEARN")
    r = requests.post(f"{base_url}/memory/learn", json={"text": TEST_STORY}, timeout=10)
    r.raise_for_status()
    task_id = r.json()["task_id"]

    result_data = poll_task(base_url, task_id, "learn")
    result = result_data.get("result") or {}
    error = result_data.get("error")

    if result_data["status"] == "failed":
        print(f"  FAILED: {error}")
        sys.exit(1)

    print(f"  chunks_total      : {result.get('chunks_total', '?')}")
    print(f"  chunks_succeeded  : {result.get('chunks_succeeded', '?')}")
    print(f"  facts_added       : {result.get('facts_added', '?')}")
    print(f"  triples_added     : {result.get('triples_added', '?')}")

    errors = result.get("errors", [])
    if errors:
        print(f"  chunk errors ({len(errors)}):")
        for e in errors:
            print(f"    chunk {e['chunk_index']}: {e['error']}")
            print(f"      text: {e['text'][:80]}...")


def step_entities(base_url: str) -> None:
    print_section("ENTITIES")
    r = requests.get(f"{base_url}/memory/all", params={"type": "entity"}, timeout=10)
    r.raise_for_status()
    entities = r.json().get("results", [])
    print(f"  {len(entities)} entities found\n")

    col_w = [30, 6, 6, 35]
    header = f"  {'Name':<{col_w[0]}} {'Hits':>{col_w[1]}} {'Chunks':>{col_w[2]}}  {'Groups':<{col_w[3]}}"
    print(header)
    print("  " + "-" * (sum(col_w) + 4))
    for e in sorted(entities, key=lambda x: x["text"]):
        groups = ", ".join(e.get("groups", [])) or "(none)"
        print(f"  {e['text']:<{col_w[0]}} {e['hit_count']:>{col_w[1]}} {e['chunk_count']:>{col_w[2]}}  {groups:<{col_w[3]}}")


def step_facts(base_url: str) -> None:
    print_section("FACTS")
    r = requests.get(f"{base_url}/memory/all", params={"type": "fact"}, timeout=10)
    r.raise_for_status()
    facts = r.json().get("results", [])
    print(f"  {len(facts)} facts found\n")

    by_status: dict[str, list] = {}
    for f in facts:
        s = f.get("temporal_status", "current")
        by_status.setdefault(s, []).append(f)

    for status in ("current", "historical", "uncertain"):
        group = by_status.get(status, [])
        if not group:
            continue
        print(f"  [{status.upper()}] ({len(group)} facts)")
        for f in group:
            period = f.get("valid_period")
            period_str = f" [{period}]" if period else ""
            print(f"    - {f['text']}{period_str}")
        print()


def step_search(base_url: str, queries: list[str]) -> None:
    for query in queries:
        print_section(f"SEARCH: {query}")
        r = requests.post(
            f"{base_url}/memory/search",
            json={"query": query, "top_k": 5},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()

        results = data.get("results", [])
        print(f"  Vector results ({len(results)}):")
        for item in results:
            ts = item.get("temporal_status", "")
            ts_str = f" [{ts}]" if ts else ""
            print(f"    - {item['text']}{ts_str}")

        rel = data.get("relational_context", "")
        if rel:
            print(f"\n  Relational context:\n    {rel.strip()}")

        temporal = data.get("temporal_context", [])
        if temporal:
            print(f"\n  Temporal context ({len(temporal)} chain(s)):")
            for chain in temporal:
                print(f"    current : {chain['current_fact']}")
                for pred in chain.get("preceded_by", []):
                    print(f"    past    : {pred['fact']}")
                    for conc in pred.get("concurrent_with", []):
                        print(f"    conc    : {conc}")

        groups = data.get("entity_groups", {})
        if groups:
            print(f"\n  Entity groups:")
            for ent, grps in groups.items():
                print(f"    {ent}: {', '.join(grps)}")


def step_consolidate(base_url: str) -> None:
    print_section("CONSOLIDATE")
    r = requests.post(f"{base_url}/memory/consolidate", timeout=10)
    r.raise_for_status()
    task_id = r.json()["task_id"]

    result_data = poll_task(base_url, task_id, "consolidate")
    result = result_data.get("result") or {}
    report = result.get("report", {})

    if result_data["status"] == "failed":
        print(f"  FAILED: {result_data.get('error')}")
        return

    print(f"  pruned    : {report.get('pruned', 0)}")
    print(f"  merged    : {report.get('merged', 0)}")
    print(f"  split     : {report.get('split', 0)}")
    print(f"  superseded: {report.get('superseded', 0)}")

    flagged = report.get("flagged", [])
    if flagged:
        print(f"\n  Flagged contradictions ({len(flagged)}):")
        for f in flagged:
            src = f.get("source", "structural")
            if src == "text_based":
                print(f"    [text] {f.get('fact_a')} <-> {f.get('fact_b')}")
            else:
                print(f"    [struct] {f.get('subject')} — {f.get('predicate_a')} vs {f.get('predicate_b')} — {f.get('object')}")

    errors = report.get("errors", [])
    if errors:
        print(f"\n  Errors: {errors}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Erebus live API integration test")
    parser.add_argument("--url", default="http://localhost:8000", help="Server base URL")
    parser.add_argument("--clear", action="store_true", help="Wipe all data before running")
    parser.add_argument("--consolidate", action="store_true", help="Run consolidation after learn")
    parser.add_argument("--query", action="append", default=[], metavar="Q", help="Search query (repeatable)")
    args = parser.parse_args()

    base = args.url.rstrip("/")

    print(f"Target: {base}")

    try:
        requests.get(f"{base}/memory/all", params={"type": "raw"}, timeout=5)
    except requests.exceptions.ConnectionError:
        print(f"\nERROR: Cannot reach {base} — is the server running?")
        print("  Start it with:  uvicorn memory_server:app --reload")
        sys.exit(1)

    if args.clear:
        step_clear(base)

    step_learn(base)
    step_entities(base)
    step_facts(base)

    queries = args.query or [
        "Where does Alice work?",
        "Who are Alice's family members?",
        "What did Alice used to do that she no longer does?",
    ]
    step_search(base, queries)

    if args.consolidate:
        step_consolidate(base)
        # Re-dump facts after consolidation to see temporal_status changes
        step_facts(base)

    print_section("DONE")


if __name__ == "__main__":
    main()
