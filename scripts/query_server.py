#!/usr/bin/env python3
"""
Simplistic CLI to send direct queries to a running Erebus server instance.
Also serves as a simple code example for deep search queries (slow) and context enrichment (fast).

Usage:
    python scripts/query_server.py [--url http://localhost:8000] [--clear] [--consolidate] [--query "..."]

Requires: pip install requests
"""

import sys
import requests


base_url = "http://127.0.0.1:8000"

def context_enrichment(query: str):
    r = requests.post(
            f"{base_url}/memory/context",
            json={"query": query, "top_k": 5},
            timeout=15,
        )
    r.raise_for_status()
    data = r.json()
    return data


def search_memory(query: str):
    r = requests.post(
            f"{base_url}/memory/search",
            json={"query": query, "top_k": 5},
            timeout=15,
        )
    r.raise_for_status()
    data = r.json()
    return data


def add_memory(text: str):
    r = requests.post(
        f"{base_url}/memory/add",
        json={"text": text},
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()
    print(data)


def main():
    print("Input your Query. To switch modes:")
    print("/context -> switches to context mode (default, fast) ")
    print("/search -> switches to deep search mode (slow)")
    print("/add [INPUT] -> digest and add the string in [INPUT] to memory")
    print("/exit -> exit program\n") 

    search_mode = False
    add_mode = False

    while True:
        user_input = input("Query: ").strip()

        if user_input.startswith("/"):
            match user_input.strip("/"):
                case "context": 
                    print("Context mode activated.")
                    search_mode = False
                    add_mode = False
                    continue
                case "search":
                    print("Search mode activated.")
                    search_mode = True
                    add_mode = False
                    continue
                case "add":
                    print("Add mode activated.")
                    add_mode = True
                    continue
                case "exit":
                    print("Closing program. Goodbye.")
                    sys.exit()
                case _: 
                    print("Invalid command.")
                    continue

        if add_mode:
            response = add_memory(user_input)
        elif not search_mode:
            response = context_enrichment(user_input)
        else:
            response = search_memory(user_input)

        print(response)


if __name__ == "__main__":
    main()

