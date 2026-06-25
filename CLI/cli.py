#!/usr/bin/env python3
"""
Simplistic CLI to send direct queries to a running Erebus server instance.
Also serves as a simple code example for deep search queries (slow) and context enrichment (fast).

Usage:
    python cli.py [--url http://localhost:8000]

Requires: pip install requests
"""

import requests


def context(query: str) -> dict:
    """
    Takes a query and runs it through the /memory/context endpoint of 
    a running Erebus instance.
    """
    pass


def search(query: str) -> dict:
    pass


def learn(filename: str):
    """
    Reads a given plain text file and feeds text towards the /memory/learn 
    endpoint.
    """
    pass


def add(text: str):
    pass


def consolidate():
    pass


def summarize():
    # TODO
    pass


def main():
    """
    Attempts to connect to running Erebus server and runs a user input loop.

    User: [command] [target]

    i.e. 
    User: context "Who is Alice Mercer?"

    or 
    User: learn Biography/Alice.md
    """
    pass


if __name__ == "__main__":
    main()

