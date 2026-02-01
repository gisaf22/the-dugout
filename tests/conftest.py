"""Pytest fixtures/config for FADE tests."""

import os
import sys


def pytest_sessionstart(session) -> None:  # type: ignore[unused-argument]
    repo_root = os.path.dirname(os.path.dirname(__file__))
    src_path = os.path.join(repo_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
