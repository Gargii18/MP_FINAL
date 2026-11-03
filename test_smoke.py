"""Smoke tests: every page renders without raising, using the built-in demo data.

Run:  pytest -q        (or)   python tests/test_smoke.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402


def _run_all_pages():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    pages = list(at.sidebar.radio[0].options)
    for p in pages:
        at.sidebar.radio[0].set_value(p).run()
        assert not at.exception, f"Page '{p}' raised: {at.exception}"
    return at


def test_all_pages_render():
    _run_all_pages()


if __name__ == "__main__":
    _run_all_pages()
    print("All pages rendered OK")
