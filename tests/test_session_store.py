"""Session store (step 4.1): ephemeral, keyed, isolated."""

import ast
from pathlib import Path

import pytest

from src.session.store import SessionNotFound, SessionStore

ROOT = Path(__file__).resolve().parents[1]


def test_two_sessions_are_isolated():
    store = SessionStore()
    a, b = store.create("A"), store.create("B")
    a.last_output = "answer for A"
    a.ledger.claims["x"] = "claim of A"
    a.turns = 3
    assert b.last_output is None and b.ledger.claims == {} and b.turns == 0
    assert a.ledger is not b.ledger
    assert store.get("A").last_output == "answer for A" and store.get("B").last_output is None


def test_end_destroys_state():
    store = SessionStore()
    s = store.create("A")
    s.ledger.claims["x"] = "c"
    store.end("A")
    assert "A" not in store and len(store) == 0 and s.ledger.claims == {}
    with pytest.raises(SessionNotFound):
        store.get("A")
    fresh = store.create("A")  # a new session with the same id starts empty
    assert fresh.ledger.claims == {} and fresh is not s


def test_unknown_session_is_an_error_and_not_created():
    store = SessionStore()
    with pytest.raises(SessionNotFound):
        store.get("nope")
    assert len(store) == 0


def test_duplicate_create_rejected_and_get_or_create_is_idempotent():
    store = SessionStore()
    store.create("A")
    with pytest.raises(ValueError):
        store.create("A")
    assert store.get_or_create("B") is store.get_or_create("B")


def test_separate_stores_share_nothing():
    s1, s2 = SessionStore(), SessionStore()
    s1.create("A").last_output = "x"
    assert "A" not in s2


def test_store_has_no_disk_or_listing_api():
    tree = ast.parse((ROOT / "src" / "session" / "store.py").read_text(encoding="utf-8"))
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert not imported & {"pickle", "shelve", "sqlite3", "json", "os", "pathlib"}
    public = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")}
    assert not public & {"list", "all", "search", "find", "sessions", "save", "load"}
