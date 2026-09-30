"""Ephemeral per-session store (step 4.1).

* In memory only: nothing is written to disk and nothing survives the process.
* Keyed by session id; lookup is by exact id only. There is no listing, search or
  cross-session query API, so one session can never read another's state.
* `end(session_id)` destroys the session and everything it holds (ledger, prior output).
* Asking for an unknown id is an error; a session is never created implicitly by `get`.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

from src.session.ledger import ClaimLedger


class SessionNotFound(KeyError):
    pass


@dataclass
class Session:
    session_id: str
    ledger: ClaimLedger = field(default_factory=ClaimLedger)
    created_at: float = field(default_factory=time.time)
    turns: int = 0
    last_output: str | None = None  # what the user last saw (answer or its presentation variant)

    @property
    def prior_output(self) -> str | None:
        return self.last_output


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self, session_id: str | None = None) -> Session:
        sid = session_id or uuid.uuid4().hex
        with self._lock:
            if sid in self._sessions:
                raise ValueError(f"session {sid!r} already exists")
            session = self._sessions[sid] = Session(sid)
        return session

    def get(self, session_id: str) -> Session:
        with self._lock:
            try:
                return self._sessions[session_id]
            except KeyError:
                raise SessionNotFound(session_id) from None

    def get_or_create(self, session_id: str) -> Session:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = Session(session_id)
            return self._sessions[session_id]

    def end(self, session_id: str) -> None:
        """Destroy the session and all of its state."""
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is not None:
            session.ledger.clear()
            session.last_output = None

    def __contains__(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._sessions

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)
