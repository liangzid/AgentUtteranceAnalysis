from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceFile:
    path: Path
    agent: str
    sha256: str
    mtime_ns: int
    size: int


@dataclass(frozen=True)
class Utterance:
    source_path: str
    source_agent: str
    conversation_id: str
    turn_index: int
    text: str
    timestamp: str | None = None
    metadata: dict[str, Any] | None = None

    @property
    def stable_id(self) -> str:
        import hashlib

        payload = "\n".join(
            [
                self.source_path,
                self.source_agent,
                self.conversation_id,
                str(self.turn_index),
                self.timestamp or "",
                self.text,
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

