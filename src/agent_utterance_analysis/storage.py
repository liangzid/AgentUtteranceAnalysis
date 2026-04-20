from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from .models import SourceFile, Utterance, utc_now_iso


SCHEMA = """
CREATE TABLE IF NOT EXISTS source_files (
    path TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    mtime_ns INTEGER NOT NULL,
    size INTEGER NOT NULL,
    imported_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS utterances (
    id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    source_agent TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    timestamp TEXT,
    text TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    imported_at TEXT NOT NULL,
    FOREIGN KEY(source_path) REFERENCES source_files(path)
);

CREATE INDEX IF NOT EXISTS idx_utterances_agent ON utterances(source_agent);
CREATE INDEX IF NOT EXISTS idx_utterances_timestamp ON utterances(timestamp);
CREATE INDEX IF NOT EXISTS idx_utterances_source ON utterances(source_path);
"""


class Store:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def source_is_current(self, source: SourceFile) -> bool:
        row = self.conn.execute(
            "SELECT sha256, mtime_ns, size FROM source_files WHERE path = ?",
            (str(source.path),),
        ).fetchone()
        if row is None:
            return False
        return (
            row["sha256"] == source.sha256
            and row["mtime_ns"] == source.mtime_ns
            and row["size"] == source.size
        )

    def replace_source(self, source: SourceFile, utterances: Iterable[Utterance]) -> int:
        imported_at = utc_now_iso()
        utterance_list = list(utterances)
        with self.conn:
            self.conn.execute("DELETE FROM utterances WHERE source_path = ?", (str(source.path),))
            self.conn.execute(
                """
                INSERT INTO source_files(path, agent, sha256, mtime_ns, size, imported_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    agent=excluded.agent,
                    sha256=excluded.sha256,
                    mtime_ns=excluded.mtime_ns,
                    size=excluded.size,
                    imported_at=excluded.imported_at
                """,
                (str(source.path), source.agent, source.sha256, source.mtime_ns, source.size, imported_at),
            )
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO utterances(
                    id, source_path, source_agent, conversation_id, turn_index,
                    timestamp, text, metadata_json, imported_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        utterance.stable_id,
                        utterance.source_path,
                        utterance.source_agent,
                        utterance.conversation_id,
                        utterance.turn_index,
                        utterance.timestamp,
                        utterance.text,
                        json.dumps(utterance.metadata or {}, ensure_ascii=False, sort_keys=True),
                        imported_at,
                    )
                    for utterance in utterance_list
                ],
            )
        return len(utterance_list)

    def iter_utterances(self) -> Iterable[sqlite3.Row]:
        yield from self.conn.execute(
            """
            SELECT id, source_path, source_agent, conversation_id, turn_index,
                   timestamp, text, metadata_json, imported_at
            FROM utterances
            ORDER BY COALESCE(timestamp, imported_at), source_path, turn_index
            """
        )

    def counts(self) -> dict[str, int]:
        source_count = self.conn.execute("SELECT COUNT(*) AS c FROM source_files").fetchone()["c"]
        utterance_count = self.conn.execute("SELECT COUNT(*) AS c FROM utterances").fetchone()["c"]
        return {"sources": int(source_count), "utterances": int(utterance_count)}

