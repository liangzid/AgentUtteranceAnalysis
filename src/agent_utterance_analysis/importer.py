from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .models import SourceFile
from .parsers import parse_file
from .storage import Store

SUPPORTED_SUFFIXES = {".json", ".jsonl", ".ndjson", ".md", ".markdown", ".txt", ".log", ".db", ".sqlite"}
EXCLUDED_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "__pycache__",
    "node_modules",
    "snapshot",
    "bin",
    "dist",
    "build",
}
AGENT_HINTS = {
    "openclaw": "openclaw",
    "claude": "claude-code",
    "claude-code": "claude-code",
    "opencode": "opencode",
    "codex": "codex",
    "kilo": "kilo-code",
    "kilo-code": "kilo-code",
}


@dataclass(frozen=True)
class ImportSummary:
    scanned_files: int = 0
    imported_files: int = 0
    skipped_current_files: int = 0
    skipped_unsupported_files: int = 0
    failed_files: int = 0
    utterances: int = 0

    def add(self, **changes: int) -> "ImportSummary":
        values = self.__dict__.copy()
        for key, value in changes.items():
            values[key] += value
        return ImportSummary(**values)


def import_paths(paths: Iterable[str | Path], store: Store, force: bool = False) -> ImportSummary:
    summary = ImportSummary()
    for path in iter_supported_files(paths):
        summary = summary.add(scanned_files=1)
        source = inspect_source(path)
        if not force and store.source_is_current(source):
            summary = summary.add(skipped_current_files=1)
            continue
        try:
            utterances = parse_file(path, source.agent)
            count = store.replace_source(source, utterances)
            summary = summary.add(imported_files=1, utterances=count)
        except Exception:
            summary = summary.add(failed_files=1)
    return summary


def iter_supported_files(paths: Iterable[str | Path]) -> Iterable[Path]:
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if path.is_file() and is_supported_file(path):
            yield path
        elif path.is_dir():
            for root, dirnames, filenames in os.walk(path):
                dirnames[:] = [dirname for dirname in dirnames if dirname not in EXCLUDED_DIR_NAMES]
                for filename in sorted(filenames):
                    child = Path(root) / filename
                    if child.is_file() and is_supported_file(child):
                        yield child


def inspect_source(path: Path) -> SourceFile:
    stat = path.stat()
    return SourceFile(
        path=path,
        agent=detect_agent(path),
        sha256=file_sha256(path),
        mtime_ns=stat.st_mtime_ns,
        size=stat.st_size,
    )


def detect_agent(path: Path) -> str:
    lowered = str(path).lower().replace("_", "-")
    for hint, agent in AGENT_HINTS.items():
        if hint in lowered:
            return agent
    return "unknown"


def is_supported_file(path: Path) -> bool:
    if path.name == "opencode.db":
        return True
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        return False
    if path.suffix.lower() in {".db", ".sqlite"}:
        return False
    return True


def is_ignored(path: Path) -> bool:
    return any(part in EXCLUDED_DIR_NAMES for part in path.parts)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
