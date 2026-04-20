from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .importer import SUPPORTED_SUFFIXES

AGENT_DIR_NAMES = {
    ".codex",
    ".claude",
    ".opencode",
    ".openclaw",
    ".kilo",
    ".kilo-code",
    ".kilocode",
}
GLOBAL_CANDIDATES = (
    ".local/share/opencode/opencode.db",
    ".local/share/opencode/log",
    ".codex",
    ".claude",
    ".config/claude",
    ".config/opencode",
    ".openclaw",
    ".kilo",
    ".kilo-code",
    ".kilocode",
)
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


@dataclass(frozen=True)
class DiscoverySummary:
    roots: tuple[Path, ...]
    files: tuple[Path, ...]


def discover_sources(
    home: str | Path = "~",
    project_globs: Iterable[str] | None = None,
    include_global: bool = True,
    max_project_depth: int = 3,
) -> DiscoverySummary:
    home_path = Path(home).expanduser()
    roots: set[Path] = set()
    files: set[Path] = set()

    if include_global:
        for relative in GLOBAL_CANDIDATES:
            candidate = home_path / relative
            add_candidate(candidate, roots, files)

    for pattern in project_globs or (str(home_path / "code" / "*"),):
        expanded_pattern = str(Path(pattern).expanduser())
        for project_raw in sorted(glob.glob(expanded_pattern)):
            project = Path(project_raw)
            if not project.exists() or not project.is_dir():
                continue
            for agent_dir in iter_agent_dirs(project, max_depth=max_project_depth):
                add_candidate(agent_dir, roots, files)
            for db_path in project.glob("**/opencode.db"):
                if is_ignored(db_path):
                    continue
                files.add(db_path)

    return DiscoverySummary(roots=tuple(sorted(roots)), files=tuple(sorted(files)))


def add_candidate(candidate: Path, roots: set[Path], files: set[Path]) -> None:
    if not candidate.exists():
        return
    if candidate.is_file():
        if candidate.suffix.lower() in SUPPORTED_SUFFIXES or candidate.name == "opencode.db":
            files.add(candidate)
    elif candidate.is_dir():
        roots.add(candidate)


def iter_agent_dirs(project: Path, max_depth: int) -> Iterable[Path]:
    base_depth = len(project.parts)
    for root, dirnames, _ in os.walk(project):
        root_path = Path(root)
        depth = len(root_path.parts) - base_depth
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if dirname not in EXCLUDED_DIR_NAMES and depth < max_depth
        ]
        for dirname in dirnames:
            if dirname in AGENT_DIR_NAMES:
                yield root_path / dirname


def is_ignored(path: Path) -> bool:
    return any(part in EXCLUDED_DIR_NAMES for part in path.parts)
