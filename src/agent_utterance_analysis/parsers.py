from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .models import Utterance

USER_ROLES = {"user", "human", "me", "you", "requester", "client"}
CONVERSATION_KEYS = ("messages", "conversation", "turns", "items", "entries")
TEXT_KEYS = ("content", "text", "message", "body", "prompt")
ROLE_KEYS = ("role", "speaker", "author", "from", "type", "kind")
TIME_KEYS = ("timestamp", "created_at", "createdAt", "time", "date")


def parse_file(path: Path, agent: str) -> list[Utterance]:
    suffix = path.suffix.lower()
    if agent == "opencode" and path.name == "opencode.db":
        return parse_opencode_sqlite(path)
    raw = path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".json":
        return parse_json(raw, path, agent)
    if suffix in {".jsonl", ".ndjson"}:
        return parse_jsonl(raw, path, agent)
    if suffix in {".md", ".markdown"}:
        return parse_labeled_text(raw, path, agent, markdown=True)
    if suffix in {".txt", ".log"}:
        return parse_labeled_text(raw, path, agent, markdown=False)
    return []


def parse_opencode_sqlite(path: Path) -> list[Utterance]:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT
            m.id AS message_id,
            m.session_id,
            m.time_created AS message_time_created,
            m.data AS message_data,
            p.id AS part_id,
            p.time_created AS part_time_created,
            p.data AS part_data,
            s.title AS session_title,
            s.directory AS session_directory,
            pr.worktree AS project_worktree,
            pr.name AS project_name
        FROM message m
        LEFT JOIN part p ON p.message_id = m.id
        LEFT JOIN session s ON s.id = m.session_id
        LEFT JOIN project pr ON pr.id = s.project_id
        ORDER BY m.time_created, p.time_created, p.id
        """
    ).fetchall()
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        message_data = _loads_json(row["message_data"])
        if str(message_data.get("role", "")).lower() not in USER_ROLES:
            continue
        item = grouped.setdefault(
            row["message_id"],
            {
                "session_id": row["session_id"],
                "time_created": row["message_time_created"],
                "parts": [],
                "session_title": row["session_title"],
                "session_directory": row["session_directory"],
                "project_worktree": row["project_worktree"],
                "project_name": row["project_name"],
                "message_data": message_data,
            },
        )
        part_data = _loads_json(row["part_data"]) if row["part_data"] else {}
        if part_data.get("type") == "text" and part_data.get("text"):
            item["parts"].append(str(part_data["text"]))

    utterances: list[Utterance] = []
    for index, item in enumerate(grouped.values()):
        text = "\n".join(part.strip() for part in item["parts"] if part.strip()).strip()
        if not text:
            continue
        utterances.append(
            Utterance(
                source_path=str(path),
                source_agent="opencode",
                conversation_id=str(item["session_id"]),
                turn_index=index,
                text=text,
                timestamp=_millis_to_iso(item["time_created"]),
                metadata={
                    "parser": "opencode_sqlite",
                    "session_title": item["session_title"],
                    "session_directory": item["session_directory"],
                    "project_worktree": item["project_worktree"],
                    "project_name": item["project_name"],
                    "agent": item["message_data"].get("agent"),
                    "model": item["message_data"].get("model"),
                },
            )
        )
    conn.close()
    return utterances


def parse_json(raw: str, path: Path, agent: str) -> list[Utterance]:
    data = json.loads(raw)
    return _utterances_from_json_value(data, path, agent)


def parse_jsonl(raw: str, path: Path, agent: str) -> list[Utterance]:
    output: list[Utterance] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        data = json.loads(line)
        utterances = _utterances_from_json_value(data, path, agent, conversation_hint=f"line-{line_number}")
        output.extend(utterances)
    return _renumber(output)


def parse_labeled_text(raw: str, path: Path, agent: str, markdown: bool) -> list[Utterance]:
    label_re = re.compile(
        r"^(?:#{1,6}\s*)?(user|human|me|you)\s*:?\s*$|^(user|human|me|you)\s*:\s*(.*)$",
        re.IGNORECASE,
    )
    other_label_re = re.compile(
        r"^(?:#{1,6}\s*)?(assistant|agent|claude|codex|openclaw|opencode|kilo(?: code)?)\s*:?\s*$|"
        r"^(assistant|agent|claude|codex|openclaw|opencode|kilo(?: code)?)\s*:",
        re.IGNORECASE,
    )
    utterances: list[Utterance] = []
    current: list[str] = []
    capture = False
    turn = 0

    def flush() -> None:
        nonlocal current, turn
        text = "\n".join(current).strip()
        if text:
            utterances.append(
                Utterance(
                    source_path=str(path),
                    source_agent=agent,
                    conversation_id=path.stem,
                    turn_index=turn,
                    text=text,
                    metadata={"parser": "markdown" if markdown else "text"},
                )
            )
            turn += 1
        current = []

    for line in raw.splitlines():
        match = label_re.match(line.strip())
        if match:
            flush()
            capture = True
            inline = match.group(3)
            if inline:
                current.append(inline)
            continue
        if other_label_re.match(line.strip()):
            flush()
            capture = False
            continue
        if capture:
            current.append(line)
    flush()
    return utterances


def _utterances_from_json_value(
    data: Any, path: Path, agent: str, conversation_hint: str | None = None
) -> list[Utterance]:
    utterances: list[Utterance] = []
    for conversation_id, messages in _find_message_lists(data, conversation_hint or path.stem):
        for index, message in enumerate(messages):
            role = _extract_role(message)
            if role not in USER_ROLES:
                continue
            text = _extract_text(message)
            if not text:
                continue
            utterances.append(
                Utterance(
                    source_path=str(path),
                    source_agent=agent,
                    conversation_id=conversation_id,
                    turn_index=index,
                    text=text,
                    timestamp=_extract_timestamp(message),
                    metadata={"parser": "json", "role": role},
                )
            )
    if not utterances and isinstance(data, dict) and _extract_role(data) in USER_ROLES:
        text = _extract_text(data)
        if text:
            utterances.append(
                Utterance(
                    source_path=str(path),
                    source_agent=agent,
                    conversation_id=conversation_hint or path.stem,
                    turn_index=0,
                    text=text,
                    timestamp=_extract_timestamp(data),
                    metadata={"parser": "json", "role": _extract_role(data)},
                )
            )
    return _renumber(utterances)


def _find_message_lists(data: Any, fallback_id: str) -> Iterable[tuple[str, list[Any]]]:
    if isinstance(data, list):
        if data and all(isinstance(item, dict) for item in data):
            yield fallback_id, data
        return
    if not isinstance(data, dict):
        return

    conversation_id = str(
        data.get("id")
        or data.get("conversation_id")
        or data.get("conversationId")
        or data.get("session_id")
        or fallback_id
    )
    for key in CONVERSATION_KEYS:
        value = data.get(key)
        if isinstance(value, list):
            yield conversation_id, value
    for key, value in data.items():
        if isinstance(value, dict):
            yield from _find_message_lists(value, f"{conversation_id}:{key}")
        elif isinstance(value, list) and key not in CONVERSATION_KEYS:
            for idx, item in enumerate(value):
                if isinstance(item, dict):
                    yield from _find_message_lists(item, f"{conversation_id}:{key}-{idx}")


def _extract_role(message: Any) -> str | None:
    if not isinstance(message, dict):
        return None
    for key in ROLE_KEYS:
        value = message.get(key)
        if isinstance(value, dict):
            value = value.get("role") or value.get("name") or value.get("type")
        if isinstance(value, str):
            role = value.strip().lower()
            if role in {"user_message", "human_message"}:
                return "user"
            return role
    return None


def _extract_text(message: Any) -> str:
    if isinstance(message, str):
        return message.strip()
    if not isinstance(message, dict):
        return ""
    for key in TEXT_KEYS:
        value = message.get(key)
        text = _coerce_text(value)
        if text:
            return text
    return ""


def _coerce_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = _coerce_text(item.get("text") or item.get("content"))
                if text:
                    parts.append(text)
        return "\n".join(parts).strip()
    if isinstance(value, dict):
        return _coerce_text(value.get("text") or value.get("content") or value.get("value"))
    return ""


def _extract_timestamp(message: Any) -> str | None:
    if not isinstance(message, dict):
        return None
    for key in TIME_KEYS:
        value = message.get(key)
        if value is not None:
            return str(value)
    return None


def _loads_json(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _millis_to_iso(value: Any) -> str | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if number > 10_000_000_000:
        number = number / 1000
    return datetime.fromtimestamp(number, tz=timezone.utc).replace(microsecond=0).isoformat()


def _renumber(utterances: list[Utterance]) -> list[Utterance]:
    return [
        Utterance(
            source_path=item.source_path,
            source_agent=item.source_agent,
            conversation_id=item.conversation_id,
            turn_index=index,
            text=item.text,
            timestamp=item.timestamp,
            metadata=item.metadata,
        )
        for index, item in enumerate(utterances)
    ]
