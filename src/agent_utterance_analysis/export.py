from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable


def export_rows(rows: Iterable[object], output: str | Path, fmt: str) -> int:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = [dict(row) for row in rows]
    if fmt == "jsonl":
        with path.open("w", encoding="utf-8") as handle:
            for row in materialized:
                row["metadata"] = json.loads(row.pop("metadata_json"))
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    elif fmt == "csv":
        with path.open("w", encoding="utf-8", newline="") as handle:
            fields = [
                "id",
                "source_agent",
                "timestamp",
                "conversation_id",
                "turn_index",
                "source_path",
                "text",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(materialized)
    elif fmt == "markdown":
        with path.open("w", encoding="utf-8") as handle:
            handle.write("# Exported Utterances\n\n")
            for row in materialized:
                timestamp = row["timestamp"] or "unknown time"
                handle.write(f"## {row['source_agent']} | {timestamp}\n\n")
                handle.write(f"- Source: `{row['source_path']}`\n")
                handle.write(f"- Conversation: `{row['conversation_id']}`\n")
                handle.write(f"- Turn: `{row['turn_index']}`\n\n")
                handle.write(row["text"].strip() + "\n\n")
    else:
        raise ValueError(f"Unsupported export format: {fmt}")
    return len(materialized)

