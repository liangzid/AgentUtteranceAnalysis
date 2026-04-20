from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Iterable


QUESTION_WORDS = {"what", "why", "how", "when", "where", "who", "which", "can", "could", "would", "should", "do", "does", "is", "are"}
REQUEST_MARKERS = {"please", "want", "need", "create", "write", "fix", "implement", "analyze", "explain", "review", "help"}
POLITENESS_MARKERS = {"please", "thanks", "thank", "could", "would"}

STYLE_WARNINGS = [
    (re.compile(r"\b(batchly)\b", re.IGNORECASE), "Use 'in batches' or 'batch' instead of 'batchly'."),
    (
        re.compile(r"\b(?:utternaces|utternace|utternances|utternance|uterances|utterences)\b", re.IGNORECASE),
        "Check spelling: 'utterances'.",
    ),
    (re.compile(r"\bdisplay\b", re.IGNORECASE), "When discussing language, 'expression' or 'wording' is usually more natural than 'display'."),
    (re.compile(r"\bcorrect or not\b", re.IGNORECASE), "More natural: 'whether ... is natural and correct'."),
    (re.compile(r"\band so on\b", re.IGNORECASE), "'and so on' is understandable, but 'and similar tools' is often more precise."),
    (re.compile(r"\bI want to know whether\b", re.IGNORECASE), "Natural, but often shorter as 'I want to know if'."),
]


def analyze_rows(rows: Iterable[object], output: str | Path | None = None) -> str:
    materialized = [dict(row) for row in rows]
    report = build_report(materialized)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")
    return report


def build_report(rows: list[dict[str, object]]) -> str:
    texts = [str(row["text"]) for row in rows]
    words_per_utterance = [word_count(text) for text in texts]
    chars_per_utterance = [len(text) for text in texts]
    agent_counts = Counter(str(row["source_agent"]) for row in rows)
    month_counts = Counter(month_bucket(row.get("timestamp")) for row in rows)
    property_counts = property_distribution(texts)
    warnings = collect_style_warnings(rows)

    lines = ["# Agent Utterance Analysis", ""]
    lines.extend(summary_section(rows, words_per_utterance, chars_per_utterance))
    lines.extend(counter_section("Distribution by Agent", agent_counts))
    lines.extend(counter_section("Distribution by Month", month_counts))
    lines.extend(counter_section("Utterance Properties", property_counts))
    lines.extend(style_section(warnings))
    return "\n".join(lines).rstrip() + "\n"


def summary_section(rows: list[dict[str, object]], words: list[int], chars: list[int]) -> list[str]:
    if not rows:
        return ["## Summary", "", "No utterances found.", ""]
    return [
        "## Summary",
        "",
        f"- Utterances: {len(rows)}",
        f"- Average words: {mean(words):.1f}",
        f"- Median words: {median(words):.1f}",
        f"- Longest utterance: {max(words)} words",
        f"- Average characters: {mean(chars):.1f}",
        "",
    ]


def counter_section(title: str, counter: Counter[str]) -> list[str]:
    lines = [f"## {title}", ""]
    if not counter:
        return lines + ["No data.", ""]
    total = sum(counter.values())
    for key, count in counter.most_common():
        percent = 100 * count / total if total else 0
        lines.append(f"- {key}: {count} ({percent:.1f}%)")
    lines.append("")
    return lines


def style_section(warnings: list[tuple[str, str, str]]) -> list[str]:
    lines = ["## English Naturalness and Correctness", ""]
    if not warnings:
        return lines + ["No repeated heuristic warnings found.", ""]
    warning_counts = Counter(message for _, message, _ in warnings)
    lines.append("### Common Warning Types")
    lines.append("")
    for message, count in warning_counts.most_common():
        lines.append(f"- {message} ({count})")
    lines.extend(["", "### Examples", ""])
    for source, message, excerpt in warnings[:20]:
        lines.append(f"- `{source}`: {message}")
        lines.append(f"  - Example: {excerpt}")
    lines.append("")
    return lines


def property_distribution(texts: list[str]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for text in texts:
        lower_words = {word.lower() for word in re.findall(r"[A-Za-z']+", text)}
        if "?" in text or lower_words.intersection(QUESTION_WORDS):
            counter["question_or_inquiry"] += 1
        if lower_words.intersection(REQUEST_MARKERS):
            counter["request_or_instruction"] += 1
        if lower_words.intersection(POLITENESS_MARKERS):
            counter["politeness_marker"] += 1
        if word_count(text) >= 80:
            counter["long_context_prompt"] += 1
        if re.search(r"\b\d+\.", text):
            counter["numbered_requirements"] += 1
        if "```" in text:
            counter["contains_code_block"] += 1
    return counter


def collect_style_warnings(rows: list[dict[str, object]]) -> list[tuple[str, str, str]]:
    warnings: list[tuple[str, str, str]] = []
    for row in rows:
        text = str(row["text"])
        for pattern, message in STYLE_WARNINGS:
            if pattern.search(text):
                warnings.append((str(row["source_path"]), message, excerpt(text)))
        if sentence_complexity_score(text) > 38:
            warnings.append(
                (
                    str(row["source_path"]),
                    "This sentence may be hard to read; consider splitting it into shorter requests.",
                    excerpt(text),
                )
            )
    return warnings


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w']+\b", text))


def month_bucket(value: object) -> str:
    if not value:
        return "unknown"
    text = str(value)
    match = re.match(r"(\d{4})[-/](\d{2})", text)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    return "unknown"


def sentence_complexity_score(text: str) -> float:
    sentences = [part for part in re.split(r"[.!?]+", text) if part.strip()]
    if not sentences:
        return 0
    words = word_count(text)
    return words / math.sqrt(len(sentences))


def excerpt(text: str, limit: int = 180) -> str:
    compact = re.sub(r"\s+", " ", text.strip())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."
