# AgentUtteranceAnalysis

AgentUtteranceAnalysis imports your dialogue utterances with AI coding agents, normalizes them into one local database, exports them for review, and analyzes the properties and English naturalness of your own messages.

The first version is intentionally local-first:

- Batch import from folders or files.
- Incremental import using file hashes and modification times.
- Flexible parsing for JSON, JSONL, Markdown, and plain text exports.
- Agent/source detection for tools such as OpenClaw, Claude Code, OpenCode, Codex, Kilo Code, and other agent logs.
- Export to JSONL, CSV, or Markdown.
- Analysis reports for utterance length, questions/requests, politeness markers, language quality warnings, and distribution by agent/date/source.

## Install

From this project directory:

```bash
uv sync --extra dev
```

You can also install with pip:

```bash
python -m pip install -e ".[dev]"
```

## Quick Start

Find likely agent dialogue stores automatically:

```bash
uv run aua discover
uv run aua discover --project-glob "~/code/*"
```

The CLI uses themed tables and spinner animations by default:

```bash
uv run aua discover --theme dark
uv run aua discover --theme light
uv run aua discover --theme mono
uv run aua discover --plain
```

Use `--plain` for logs, scripts, or terminals where ANSI styling is not wanted.

Import discovered conversations:

```bash
uv run aua import --db data/utterances.sqlite
```

Import conversations from one or more explicit folders:

```bash
uv run aua import ~/path/to/agent/logs --db data/utterances.sqlite
```

Export all normalized utterances:

```bash
uv run aua export --db data/utterances.sqlite --format markdown --output exports/utterances.md
uv run aua export --db data/utterances.sqlite --format jsonl --output exports/utterances.jsonl
uv run aua export --db data/utterances.sqlite --format csv --output exports/utterances.csv
```

Analyze your utterances:

```bash
uv run aua analyze --db data/utterances.sqlite --output reports/analysis.md
```

Run a full pipeline:

```bash
uv run aua run --db data/utterances.sqlite --export exports/utterances.md --report reports/analysis.md
```

Analyze only English-dominant user utterances:

```bash
uv run aua run --english-only --project-glob "~/code/*" --db data/english_utterances.sqlite --export exports/english_utterances.md --report reports/english_analysis.md
```

`import` and `run` default to auto-discovery when no paths are supplied. Use `--project-glob` to control where project-level agent folders are searched, such as `--project-glob "~/code/*"`.

## Supported Input Shapes

The importer is deliberately tolerant. It extracts user utterances from common patterns:

- JSON objects with `messages`, `conversation`, `turns`, or `items`.
- Message fields such as `role`, `speaker`, `author`, `type`, `content`, `text`, or `message`.
- JSONL files with one message or conversation object per line.
- Markdown transcripts with headings such as `## User`, `### Human`, `User:`, or `You:`.
- Plain text transcripts with speaker labels.
- OpenCode's SQLite store at `~/.local/share/opencode/opencode.db`.

Files that cannot be parsed are skipped and reported in the import summary.

## Auto-Discovery

The discovery command currently checks common global locations and project-local hidden agent directories:

- `~/.local/share/opencode/opencode.db`
- `~/.local/share/opencode/log`
- `~/.codex`, `~/.claude`, `~/.openclaw`
- `~/.config/opencode`, `~/.config/claude`
- `~/.kilo`, `~/.kilo-code`, `~/.kilocode`
- `.codex`, `.claude`, `.opencode`, `.openclaw`, `.kilo`, `.kilo-code`, `.kilocode` under project globs such as `~/code/*`

Large dependency, container, telemetry, generated context, and VCS directories such as `node_modules`, `.git`, `snapshot`, `.docker`, `docker`, `containers`, `overlay2`, `volumes`, `telemetry`, `.codex/sessions`, `dist`, and `build` are skipped.

## Project Layout

```text
src/agent_utterance_analysis/
  cli.py          Command line interface.
  discovery.py    Auto-discovery for global and project-local agent stores.
  importer.py     Batch and incremental source scanning.
  parsers.py      JSON/JSONL/Markdown/text extraction.
  models.py       Normalized data model.
  storage.py      SQLite persistence.
  export.py       JSONL, CSV, and Markdown output.
  analysis.py     Distribution and English-quality analysis.
tests/
  test_*.py       Core behavior tests.
examples/
  sample_codex.json
```

## Notes

The English analysis is heuristic in this initial version. It is useful for finding likely unnatural phrasing, typo-prone patterns, overlong requests, repeated wording, and distribution trends. A later version can add an optional LLM reviewer for deeper grammar and naturalness feedback.
