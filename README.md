# Agentrace

Track, analyze, and visualize your AI coding agent interactions. Agentrace imports your conversation logs from Claude Code, Codex, OpenCode, and other AI coding tools, builds a 3D knowledge graph with semantic embeddings, and uses LLM-powered coaching to help you become a better AI collaborator.

## Features

### Data Pipeline

- **Auto-discovery** — scans `~/.codex`, `~/.claude`, `~/.opencode`, and 7+ other agent stores
- **Multi-format parser** — JSON, JSONL, Markdown, plain text, and OpenCode SQLite
- **Incremental import** — SHA256-based dedup; re-import skips unchanged files

### Analysis

- **Heuristic stats** — word/character counts, agent distribution, time distribution
- **Property detection** — questions, requests, politeness markers, code blocks, long-context prompts
- **English naturalness** — regex-based style warnings (spelling, phrasing, sentence complexity)
- **DeepSeek Coaching** — LLM-powered interaction coach analyzes each conversation turn:
  - What you did well (positive reinforcement)
  - What could be improved (actionable feedback)
  - A better way to phrase the prompt
  - Hidden tool/command that could solve the problem directly
  - Knowledge gaps to fill
  - Clarity score (1–5)
- **Coach's Summary** — aggregate insights across all conversations: dominant interaction style, common issues, top tips

### 3D Knowledge Graph

- **Semantic embeddings** — 384-dimensional BERT embeddings via [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (local, no API calls)
- **PCA projection** — dimensionality reduction to 3D coordinates with variance explained
- **Similarity edges** — connect semantically related utterances in the graph
- **Interactive Dashboard** — Deck.gl WebGL renderer with:
  - Rotate / zoom / pitch 3D navigation
  - Click-to-inspect utterance detail popup
  - Color mode switcher: agent / time / properties
  - Time range dual-slider filter
  - Conversation trajectory lines
  - Agent distribution legend

### Single Binary

Frontend (React + TypeScript + Vite) is embedded into the Rust binary via `rust-embed`. One binary contains the full dashboard.

## Quick Start

### Build

```bash
cargo build --release -p agentrace-cli
cargo build --release -p agentrace-server
```

### Import your conversations

```bash
# Auto-discover agent stores
agentrace-cli discover

# Import from discovered locations
agentrace-cli import ~/code

# Import specific files/folders
agentrace-cli import ~/.codex/history.jsonl examples/

# Import with semantic embeddings (for knowledge graph)
agentrace-cli import --embed ~/.claude/projects/

# Force re-import changed files
agentrace-cli import --force ~/.codex/
```

### Analyze

```bash
# Basic stats analysis
agentrace-cli analyze

# LLM coaching analysis (requires DEEPSEEK_API_KEY env var)
agentrace-cli analyze --coach
```

### Knowledge Graph

```bash
# Build 3D knowledge graph from embeddings
agentrace-cli build-graph
```

### Dashboard

```bash
# Start web dashboard
agentrace-cli serve

# Or run server standalone
AGENTRACE_DB=data/agentrace.sqlite agentrace-server
```

Open `http://localhost:3000` to see the 3D knowledge graph.

## Supported Input Formats

The parser extracts user utterances from:

- JSON objects with `messages`, `conversation`, `turns`, or `items` arrays
- JSONL files (one message per line)
- Markdown transcripts (`## User`, `User:`, `Human:` labels)
- Plain text transcripts with speaker labels
- OpenCode SQLite stores (`opencode.db`)
- Codex `history.jsonl` format

Agent detection from path hints: `codex`, `claude-code`, `opencode`, `openclaw`, `kilo-code`.

## DeepSeek Coaching

Set your API key:

```bash
export DEEPSEEK_API_KEY=sk-...
```

Then run coaching analysis:

```bash
agentrace-cli analyze --coach
```

Each user utterance is paired with the next AI assistant response and sent to DeepSeek for coaching feedback:

```json
{
  "intent": "debug a Rust compilation error",
  "what_worked": "Provided the error code E0308 and relevant code snippet",
  "could_improve": "Missing the full Cargo.toml — the error might be in dependencies",
  "better_prompt": "I'm getting E0308 in Rust 1.85 with this code: ... and Cargo.toml: ...",
  "hidden_tip": "Try `cargo expand` to see the desugared code before asking the LLM",
  "knowledge_gap": "Rust type inference and turbofish syntax",
  "interaction_style": "vague",
  "clarity_score": 3
}
```

## Project Structure

```
agentrace/
├── crates/
│   ├── agentrace-core/         # Data models (Utterance, SourceFile, AgentKind)
│   ├── agentrace-discovery/    # Agent store auto-discovery
│   ├── agentrace-parser/       # JSON/JSONL/Markdown/text/SQLite parsers
│   ├── agentrace-storage/      # SQLite persistence + sqlite-vec (embeddings)
│   ├── agentrace-embedding/    # BERT embeddings via candle (all-MiniLM-L6-v2)
│   ├── agentrace-llm/          # DeepSeek API client (coaching analysis)
│   ├── agentrace-analysis/     # Stats, heuristics, PCA graph, coaching engine
│   ├── agentrace-cli/          # CLI entrypoint (clap)
│   └── agentrace-server/       # axum web server + embedded frontend
├── frontend/                   # React + TypeScript + Vite + Deck.gl
├── examples/                   # Sample conversation data
└── records/                    # Design discussion records
```

## Installation

Requires Rust 1.85+ and Node.js 22+.

```bash
git clone <repo>
cd AgentUtteranceAnalysis

# Build CLI + server
cargo build --release -p agentrace-cli -p agentrace-server

# Binaries at target/release/agentrace-cli and target/release/agentrace-server
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Rust (edition 2024) |
| Web Server | axum + tokio |
| Database | SQLite (rusqlite, bundled) |
| Vector Search | sqlite-vec |
| Embeddings | candle + all-MiniLM-L6-v2 (384-dim BERT) |
| LLM Coaching | DeepSeek API (OpenAI-compatible) |
| Frontend | React 19 + TypeScript + Vite |
| 3D Rendering | Deck.gl + WebGL |
| UI Components | shadcn/ui + Tailwind CSS |

## Tests

```bash
cargo test --lib    # 76 unit/integration tests
```

## License

MIT
