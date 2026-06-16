// ======================================================================
// `AGENTRACE-PARSER`
//
// 1. Multi-format parser: extracts user utterances from JSON, JSONL,
//    SQLite (OpenCode), Markdown, and plain text agent conversation logs.
// 2. Called by: agentrace-cli (import subcommand), agentrace-server (daemon).
// 3. Ported from Python parsers.py (Phase 2).
//
//     Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
//     Copyright © 2025, Zi Liang, all rights reserved.
//     Created: 16 June 2025
// ======================================================================

pub mod json_parser;
pub mod sqlite_parser;
pub mod text_parser;

use agentrace_core::models::{AgentKind, Utterance};
use anyhow::Result;
use std::path::Path;

/// Parse a file and extract all user utterances.
///
/// KEY REVIEW POINT: The `agent` parameter should match the AgentKind detected
/// from the file path. This is used for metadata tagging.
pub fn parse_file(path: &Path, agent: &AgentKind) -> Result<Vec<Utterance>> {
    let suffix = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    let is_opencode_db = *agent == AgentKind::OpenCode
        && path
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| n == "opencode.db")
            .unwrap_or(false);

    if is_opencode_db {
        return Ok(sqlite_parser::parse_opencode_sqlite(path, agent));
    }

    let raw = std::fs::read_to_string(path)?;

    match suffix.as_str() {
        "json" => Ok(json_parser::parse_json(&raw, path, agent)),
        "jsonl" | "ndjson" => Ok(json_parser::parse_jsonl(&raw, path, agent)),
        "md" | "markdown" => Ok(text_parser::parse_labeled_text(
            &raw,
            path,
            agent,
            true,
        )),
        "txt" | "log" => Ok(text_parser::parse_labeled_text(
            &raw,
            path,
            agent,
            false,
        )),
        _ => Ok(vec![]),
    }
}

// ======================================================================
// Tests
// ======================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_unknown_extension_returns_empty() {
        let tmp = std::env::temp_dir().join("test.xyz");
        std::fs::write(&tmp, "hello").unwrap();
        let result = parse_file(&tmp, &AgentKind::Generic("unknown".into())).unwrap();
        assert!(result.is_empty());
        std::fs::remove_file(&tmp).ok();
    }
}
