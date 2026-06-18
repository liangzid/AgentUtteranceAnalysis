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

/// System messages from agent harnesses that are not real user queries.
const FAKE_QUERY_PREFIXES: &[&str] = &[
    "Plan mode",
    "Plan approved",
    "Host final-answer readiness check",
    "<background-jobs>",
];
const FAKE_QUERY_EXACTS: &[&str] = &[
    "Background jobs finished since your last message",
];

/// Returns true if the text is a system harness message, not a real user query.
pub fn is_fake_query(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return true;
    }
    for prefix in FAKE_QUERY_PREFIXES {
        if trimmed.starts_with(prefix) {
            return true;
        }
    }
    for exact in FAKE_QUERY_EXACTS {
        if trimmed.starts_with(exact) {
            return true;
        }
    }
    false
}

/// Filter out fake/system utterances from a parsed list.
/// Only applies to user-role utterances — assistant responses are kept.
pub fn filter_fake_utterances(utterances: Vec<Utterance>) -> Vec<Utterance> {
    utterances
        .into_iter()
        .filter(|u| {
            if u.role != "user" && u.role != "human" && u.role != "me" {
                return true; // keep assistant/tool messages
            }
            !is_fake_query(&u.text)
        })
        .collect()
}

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
            .map(|n| n == "opencode.db" || n == "opencode-stable.db")
            .unwrap_or(false);

    if is_opencode_db {
        return Ok(filter_fake_utterances(sqlite_parser::parse_opencode_sqlite(path, agent)));
    }

    let raw = std::fs::read_to_string(path)?;

    let utterances = match suffix.as_str() {
        "json" => json_parser::parse_json(&raw, path, agent),
        "jsonl" | "ndjson" => json_parser::parse_jsonl(&raw, path, agent),
        "md" | "markdown" => text_parser::parse_labeled_text(
            &raw,
            path,
            agent,
            true,
        ),
        "txt" | "log" => text_parser::parse_labeled_text(
            &raw,
            path,
            agent,
            false,
        ),
        _ => vec![],
    };
    Ok(filter_fake_utterances(utterances))
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

    #[test]
    fn is_fake_query_detects_plan_mode() {
        assert!(is_fake_query("Plan mode — read-only. Explore the codebase..."));
        assert!(is_fake_query("Plan approved — plan mode is off; you're cleared..."));
        assert!(is_fake_query("Host final-answer readiness check failed..."));
        assert!(is_fake_query("<background-jobs>\nBackground jobs finished..."));
        assert!(is_fake_query("Background jobs finished since your last message: bash-1..."));
    }

    #[test]
    fn is_fake_query_passes_real_queries() {
        assert!(!is_fake_query("What is the status of this project now?"));
        assert!(!is_fake_query("I have installed the openssl. Now can you compile?"));
        assert!(!is_fake_query("Please add these agents to the discovery..."));
    }
}
