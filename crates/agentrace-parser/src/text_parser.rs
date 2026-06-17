// ======================================================================
// TEXT PARSER
//
// 1. Extracts user utterances from Markdown and plain text files using
//    label-based regex matching (e.g., "## User", "User:", "Human:").
// 2. Ported from Python parsers.py parse_labeled_text().
//
//     Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
//     Copyright © 2025, Zi Liang, all rights reserved.
//     Created: 16 June 2025
// ======================================================================

use agentrace_core::models::{AgentKind, Utterance};
use regex::Regex;
use std::collections::HashMap;
use std::path::Path;

/// Parse a Markdown or plain text file with labeled user/assistant sections.
pub fn parse_labeled_text(
    raw: &str,
    path: &Path,
    agent: &AgentKind,
    markdown: bool,
) -> Vec<Utterance> {
    let user_label = Regex::new(
        r"(?i)^(?:#{1,6}\s*)?(user|human|me|you)\s*:?\s*$|^(?:#{1,6}\s*)?(user|human|me|you)\s*:\s*(.*)$",
    )
    .unwrap();
    let other_label = Regex::new(
        r"(?i)^(?:#{1,6}\s*)?(assistant|agent|ai|claude|codex|openclaw|opencode|kilo(?: ?code)?)\s*:?\s*$|^(?:#{1,6}\s*)?(assistant|agent|ai|claude|codex|openclaw|opencode|kilo(?: ?code)?)\s*:",
    )
    .unwrap();

    let mut utterances: Vec<Utterance> = Vec::new();
    let mut current_lines: Vec<String> = Vec::new();
    let mut capturing = false;
    let mut turn: u32 = 0;
    let conversation_id = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    let parser_name = if markdown { "markdown" } else { "text" };

    let flush = |current_lines: &mut Vec<String>, utterances: &mut Vec<Utterance>, turn: &mut u32| {
        let text = current_lines.join("\n").trim().to_string();
        if !text.is_empty() {
            let mut metadata = HashMap::new();
            metadata.insert("parser".into(), parser_name.to_string());
            utterances.push(Utterance {
                source_path: path.to_string_lossy().to_string(),
                source_agent: agent.clone(),
                conversation_id: conversation_id.clone(),
                turn_index: *turn,
                role: "user".into(),
                text,
                timestamp: None,
                model: None,
                metadata,
            });
            *turn += 1;
        }
        current_lines.clear();
    };

    for line in raw.lines() {
        let trimmed = line.trim();
        if let Some(caps) = user_label.captures(trimmed) {
            flush(&mut current_lines, &mut utterances, &mut turn);
            capturing = true;
            // Check for inline text after label (group 3)
            if let Some(inline) = caps.get(3) {
                if !inline.as_str().is_empty() {
                    current_lines.push(inline.as_str().to_string());
                }
            }
            continue;
        }
        if other_label.is_match(trimmed) {
            flush(&mut current_lines, &mut utterances, &mut turn);
            capturing = false;
            continue;
        }
        if capturing {
            current_lines.push(line.to_string());
        }
    }
    flush(&mut current_lines, &mut utterances, &mut turn);

    utterances
}

// ======================================================================
// Tests
// ======================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn test_path() -> PathBuf {
        PathBuf::from("/tmp/test.md")
    }

    fn test_agent() -> AgentKind {
        AgentKind::ClaudeCode
    }

    #[test]
    fn parse_markdown_with_user_labels() {
        let raw = "## User\nhello world\n\n## Assistant\nresponse here\n\n## User\nanother question";
        let utterances = parse_labeled_text(raw, &test_path(), &test_agent(), true);
        assert_eq!(utterances.len(), 2);
        assert_eq!(utterances[0].text, "hello world");
        assert_eq!(utterances[1].text, "another question");
        assert_eq!(utterances[0].turn_index, 0);
        assert_eq!(utterances[1].turn_index, 1);
    }

    #[test]
    fn parse_with_inline_user_label() {
        let raw = "User: what is rust?\n\nAssistant: a programming language";
        let utterances = parse_labeled_text(raw, &test_path(), &test_agent(), false);
        assert_eq!(utterances.len(), 1);
        assert_eq!(utterances[0].text, "what is rust?");
    }

    #[test]
    fn parse_human_labels() {
        let raw = "Human: help\n\n## Assistant: I'm here";
        let utterances = parse_labeled_text(raw, &test_path(), &test_agent(), true);
        assert_eq!(utterances.len(), 1);
        assert_eq!(utterances[0].text, "help");
    }

    #[test]
    fn parse_empty_returns_empty() {
        let utterances = parse_labeled_text("", &test_path(), &test_agent(), false);
        assert_eq!(utterances.len(), 0);
    }

    #[test]
    fn parse_no_user_labels_returns_empty() {
        let raw = "## Assistant\nonly assistant messages here";
        let utterances = parse_labeled_text(raw, &test_path(), &test_agent(), true);
        assert_eq!(utterances.len(), 0);
    }

    #[test]
    fn parse_multiline_utterance() {
        let raw = "## User\nline one\nline two\nline three\n\n## Assistant\nreply";
        let utterances = parse_labeled_text(raw, &test_path(), &test_agent(), true);
        assert_eq!(utterances.len(), 1);
        assert_eq!(utterances[0].text, "line one\nline two\nline three");
    }

    #[test]
    fn parse_metadata_contains_parser_name() {
        let raw = "## User\nhello";
        let utterances = parse_labeled_text(raw, &test_path(), &test_agent(), true);
        assert_eq!(utterances[0].metadata.get("parser").unwrap(), "markdown");
    }

    #[test]
    fn parse_text_uses_text_parser_name() {
        let raw = "User: hello";
        let utterances = parse_labeled_text(raw, &test_path(), &test_agent(), false);
        assert_eq!(utterances[0].metadata.get("parser").unwrap(), "text");
    }
}
