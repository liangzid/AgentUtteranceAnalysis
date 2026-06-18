// ======================================================================
// JSON PARSER
//
// 1. Extracts user utterances from JSON and JSONL agent conversation logs.
// 2. Handles nested message lists, Codex history.jsonl format, and
//    generic JSON objects with user-role fields.
//
//     Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
//     Copyright © 2025, Zi Liang, all rights reserved.
//     Created: 16 June 2025
// ======================================================================

use agentrace_core::models::{AgentKind, ModelInfo, Utterance};
use chrono::{DateTime, Utc};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

/// User-role strings that indicate a user message.
const USER_ROLES: &[&str] = &["user", "human", "me", "you", "requester", "client"];
/// Roles we collect — both user messages and AI assistant responses.
const COLLECTED_ROLES: &[&str] = &[
    "user", "human", "me", "you", "requester", "client",
    "assistant", "agent", "tool",
];
const KNOWN_ROLES: &[&str] = &[
    "user", "human", "me", "you", "requester", "client",
    "assistant", "agent", "system", "developer", "tool",
];
const CONVERSATION_KEYS: &[&str] = &["messages", "conversation", "turns", "items", "entries"];
const TEXT_KEYS: &[&str] = &["content", "text", "message", "body", "prompt"];
const ROLE_KEYS: &[&str] = &["role", "speaker", "author", "from", "type", "kind"];
const TIME_KEYS: &[&str] = &["timestamp", "created_at", "createdAt", "time", "date"];

/// Parse a single JSON file into utterances.
pub fn parse_json(raw: &str, path: &Path, agent: &AgentKind) -> Vec<Utterance> {
    let data: Value = match serde_json::from_str(raw) {
        Ok(v) => v,
        Err(_) => return vec![],
    };
    utterances_from_json_value(&data, path, agent, None)
}

/// Parse a JSONL file (one JSON object per line) into utterances.
pub fn parse_jsonl(raw: &str, path: &Path, agent: &AgentKind) -> Vec<Utterance> {
    let mut output: Vec<Utterance> = Vec::new();
    for (line_number, line) in raw.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let data: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let hint = format!("line-{}", line_number + 1);
        let mut found = utterances_from_json_value(&data, path, agent, Some(&hint));
        output.append(&mut found);
    }
    renumber(&output)
}

/// Core JSON extraction: walk a Value tree and extract user utterances.
pub fn utterances_from_json_value(
    data: &Value,
    path: &Path,
    agent: &AgentKind,
    conversation_hint: Option<&str>,
) -> Vec<Utterance> {
    let id_hint = conversation_hint.unwrap_or("unknown");
    let fallback_id = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(id_hint);
    let mut utterances: Vec<Utterance> = Vec::new();

    // Search for message lists
    for (conversation_id, messages) in find_message_lists(data, fallback_id) {
        for (index, message) in messages.iter().enumerate() {
            let role = extract_role(message);
            if role.is_none() || !COLLECTED_ROLES.contains(&role.as_ref().unwrap().as_str()) {
                continue;
            }
            let role_str = role.clone().unwrap();
            let text = extract_text(message);
            if text.is_empty() {
                continue;
            }
            let ts = extract_timestamp(message);
            let mut metadata = HashMap::new();
            metadata.insert("parser".into(), "json".into());
            metadata.insert("role".into(), role_str.clone());

            // Try to extract model info
            let model = extract_model_info(message);

            utterances.push(Utterance {
                source_path: path.to_string_lossy().to_string(),
                source_agent: agent.clone(),
                conversation_id: conversation_id.clone(),
                turn_index: index as u32,
                role: role_str,
                text,
                timestamp: ts,
                model,
                metadata,
            });
        }
    }

    // Fallback: Codex history.jsonl format
    if utterances.is_empty() && data.is_object() {
        if is_codex_history_entry(data, path) {
            let text = extract_text(data);
            if !text.is_empty() {
                let conv_id = data
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or(fallback_id)
                    .to_string();
                let ts = data.get("ts").and_then(|v| v.as_i64()).and_then(epoch_to_iso);
                let mut metadata = HashMap::new();
                metadata.insert("parser".into(), "codex_history".into());
                utterances.push(Utterance {
                    source_path: path.to_string_lossy().to_string(),
                    source_agent: agent.clone(),
                    conversation_id: conv_id,
                    turn_index: 0,
                    role: "user".into(),
                    text,
                    timestamp: ts,
                    model: None,
                    metadata,
                });
            }
        }
    }

    // Fallback: single dict with user role
    if utterances.is_empty() && data.is_object() {
        let obj = data.as_object().unwrap();
        let role = extract_role(data);
        if let Some(ref r) = role {
            if USER_ROLES.contains(&r.as_str()) {
                let text = extract_text(data);
                if !text.is_empty() {
                    let mut metadata = HashMap::new();
                    metadata.insert("parser".into(), "json".into());
                    metadata.insert("role".into(), r.clone());
                    utterances.push(Utterance {
                        source_path: path.to_string_lossy().to_string(),
                        source_agent: agent.clone(),
                        conversation_id: fallback_id.to_string(),
                        turn_index: 0,
                        role: r.clone(),
                        text,
                        timestamp: extract_timestamp(data),
                        model: None,
                        metadata,
                    });
                }
            }
        }
    }

    // Fallback: Reasonix checkpoint files — objects with "prompt" key
    // indicating a user message, but no explicit role field.
    if utterances.is_empty() && data.is_object() {
        let obj = data.as_object().unwrap();
        if obj.contains_key("prompt") && extract_role(data).is_none() {
            if let Some(text) = obj.get("prompt").and_then(|v| v.as_str()) {
                let text = text.trim().to_string();
                if !text.is_empty() {
                    let ts = obj
                        .get("time")
                        .and_then(|v| v.as_str())
                        .and_then(|s| {
                            chrono::DateTime::parse_from_rfc3339(s)
                                .ok()
                                .map(|dt| dt.with_timezone(&Utc))
                        });
                    let mut metadata = HashMap::new();
                    metadata.insert("parser".into(), "reasonix_ckpt".into());
                    metadata.insert("role".into(), "user".into());
                    if let Some(turn) = obj.get("turn").and_then(|v| v.as_u64()) {
                        metadata.insert("source_turn".into(), turn.to_string());
                    }
                    utterances.push(Utterance {
                        source_path: path.to_string_lossy().to_string(),
                        source_agent: agent.clone(),
                        conversation_id: fallback_id.to_string(),
                        turn_index: 0,
                        role: "user".into(),
                        text,
                        timestamp: ts,
                        model: None,
                        metadata,
                    });
                }
            }
        }
    }

    renumber(&utterances)
}

// --- Message list discovery ---

fn find_message_lists(data: &Value, fallback_id: &str) -> Vec<(String, Vec<Value>)> {
    let mut results: Vec<(String, Vec<Value>)> = Vec::new();

    if data.is_array() {
        let arr = data.as_array().unwrap();
        if !arr.is_empty() && arr.iter().all(|item| item.is_object()) {
            results.push((fallback_id.to_string(), arr.clone()));
        }
        return results;
    }

    if !data.is_object() {
        return results;
    }

    let obj = data.as_object().unwrap();

    let conversation_id = obj
        .get("id")
        .or_else(|| obj.get("conversation_id"))
        .or_else(|| obj.get("conversationId"))
        .or_else(|| obj.get("session_id"))
        .and_then(|v| v.as_str())
        .unwrap_or(fallback_id)
        .to_string();

    for key in CONVERSATION_KEYS {
        if let Some(val) = obj.get(*key) {
            if let Some(arr) = val.as_array() {
                results.push((conversation_id.clone(), arr.clone()));
            }
        }
    }

    for (key, value) in obj.iter() {
        if value.is_object() {
            let nested = find_message_lists(value, &format!("{}:{}", conversation_id, key));
            results.extend(nested);
        } else if let Some(arr) = value.as_array() {
            if !CONVERSATION_KEYS.contains(&key.as_str()) {
                for (idx, item) in arr.iter().enumerate() {
                    if item.is_object() {
                        let nested = find_message_lists(
                            item,
                            &format!("{}:{}-{}", conversation_id, key, idx),
                        );
                        results.extend(nested);
                    }
                }
            }
        }
    }

    results
}

// --- Role extraction ---

pub fn extract_role(message: &Value) -> Option<String> {
    let obj = message.as_object()?;
    for key in ROLE_KEYS {
        if let Some(value) = obj.get(*key) {
            let role_str = if value.is_object() {
                value
                    .get("role")
                    .or_else(|| value.get("name"))
                    .or_else(|| value.get("type"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            } else if let Some(s) = value.as_str() {
                Some(s.to_string())
            } else {
                None
            };

            if let Some(role) = role_str {
                let lower = role.trim().to_lowercase();
                if (key == &"type" || key == &"kind")
                    && !KNOWN_ROLES.contains(&lower.as_str())
                    && lower != "user_message"
                    && lower != "human_message"
                {
                    continue;
                }
                if lower == "user_message" || lower == "human_message" {
                    return Some("user".to_string());
                }
                return Some(lower);
            }
        }
    }

    // Try nested payload
    if let Some(payload) = obj.get("payload") {
        if payload.is_object() {
            return extract_role(payload);
        }
    }

    None
}

// --- Text extraction ---

pub fn extract_text(message: &Value) -> String {
    if let Some(s) = message.as_str() {
        return s.trim().to_string();
    }
    let obj = match message.as_object() {
        Some(o) => o,
        None => return String::new(),
    };

    for key in TEXT_KEYS {
        if let Some(value) = obj.get(*key) {
            let text = coerce_text(value);
            if !text.is_empty() {
                return text;
            }
        }
    }

    // Try nested message
    if let Some(nested_msg) = obj.get("message") {
        if nested_msg.is_object() {
            let role = extract_role(nested_msg);
            if role.is_none() || USER_ROLES.contains(&role.unwrap().as_str()) {
                let text = extract_text(nested_msg);
                if !text.is_empty() {
                    return text;
                }
            }
        }
    }

    // Try payload
    if let Some(payload) = obj.get("payload") {
        if payload.is_object() {
            let role = extract_role(payload);
            if role.is_none() || USER_ROLES.contains(&role.unwrap().as_str()) {
                let text = extract_text(payload);
                if !text.is_empty() {
                    return text;
                }
            }
        }
    }

    String::new()
}

fn coerce_text(value: &Value) -> String {
    if let Some(s) = value.as_str() {
        return s.trim().to_string();
    }
    if let Some(arr) = value.as_array() {
        let parts: Vec<String> = arr
            .iter()
            .filter_map(|item| {
                if let Some(s) = item.as_str() {
                    Some(s.to_string())
                } else if let Some(obj) = item.as_object() {
                    obj.get("text")
                        .or_else(|| obj.get("content"))
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                } else {
                    None
                }
            })
            .collect();
        return parts.join("\n").trim().to_string();
    }
    if let Some(obj) = value.as_object() {
        return obj
            .get("text")
            .or_else(|| obj.get("content"))
            .or_else(|| obj.get("value"))
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .unwrap_or_default();
    }
    String::new()
}

// --- Timestamp extraction ---

fn extract_timestamp(message: &Value) -> Option<DateTime<Utc>> {
    let obj = message.as_object()?;
    for key in TIME_KEYS {
        if let Some(value) = obj.get(*key) {
            if let Some(s) = value.as_str() {
                if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) {
                    return Some(dt.with_timezone(&Utc));
                }
            }
            if let Some(n) = value.as_i64() {
                return epoch_to_iso(n);
            }
            if let Some(n) = value.as_f64() {
                return epoch_to_iso(n as i64);
            }
        }
    }
    None
}

fn epoch_to_iso(value: i64) -> Option<DateTime<Utc>> {
    if value > 10_000_000_000 {
        // milliseconds
        let secs = value / 1000;
        let nsecs = ((value % 1000) * 1_000_000) as u32;
        chrono::DateTime::from_timestamp(secs, nsecs)
    } else {
        chrono::DateTime::from_timestamp(value, 0)
    }
}

// --- Model info extraction ---

fn extract_model_info(message: &Value) -> Option<ModelInfo> {
    let obj = message.as_object()?;
    let provider = obj
        .get("model_provider")
        .or_else(|| obj.get("provider"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let model_name = obj
        .get("model")
        .or_else(|| obj.get("model_name"))
        .or_else(|| obj.get("modelName"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    // Also check in the `agent` field (format: "provider/model")
    if provider.is_none() {
        if let Some(agent_str) = obj.get("agent").and_then(|v| v.as_str()) {
            if let Some((prov, model)) = agent_str.split_once('/') {
                return Some(ModelInfo {
                    provider: prov.to_string(),
                    model_name: model.to_string(),
                });
            }
        }
    }

    match (provider, model_name) {
        (Some(p), Some(m)) => Some(ModelInfo {
            provider: p,
            model_name: m,
        }),
        _ => None,
    }
}

// --- Codex history detection ---

fn is_codex_history_entry(data: &Value, path: &Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .map(|n| n == "history.jsonl")
        .unwrap_or(false)
        && data.get("session_id").is_some()
        && data.get("text").is_some()
}

// --- Renumber utterances by conversation order ---

pub fn renumber(utterances: &[Utterance]) -> Vec<Utterance> {
    utterances
        .iter()
        .enumerate()
        .map(|(index, item)| Utterance {
            turn_index: index as u32,
            role: item.role.clone(),
            ..item.clone()
        })
        .collect()
}

// ======================================================================
// Tests
// ======================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn test_path() -> PathBuf {
        PathBuf::from("/tmp/test.json")
    }

    fn test_agent() -> AgentKind {
        AgentKind::Codex
    }

    #[test]
    fn parse_json_with_messages() {
        let raw = r#"{
            "id": "conv-1",
            "messages": [
                {"role": "user", "content": "hello world"},
                {"role": "assistant", "content": "hi there"},
                {"role": "user", "content": "help me"}
            ]
        }"#;
        let utterances = parse_json(raw, &test_path(), &test_agent());
        assert_eq!(utterances.len(), 3);
        assert_eq!(utterances[0].text, "hello world");
        assert_eq!(utterances[0].role, "user");
        assert_eq!(utterances[1].text, "hi there");
        assert_eq!(utterances[1].role, "assistant");
        assert_eq!(utterances[2].text, "help me");
        assert_eq!(utterances[2].role, "user");
        assert_eq!(utterances[0].conversation_id, "conv-1");
    }

    #[test]
    fn parse_jsonl_multiple_lines() {
        let raw = r#"{"role": "user", "text": "first"}
{"role": "user", "text": "second"}
{"role": "assistant", "text": "reply"}"#;
        let utterances = parse_jsonl(raw, &test_path(), &test_agent());
        assert_eq!(utterances.len(), 2);
        assert_eq!(utterances[0].text, "first");
        assert_eq!(utterances[1].text, "second");
    }

    #[test]
    fn parse_json_single_user_object() {
        let raw = r#"{"role": "user", "content": "single message"}"#;
        let utterances = parse_json(raw, &test_path(), &test_agent());
        assert_eq!(utterances.len(), 1);
        assert_eq!(utterances[0].text, "single message");
    }

    #[test]
    fn parse_json_empty_returns_empty() {
        let utterances = parse_json("{}", &test_path(), &test_agent());
        assert_eq!(utterances.len(), 0);
    }

    #[test]
    fn parse_json_invalid_returns_empty() {
        let utterances = parse_json("not json", &test_path(), &test_agent());
        assert_eq!(utterances.len(), 0);
    }

    #[test]
    fn parse_jsonl_empty_lines_skipped() {
        let raw = "\n\n{\"role\": \"user\", \"text\": \"ok\"}\n\n";
        let utterances = parse_jsonl(raw, &test_path(), &test_agent());
        assert_eq!(utterances.len(), 1);
    }

    #[test]
    fn parse_json_with_timestamp() {
        let raw = r#"{
            "messages": [
                {"role": "user", "content": "test", "timestamp": "2024-06-15T10:00:00Z"}
            ]
        }"#;
        let utterances = parse_json(raw, &test_path(), &test_agent());
        assert_eq!(utterances.len(), 1);
        assert!(utterances[0].timestamp.is_some());
    }

    #[test]
    fn parse_json_with_model_info() {
        let raw = r#"{
            "messages": [
                {
                    "role": "user",
                    "content": "test",
                    "agent": "anthropic/claude-sonnet-4-20250514"
                }
            ]
        }"#;
        let utterances = parse_json(raw, &test_path(), &test_agent());
        assert_eq!(utterances.len(), 1);
        let model = utterances[0].model.as_ref().unwrap();
        assert_eq!(model.provider, "anthropic");
        assert_eq!(model.model_name, "claude-sonnet-4-20250514");
    }

    #[test]
    fn parse_json_nested_conversation_keys() {
        let raw = r#"{
            "data": {
                "conversation": [
                    {"role": "user", "content": "nested message"}
                ]
            }
        }"#;
        let utterances = parse_json(raw, &test_path(), &test_agent());
        assert_eq!(utterances.len(), 1);
        assert_eq!(utterances[0].text, "nested message");
    }

    #[test]
    fn extract_role_detects_user_variants() {
        let v: Value = serde_json::from_str(r#"{"role": "human"}"#).unwrap();
        assert_eq!(extract_role(&v).unwrap(), "human");

        let v: Value = serde_json::from_str(r#"{"type": "user"}"#).unwrap();
        assert_eq!(extract_role(&v).unwrap(), "user");

        let v: Value = serde_json::from_str(r#"{"speaker": "me"}"#).unwrap();
        assert_eq!(extract_role(&v).unwrap(), "me");
    }

    #[test]
    fn extract_role_skips_assistant() {
        let v: Value = serde_json::from_str(r#"{"role": "assistant"}"#).unwrap();
        let role = extract_role(&v).unwrap();
        assert!(!USER_ROLES.contains(&role.as_str()));
    }

    #[test]
    fn extract_text_from_multiple_keys() {
        let v: Value = serde_json::from_str(r#"{"content": "hello"}"#).unwrap();
        assert_eq!(extract_text(&v), "hello");

        let v: Value = serde_json::from_str(r#"{"text": "world"}"#).unwrap();
        assert_eq!(extract_text(&v), "world");

        let v: Value = serde_json::from_str(r#"{"body": "greetings"}"#).unwrap();
        assert_eq!(extract_text(&v), "greetings");
    }

    #[test]
    fn extract_text_from_deeply_nested_payload() {
        let raw = r#"{"payload": {"message": {"content": "deeply nested"}}}"#;
        let v: Value = serde_json::from_str(raw).unwrap();
        assert_eq!(extract_text(&v), "deeply nested");
    }

    #[test]
    fn parse_reasonix_ckpt_turn_json() {
        let raw = r#"{"turn": 3, "time": "2026-06-17T13:57:55.196761661+08:00", "prompt": "Fix the parser bug", "msgIndex": 6, "files": null}"#;
        let utterances = parse_json(raw, &test_path(), &AgentKind::Reasonix);
        assert_eq!(utterances.len(), 1);
        assert_eq!(utterances[0].text, "Fix the parser bug");
        assert_eq!(utterances[0].role, "user");
        assert!(utterances[0].timestamp.is_some());
        assert_eq!(
            utterances[0].metadata.get("parser").unwrap(),
            "reasonix_ckpt"
        );
        // Source turn number preserved in metadata
        assert_eq!(
            utterances[0].metadata.get("source_turn").unwrap(),
            "3"
        );
    }
}
