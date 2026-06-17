// ======================================================================
// SQLITE PARSER (OpenCode)
//
// 1. Parses OpenCode's internal SQLite database to extract user utterances.
// 2. Joins message, part, session, project tables.
// 3. Ported from Python parsers.py parse_opencode_sqlite().
//
//     Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
//     Copyright © 2025, Zi Liang, all rights reserved.
//     Created: 16 June 2025
// ======================================================================

use agentrace_core::models::{AgentKind, ModelInfo, Utterance};
use chrono::{DateTime, Utc};
use rusqlite::Connection;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

/// Parse an OpenCode SQLite database file.
pub fn parse_opencode_sqlite(path: &Path, agent: &AgentKind) -> Vec<Utterance> {
    let conn = match Connection::open(path) {
        Ok(c) => c,
        Err(_) => return vec![],
    };

    let query = "
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
    ";

    let mut stmt = match conn.prepare(query) {
        Ok(s) => s,
        Err(_) => return vec![],
    };

    let rows = match stmt.query_map([], |row| {
        Ok(OpenCodeRow {
            message_id: row.get(0)?,
            session_id: row.get(1)?,
            message_time_created: row.get::<_, i64>(2).ok(),
            message_data: row.get::<_, String>(3).ok(),
            part_id: row.get::<_, String>(4).ok(),
            part_time_created: row.get::<_, i64>(5).ok(),
            part_data: row.get::<_, String>(6).ok(),
            session_title: row.get::<_, String>(7).ok(),
            session_directory: row.get::<_, String>(8).ok(),
            project_worktree: row.get::<_, String>(9).ok(),
            project_name: row.get::<_, String>(10).ok(),
        })
    }) {
        Ok(r) => r,
        Err(_) => return vec![],
    };

    // Group parts by message_id
    let mut groups: HashMap<String, GroupedMessage> = HashMap::new();

    for row in rows.flatten() {
        let message_data = parse_json_str(&row.message_data);
        let role = message_data
            .get("role")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_lowercase();

        if !["user", "human", "me", "you"].contains(&role.as_str()) {
            continue;
        }

        let entry = groups.entry(row.message_id.clone()).or_insert_with(|| {
            GroupedMessage {
                session_id: row.session_id.clone(),
                time_created: row.message_time_created,
                parts: Vec::new(),
                session_title: row.session_title.clone(),
                session_directory: row.session_directory.clone(),
                project_worktree: row.project_worktree.clone(),
                project_name: row.project_name.clone(),
                message_data,
            }
        });

        let part_data = parse_json_str(&row.part_data);
        if part_data.get("type").and_then(|v| v.as_str()) == Some("text") {
            if let Some(text) = part_data.get("text").and_then(|v| v.as_str()) {
                entry.parts.push(text.to_string());
            }
        }
    }

    // Build utterances from grouped messages
    let mut utterances: Vec<Utterance> = Vec::new();

    // Sort by message_id for stable ordering (they have embedded timestamps)
    let mut group_keys: Vec<String> = groups.keys().cloned().collect();
    group_keys.sort();

    for (index, key) in group_keys.iter().enumerate() {
        let item = &groups[key];
        let text = item
            .parts
            .iter()
            .map(|p| p.trim())
            .filter(|p| !p.is_empty())
            .collect::<Vec<_>>()
            .join("\n")
            .trim()
            .to_string();

        if text.is_empty() {
            continue;
        }

        let mut metadata = HashMap::new();
        metadata.insert("parser".into(), "opencode_sqlite".into());
        if let Some(ref title) = item.session_title {
            metadata.insert("session_title".into(), title.clone());
        }
        if let Some(ref dir) = item.session_directory {
            metadata.insert("session_directory".into(), dir.clone());
        }
        if let Some(ref worktree) = item.project_worktree {
            metadata.insert("project_worktree".into(), worktree.clone());
        }
        if let Some(ref name) = item.project_name {
            metadata.insert("project_name".into(), name.clone());
        }

        let model = extract_opencode_model(&item.message_data);

        utterances.push(Utterance {
            source_path: path.to_string_lossy().to_string(),
            source_agent: agent.clone(),
            conversation_id: item.session_id.clone(),
            turn_index: index as u32,
            role: "user".into(),
            text,
            timestamp: millis_to_dt(item.time_created),
            model,
            metadata,
        });
    }

    utterances
}

struct OpenCodeRow {
    message_id: String,
    session_id: String,
    message_time_created: Option<i64>,
    message_data: Option<String>,
    part_id: Option<String>,
    part_time_created: Option<i64>,
    part_data: Option<String>,
    session_title: Option<String>,
    session_directory: Option<String>,
    project_worktree: Option<String>,
    project_name: Option<String>,
}

struct GroupedMessage {
    session_id: String,
    time_created: Option<i64>,
    parts: Vec<String>,
    session_title: Option<String>,
    session_directory: Option<String>,
    project_worktree: Option<String>,
    project_name: Option<String>,
    message_data: Value,
}

fn parse_json_str(value: &Option<String>) -> Value {
    match value {
        Some(s) => serde_json::from_str(s).unwrap_or(Value::Null),
        None => Value::Null,
    }
}

fn millis_to_dt(millis: Option<i64>) -> Option<DateTime<Utc>> {
    let ms = millis?;
    if ms > 10_000_000_000 {
        let secs = ms / 1000;
        let nsecs = ((ms % 1000) * 1_000_000) as u32;
        chrono::DateTime::from_timestamp(secs, nsecs)
    } else {
        chrono::DateTime::from_timestamp(ms, 0)
    }
}

fn extract_opencode_model(message_data: &Value) -> Option<ModelInfo> {
    let obj = message_data.as_object()?;
    let agent_str = obj.get("agent").and_then(|v| v.as_str());
    let model_str = obj.get("model").and_then(|v| v.as_str());

    match (agent_str, model_str) {
        (Some(agent), Some(model)) => {
            // agent field may be "provider/model" — split off the provider
            let provider = if let Some((prov, _)) = agent.split_once('/') {
                prov
            } else {
                agent
            };
            Some(ModelInfo {
                provider: provider.to_string(),
                model_name: model.to_string(),
            })
        }
        (Some(agent), None) if agent.contains('/') => {
            let (provider, model) = agent.split_once('/')?;
            Some(ModelInfo {
                provider: provider.to_string(),
                model_name: model.to_string(),
            })
        }
        _ => None,
    }
}

// ======================================================================
// Tests
// ======================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_nonexistent_db_returns_empty() {
        let path = Path::new("/tmp/nonexistent_opencode.db");
        let utterances = parse_opencode_sqlite(path, &AgentKind::OpenCode);
        assert!(utterances.is_empty());
    }

    #[test]
    fn millis_to_dt_converts_correctly() {
        // 2024-06-15T10:00:00Z in milliseconds
        let millis: i64 = 1_718_445_600_000;
        let dt = millis_to_dt(Some(millis)).unwrap();
        assert_eq!(dt.timestamp(), 1_718_445_600);
    }

    #[test]
    fn millis_to_dt_handles_seconds() {
        let secs: i64 = 1_718_445_600;
        let dt = millis_to_dt(Some(secs)).unwrap();
        assert_eq!(dt.timestamp(), 1_718_445_600);
    }

    #[test]
    fn millis_to_dt_none_returns_none() {
        assert!(millis_to_dt(None).is_none());
    }

    #[test]
    fn extract_model_from_agent_field() {
        let data: Value = serde_json::from_str(
            r#"{"agent": "anthropic/claude-sonnet-4-20250514", "model": "claude-sonnet-4-20250514"}"#,
        )
        .unwrap();
        let model = extract_opencode_model(&data).unwrap();
        assert_eq!(model.provider, "anthropic");
        assert_eq!(model.model_name, "claude-sonnet-4-20250514");
    }
}
