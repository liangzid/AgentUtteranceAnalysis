// ======================================================================
// `MODELS`
//
// 1. Core data types for agentrace — Utterance, SourceFile, Conversation, etc.
// 2. Used by all other crates as the shared type foundation.
// 3. Modification history:
//    - 16 June 2025: Initial creation
//
//     Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
//     Copyright © 2025, Zi Liang, all rights reserved.
//     Created: 16 June 2025
// ======================================================================

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

/// The AI coding agent that produced the conversation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AgentKind {
    OpenCode,
    Codex,
    ClaudeCode,
    OpenClaw,
    KiloCode,
    Reasonix,
    CodeWhale,
    Generic(String),
}

impl AgentKind {
    /// Detect agent kind from a file path or directory name.
    pub fn from_path_hint(path: &str) -> Self {
        let lower = path.to_lowercase();
        if lower.contains("opencode") {
            Self::OpenCode
        } else if lower.contains("codex") {
            Self::Codex
        } else if lower.contains("claude") {
            Self::ClaudeCode
        } else if lower.contains("openclaw") {
            Self::OpenClaw
        } else if lower.contains("kilo") {
            Self::KiloCode
        } else if lower.contains("reasonix") {
            Self::Reasonix
        } else if lower.contains("codewhale") {
            Self::CodeWhale
        } else {
            Self::Generic("unknown".into())
        }
    }
}

impl std::fmt::Display for AgentKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OpenCode => write!(f, "opencode"),
            Self::Codex => write!(f, "codex"),
            Self::ClaudeCode => write!(f, "claude-code"),
            Self::OpenClaw => write!(f, "openclaw"),
            Self::KiloCode => write!(f, "kilo-code"),
            Self::Reasonix => write!(f, "reasonix"),
            Self::CodeWhale => write!(f, "codewhale"),
            Self::Generic(s) => write!(f, "{}", s),
        }
    }
}

/// The LLM model used in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ModelInfo {
    pub provider: String,
    pub model_name: String,
}

impl std::fmt::Display for ModelInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.provider, self.model_name)
    }
}

/// A source file on disk containing agent conversation logs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceFile {
    pub path: String,
    pub agent: AgentKind,
    pub sha256: String,
    pub mtime_ns: i64,
    pub size: u64,
}

/// A single user utterance extracted from an agent conversation.
///
/// KEY REVIEW POINT: The stable_id computation uses SHA256 of all core fields.
/// Changing which fields are included changes the dedup behaviour.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Utterance {
    pub source_path: String,
    pub source_agent: AgentKind,
    pub conversation_id: String,
    pub turn_index: u32,
    pub role: String,
    pub text: String,
    pub timestamp: Option<DateTime<Utc>>,
    pub model: Option<ModelInfo>,
    pub metadata: HashMap<String, String>,
}

impl Utterance {
    /// Compute a stable, content-addressed ID for deduplication.
    pub fn stable_id(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.source_path.as_bytes());
        hasher.update(self.source_agent.to_string().as_bytes());
        hasher.update(self.conversation_id.as_bytes());
        hasher.update(self.turn_index.to_le_bytes());
        hasher.update(self.role.as_bytes());
        hasher.update(self.text.as_bytes());
        let result = hasher.finalize();
        format!("{:x}", result)
    }
}

/// A conversation — multiple utterances within one session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub id: String,
    pub agent: AgentKind,
    pub model: Option<ModelInfo>,
    pub title: Option<String>,
    pub project: Option<String>,
    pub utterances: Vec<Utterance>,
    pub started_at: Option<DateTime<Utc>>,
    pub ended_at: Option<DateTime<Utc>>,
}

/// Result of an analysis run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    pub run_id: String,
    pub created_at: DateTime<Utc>,
    pub summary: serde_json::Value,
}

/// The type of task inferred from an utterance.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum TaskKind {
    CodeGeneration,
    Debugging,
    Refactoring,
    Exploration,
    FeatureRequest,
    SecurityRelated,
    Learning,
    Review,
    Deployment,
    Unknown,
}

// ======================================================================
// Tests
// ======================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- AgentKind ---

    #[test]
    fn agent_kind_from_path_opencode() {
        let kind = AgentKind::from_path_hint("/home/user/.local/share/opencode/opencode.db");
        assert_eq!(kind, AgentKind::OpenCode);
    }

    #[test]
    fn agent_kind_from_path_codex() {
        let kind = AgentKind::from_path_hint("/home/user/.codex/history.jsonl");
        assert_eq!(kind, AgentKind::Codex);
    }

    #[test]
    fn agent_kind_from_path_claude() {
        let kind = AgentKind::from_path_hint("/home/user/.claude/projects/foo/conv.jsonl");
        assert_eq!(kind, AgentKind::ClaudeCode);
    }

    #[test]
    fn agent_kind_from_path_openclaw() {
        let kind = AgentKind::from_path_hint("/home/user/.openclaw/logs/session.jsonl");
        assert_eq!(kind, AgentKind::OpenClaw);
    }

    #[test]
    fn agent_kind_from_path_kilo() {
        let kind = AgentKind::from_path_hint("/home/user/.kilo-code/history.jsonl");
        assert_eq!(kind, AgentKind::KiloCode);
    }

    #[test]
    fn agent_kind_from_path_reasonix() {
        let kind = AgentKind::from_path_hint("/home/user/.config/reasonix/sessions/sess.jsonl");
        assert_eq!(kind, AgentKind::Reasonix);
    }

    #[test]
    fn agent_kind_from_path_codewhale() {
        let kind = AgentKind::from_path_hint("/home/user/.codewhale/sessions/sess.jsonl");
        assert_eq!(kind, AgentKind::CodeWhale);
    }

    #[test]
    fn agent_kind_from_path_unknown() {
        let kind = AgentKind::from_path_hint("/tmp/random.txt");
        assert_eq!(kind, AgentKind::Generic("unknown".into()));
    }

    #[test]
    fn agent_kind_display() {
        assert_eq!(AgentKind::OpenCode.to_string(), "opencode");
        assert_eq!(AgentKind::Codex.to_string(), "codex");
        assert_eq!(AgentKind::ClaudeCode.to_string(), "claude-code");
        assert_eq!(AgentKind::OpenClaw.to_string(), "openclaw");
        assert_eq!(AgentKind::KiloCode.to_string(), "kilo-code");
        assert_eq!(AgentKind::Reasonix.to_string(), "reasonix");
        assert_eq!(AgentKind::CodeWhale.to_string(), "codewhale");
        assert_eq!(AgentKind::Generic("custom".into()).to_string(), "custom");
    }

    #[test]
    fn agent_kind_serde_roundtrip() {
        let kind = AgentKind::Codex;
        let json = serde_json::to_string(&kind).unwrap();
        let back: AgentKind = serde_json::from_str(&json).unwrap();
        assert_eq!(kind, back);
    }

    // --- ModelInfo ---

    #[test]
    fn model_info_display() {
        let info = ModelInfo {
            provider: "anthropic".into(),
            model_name: "claude-sonnet-4-20250514".into(),
        };
        assert_eq!(info.to_string(), "anthropic/claude-sonnet-4-20250514");
    }

    // --- Utterance ---

    fn make_test_utterance(text: &str) -> Utterance {
        Utterance {
            source_path: "/test/conv.json".into(),
            source_agent: AgentKind::Codex,
            conversation_id: "conv-001".into(),
            turn_index: 1,
            role: "user".into(),
            text: text.into(),
            timestamp: None,
            model: None,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn utterance_stable_id_is_deterministic() {
        let u1 = make_test_utterance("hello world");
        let u2 = make_test_utterance("hello world");
        assert_eq!(u1.stable_id(), u2.stable_id());
    }

    #[test]
    fn utterance_stable_id_changes_with_text() {
        let u1 = make_test_utterance("hello world");
        let u2 = make_test_utterance("hello world!");
        assert_ne!(u1.stable_id(), u2.stable_id());
    }

    #[test]
    fn utterance_stable_id_changes_with_turn() {
        let mut u1 = make_test_utterance("test");
        u1.turn_index = 1;
        let mut u2 = make_test_utterance("test");
        u2.turn_index = 2;
        assert_ne!(u1.stable_id(), u2.stable_id());
    }

    #[test]
    fn utterance_stable_id_changes_with_conversation() {
        let mut u1 = make_test_utterance("test");
        u1.conversation_id = "conv-A".into();
        let mut u2 = make_test_utterance("test");
        u2.conversation_id = "conv-B".into();
        assert_ne!(u1.stable_id(), u2.stable_id());
    }

    #[test]
    fn utterance_stable_id_changes_with_role() {
        let mut u1 = make_test_utterance("test");
        u1.role = "user".into();
        let mut u2 = make_test_utterance("test");
        u2.role = "assistant".into();
        assert_ne!(u1.stable_id(), u2.stable_id());
    }

    #[test]
    fn utterance_stable_id_format_is_hex() {
        let u = make_test_utterance("test");
        let id = u.stable_id();
        assert_eq!(id.len(), 64); // SHA256 hex digest
        assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn utterance_serde_roundtrip() {
        let mut metadata = HashMap::new();
        metadata.insert("parser".into(), "jsonl".into());

        let u = Utterance {
            source_path: "/data/test.jsonl".into(),
            source_agent: AgentKind::ClaudeCode,
            conversation_id: "sess-42".into(),
            turn_index: 5,
            role: "user".into(),
            text: "Fix the bug in parser.rs".into(),
            timestamp: Some(chrono::Utc::now()),
            model: Some(ModelInfo {
                provider: "anthropic".into(),
                model_name: "claude-opus-4-20250514".into(),
            }),
            metadata,
        };

        let json = serde_json::to_string(&u).unwrap();
        let back: Utterance = serde_json::from_str(&json).unwrap();

        assert_eq!(u.stable_id(), back.stable_id());
        assert_eq!(u.source_path, back.source_path);
        assert_eq!(u.source_agent, back.source_agent);
        assert_eq!(u.conversation_id, back.conversation_id);
        assert_eq!(u.turn_index, back.turn_index);
        assert_eq!(u.text, back.text);
        assert_eq!(u.model, back.model);
        assert_eq!(u.metadata, back.metadata);
    }

    // --- SourceFile ---

    #[test]
    fn source_file_serde_roundtrip() {
        let sf = SourceFile {
            path: "/home/user/.codex/history.jsonl".into(),
            agent: AgentKind::Codex,
            sha256: "abc123".into(),
            mtime_ns: 1_700_000_000_000_000_000,
            size: 42,
        };
        let json = serde_json::to_string(&sf).unwrap();
        let back: SourceFile = serde_json::from_str(&json).unwrap();
        assert_eq!(sf.path, back.path);
        assert_eq!(sf.agent, back.agent);
        assert_eq!(sf.sha256, back.sha256);
        assert_eq!(sf.mtime_ns, back.mtime_ns);
        assert_eq!(sf.size, back.size);
    }
}
