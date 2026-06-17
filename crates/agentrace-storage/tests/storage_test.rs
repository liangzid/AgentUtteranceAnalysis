// ======================================================================
// Storage integration tests for agentrace.
// Tests the SQLite schema creation, utterance insertion, and querying.
// ======================================================================

use agentrace_core::models::{AgentKind, SourceFile, Utterance};
use agentrace_storage::Store;
use std::collections::HashMap;

fn make_test_source(path: &str) -> SourceFile {
    SourceFile {
        path: path.into(),
        agent: AgentKind::Codex,
        sha256: "deadbeef".into(),
        mtime_ns: 1_700_000_000_000_000_000,
        size: 1024,
    }
}

fn make_test_utterance(text: &str, turn: u32) -> Utterance {
    Utterance {
        source_path: "/data/test.jsonl".into(),
        source_agent: AgentKind::Codex,
        conversation_id: "conv-001".into(),
        turn_index: turn,
            role: "user".into(),
        text: text.into(),
        timestamp: None,
        model: None,
        metadata: HashMap::new(),
    }
}

#[test]
fn store_open_creates_tables() {
    let store = Store::open(":memory:").expect("should open in-memory store");
    assert_eq!(store.utterance_count().unwrap(), 0);
}

#[test]
fn store_replace_source_inserts_utterances() {
    let store = Store::open(":memory:").unwrap();
    let source = make_test_source("/data/test.jsonl");
    let utterances = vec![
        make_test_utterance("hello", 1),
        make_test_utterance("world", 2),
    ];

    store.replace_source(&source, &utterances).unwrap();
    assert_eq!(store.utterance_count().unwrap(), 2);
}

#[test]
fn store_replace_source_is_idempotent() {
    let store = Store::open(":memory:").unwrap();
    let source = make_test_source("/data/test.jsonl");
    let utterances = vec![make_test_utterance("hello", 1)];

    store.replace_source(&source, &utterances).unwrap();
    store.replace_source(&source, &utterances).unwrap();
    assert_eq!(store.utterance_count().unwrap(), 1);
}

#[test]
fn store_replace_source_deletes_old_utterances() {
    let store = Store::open(":memory:").unwrap();
    let source = make_test_source("/data/test.jsonl");

    let first_batch: Vec<Utterance> = (0..3)
        .map(|i| make_test_utterance(&format!("batch1-{}", i), i))
        .collect();
    store.replace_source(&source, &first_batch).unwrap();

    let second_batch: Vec<Utterance> = (0..2)
        .map(|i| make_test_utterance(&format!("batch2-{}", i), i))
        .collect();
    store.replace_source(&source, &second_batch).unwrap();

    assert_eq!(store.utterance_count().unwrap(), 2);
}

#[test]
fn store_agent_distribution() {
    let store = Store::open(":memory:").unwrap();
    let source = make_test_source("/data/test.jsonl");
    let utterances = vec![make_test_utterance("test", 0)];
    store.replace_source(&source, &utterances).unwrap();

    let dist = store.agent_distribution().unwrap();
    assert_eq!(dist.len(), 1);
    assert_eq!(dist[0].0, "codex");
    assert_eq!(dist[0].1, 1);
}

#[test]
fn store_conversation_count() {
    let store = Store::open(":memory:").unwrap();

    let source_a = SourceFile {
        path: "/data/a.jsonl".into(),
        agent: AgentKind::Codex,
        sha256: "a".into(),
        mtime_ns: 1,
        size: 1,
    };
    let source_b = SourceFile {
        path: "/data/b.jsonl".into(),
        agent: AgentKind::Codex,
        sha256: "b".into(),
        mtime_ns: 1,
        size: 1,
    };

    let mut u1 = make_test_utterance("a", 0);
    u1.source_path = "/data/a.jsonl".into();
    u1.conversation_id = "conv-A".into();

    let mut u2 = make_test_utterance("b", 0);
    u2.source_path = "/data/b.jsonl".into();
    u2.conversation_id = "conv-B".into();

    store.replace_source(&source_a, &[u1]).unwrap();
    store.replace_source(&source_b, &[u2]).unwrap();

    assert_eq!(store.conversation_count().unwrap(), 2);
}
