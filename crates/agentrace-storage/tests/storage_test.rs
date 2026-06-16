// ======================================================================
// Storage integration tests for agentrace.
//
// Tests the SQLite schema creation, utterance insertion, and querying.
// Uses an in-memory database for speed and isolation.
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
        text: text.into(),
        timestamp: None,
        model: None,
        metadata: HashMap::new(),
    }
}

#[test]
fn store_open_creates_tables() {
    let store = Store::open(":memory:").expect("should open in-memory store");
    // If we got here, migration succeeded
    drop(store);
}

#[test]
fn store_replace_source_inserts_utterances() {
    let mut store = Store::open(":memory:").unwrap();
    let source = make_test_source("/data/test.jsonl");
    let utterances = vec![
        make_test_utterance("hello", 1),
        make_test_utterance("world", 2),
    ];

    store.replace_source(&source, &utterances).unwrap();

    // Verify: count of utterances via raw SQL
    let conn = store.conn();
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM utterances", [], |row| row.get(0))
        .unwrap();
    assert_eq!(count, 2);

    let source_count: i64 = conn
        .query_row("SELECT COUNT(*) FROM source_files", [], |row| row.get(0))
        .unwrap();
    assert_eq!(source_count, 1);
}

#[test]
fn store_replace_source_is_idempotent() {
    let mut store = Store::open(":memory:").unwrap();
    let source = make_test_source("/data/test.jsonl");
    let utterances = vec![make_test_utterance("hello", 1)];

    // Insert twice
    store.replace_source(&source, &utterances).unwrap();
    store.replace_source(&source, &utterances).unwrap();

    let conn = store.conn();
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM utterances", [], |row| row.get(0))
        .unwrap();
    assert_eq!(count, 1);
}

#[test]
fn store_replace_source_deletes_old_utterances() {
    let mut store = Store::open(":memory:").unwrap();
    let source = make_test_source("/data/test.jsonl");

    // First insert with 3 utterances
    let first_batch: Vec<Utterance> = (0..3)
        .map(|i| make_test_utterance(&format!("batch1-{}", i), i))
        .collect();
    store.replace_source(&source, &first_batch).unwrap();

    // Replace with 2 different utterances
    let second_batch: Vec<Utterance> = (0..2)
        .map(|i| make_test_utterance(&format!("batch2-{}", i), i))
        .collect();
    store.replace_source(&source, &second_batch).unwrap();

    let conn = store.conn();
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM utterances", [], |row| row.get(0))
        .unwrap();
    assert_eq!(count, 2);

    // Verify the text is from batch2
    let text: String = conn
        .query_row(
            "SELECT text FROM utterances ORDER BY turn_index LIMIT 1",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(text, "batch2-0");
}

#[test]
fn store_utterances_have_correct_schema() {
    let mut store = Store::open(":memory:").unwrap();
    let source = make_test_source("/data/test.jsonl");
    let utterances = vec![make_test_utterance("test message", 0)];

    store.replace_source(&source, &utterances).unwrap();

    let conn = store.conn();
    let (id, source_path, agent, conv_id, turn, text): (String, String, String, String, u32, String) =
        conn.query_row(
            "SELECT id, source_path, source_agent, conversation_id, turn_index, text FROM utterances",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?, row.get(5)?)),
        )
        .unwrap();

    assert!(!id.is_empty());
    assert_eq!(source_path, "/data/test.jsonl");
    assert_eq!(agent, "codex");
    assert_eq!(conv_id, "conv-001");
    assert_eq!(turn, 0);
    assert_eq!(text, "test message");
}

#[test]
fn store_indexes_exist() {
    let store = Store::open(":memory:").unwrap();
    let conn = store.conn();

    // Check that expected indexes exist
    let mut stmt = conn
        .prepare("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='utterances'")
        .unwrap();
    let indexes: Vec<String> = stmt
        .query_map([], |row| row.get(0))
        .unwrap()
        .filter_map(|r| r.ok())
        .collect();

    assert!(indexes.iter().any(|i| i.contains("agent")), "Missing agent index");
    assert!(indexes.iter().any(|i| i.contains("timestamp")), "Missing timestamp index");
    assert!(indexes.iter().any(|i| i.contains("conv")), "Missing conversation index");
}

#[test]
fn store_embeddings_table_exists() {
    let store = Store::open(":memory:").unwrap();
    let conn = store.conn();

    let table_exists: bool = conn
        .query_row(
            "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='embeddings'",
            [],
            |row| row.get(0),
        )
        .unwrap();

    assert!(table_exists, "embeddings table should exist");
}
