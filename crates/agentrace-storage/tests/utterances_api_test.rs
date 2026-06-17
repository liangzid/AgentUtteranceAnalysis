// ======================================================================
// Integration test: verify utterances API data flow end-to-end.
// ======================================================================

use agentrace_core::models::{AgentKind, SourceFile, Utterance};
use agentrace_storage::Store;
use std::collections::HashMap;

#[test]
fn utterances_api_returns_real_data() {
    let store = Store::open(":memory:").unwrap();

    let source = SourceFile {
        path: "/test/conv.json".into(),
        agent: AgentKind::Codex,
        sha256: "abc".into(),
        mtime_ns: 1,
        size: 100,
    };

    let mut u1 = Utterance {
        source_path: "/test/conv.json".into(),
        source_agent: AgentKind::Codex,
        conversation_id: "conv-1".into(),
        turn_index: 0,
            role: "user".into(),
        text: "hello world".into(),
        timestamp: Some(chrono::Utc::now()),
        model: None,
        metadata: HashMap::new(),
    };
    let mut u2 = u1.clone();
    u2.turn_index = 1;
    u2.text = "second message".into();

    store.replace_source(&source, &[u1, u2]).unwrap();

    // Verify counts
    assert_eq!(store.utterance_count().unwrap(), 2);
    assert_eq!(store.conversation_count().unwrap(), 1);

    // Verify all_rows
    let rows = store.all_rows().unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].text, "hello world");
    assert_eq!(rows[0].source_agent, "codex");
    assert_eq!(rows[0].conversation_id, "conv-1");
    assert_eq!(rows[0].turn_index, 0);
    assert_eq!(rows[1].text, "second message");
    assert_eq!(rows[1].turn_index, 1);

    // Verify agent_distribution
    let dist = store.agent_distribution().unwrap();
    assert_eq!(dist.len(), 1);
    assert_eq!(dist[0].0, "codex");
    assert_eq!(dist[0].1, 2);
}

#[test]
fn utterances_api_empty_db_returns_empty() {
    let store = Store::open(":memory:").unwrap();
    assert_eq!(store.utterance_count().unwrap(), 0);
    let rows = store.all_rows().unwrap();
    assert!(rows.is_empty());
}
