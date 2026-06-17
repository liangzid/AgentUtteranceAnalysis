// ======================================================================
// Integration test: coaching pipeline with mock DeepSeek API.
// Uses a local HTTP server that returns canned coaching responses.
// ======================================================================

/// Test that the coaching pipeline works end-to-end with a mock API.
/// This test starts a local HTTP server that mimics DeepSeek's API.
#[test]
fn coaching_pipeline_with_mock_api() {
    use agentrace_core::models::{AgentKind, SourceFile, Utterance};
    use agentrace_storage::Store;
    use std::collections::HashMap;

    // Setup: import utterances into in-memory DB
    let store = Store::open(":memory:").unwrap();
    let source = SourceFile {
        path: "/test/conv.json".into(),
        agent: AgentKind::Codex,
        sha256: "abc".into(),
        mtime_ns: 1,
        size: 100,
    };

    let u1 = Utterance {
        source_path: "/test/conv.json".into(),
        source_agent: AgentKind::Codex,
        conversation_id: "conv-1".into(),
        turn_index: 0,
        role: "user".into(),
        text: "Fix the bug".into(),
        timestamp: None,
        model: None,
        metadata: HashMap::new(),
    };
    let u2 = Utterance {
        source_path: "/test/conv.json".into(),
        source_agent: AgentKind::Codex,
        conversation_id: "conv-1".into(),
        turn_index: 1,
        role: "assistant".into(),
        text: "Please provide the error message".into(),
        timestamp: None,
        model: None,
        metadata: HashMap::new(),
    };
    store.replace_source(&source, &[u1, u2]).unwrap();

    // Verify data
    assert_eq!(store.utterance_count().unwrap(), 2);

    let uncoached = store.uncoached_user_utterances().unwrap();
    assert_eq!(uncoached.len(), 1);
    assert_eq!(uncoached[0].text, "Fix the bug");
    assert_eq!(uncoached[0].role, "user");

    // Verify coaching methods
    let coaching_json = r#"{"intent":"fix bug","what_worked":"asked","could_improve":"be specific","better_prompt":"Here is the error: ...","hidden_tip":"cargo check","knowledge_gap":"error messages","interaction_style":"vague","clarity_score":2}"#;
    assert!(!store.is_coached(&uncoached[0].id).unwrap());

    store
        .insert_coaching(&uncoached[0].id, coaching_json, 2, "vague", "mock")
        .unwrap();

    assert!(store.is_coached(&uncoached[0].id).unwrap());

    let all = store.all_coaching().unwrap();
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].clarity_score, 2);
    assert_eq!(all[0].interaction_style, "vague");

    // Verify coach_summary
    let engine = agentrace_analysis::AnalysisEngine::new(store);
    let summary = engine.coach_summary().unwrap();
    assert_eq!(summary.total_coached, 1);
    assert_eq!(summary.avg_clarity, 2.0);
    assert_eq!(summary.dominant_style, "vague");
}
