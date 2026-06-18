// ======================================================================
// `API`
//
// 1. REST API route handlers for the agentrace dashboard.
// 2. Modification history:
//    - 16 June 2025: Initial skeleton
//    - 16 June 2025: Phase 3 — real data endpoints
//
//     Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
//     Copyright © 2025, Zi Liang, all rights reserved.
//     Created: 16 June 2025
// ======================================================================

use axum::extract::State;
use axum::response::Json;
use axum::Router;
use serde_json::{Value, json};

use agentrace_storage::Store;
use std::sync::Arc;

/// Shared application state for API handlers.
#[derive(Clone)]
pub struct AppState {
    pub store: Store,
}

pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
}

/// GET /api/v1/stats — basic counts and distributions.
pub async fn get_stats(State(state): State<Arc<AppState>>) -> Json<Value> {
    let utterances = state.store.utterance_count().unwrap_or(0);
    let conversations = state.store.conversation_count().unwrap_or(0);
    let agent_dist = state.store.agent_distribution().unwrap_or_default();
    let month_dist = state.store.month_distribution().unwrap_or_default();

    Json(json!({
        "utterances": utterances,
        "conversations": conversations,
        "agents": agent_dist.into_iter().map(|(k, v)| {
            json!({"agent": k, "count": v})
        }).collect::<Vec<_>>(),
        "months": month_dist.into_iter().map(|(k, v)| {
            json!({"month": k, "count": v})
        }).collect::<Vec<_>>(),
    }))
}

/// GET /api/v1/analysis — full analysis report.
pub async fn get_analysis(State(state): State<Arc<AppState>>) -> Json<Value> {
    let engine = agentrace_analysis::AnalysisEngine::new(state.store.clone());
    match engine.run() {
        Ok(result) => Json(serde_json::to_value(result).unwrap_or(json!({"error": "serialization failed"}))),
        Err(e) => Json(json!({"error": e.to_string()})),
    }
}

/// GET /api/v1/utterances — list of utterances with details.
pub async fn get_utterances(State(state): State<Arc<AppState>>) -> Json<Value> {
    let rows = state.store.all_rows().unwrap_or_default();
    let utterances: Vec<Value> = rows
        .into_iter()
        .map(|r| {
            json!({
                "id": r.id,
                "source_path": r.source_path,
                "source_agent": r.source_agent,
                "conversation_id": r.conversation_id,
                "turn_index": r.turn_index,
                "timestamp": r.timestamp,
                "text": r.text,
            })
        })
        .collect();
    Json(json!({ "utterances": utterances }))
}

/// GET /api/v1/graph — 3D knowledge graph data (nodes + similarity edges).
pub async fn get_graph(State(state): State<Arc<AppState>>) -> Json<Value> {
    let positions = state.store.all_graph_positions().unwrap_or_default();
    let edges = state.store.all_graph_edges().unwrap_or_default();
    let nodes: Vec<Value> = positions
        .into_iter()
        .map(|p| {
            json!({
                "id": p.utterance_id,
                "text": p.text,
                "source_agent": p.source_agent,
                "x": p.x,
                "y": p.y,
                "z": p.z,
                "cluster_id": p.cluster_id,
            })
        })
        .collect();
    let edge_list: Vec<Value> = edges
        .into_iter()
        .map(|e| {
            json!({
                "source": e.source,
                "target": e.target,
                "similarity": e.similarity,
            })
        })
        .collect();
    Json(json!({ "nodes": nodes, "edges": edge_list }))
}

/// GET /api/v1/coaching — LLM coaching results.
pub async fn get_coaching(State(state): State<Arc<AppState>>) -> Json<Value> {
    let rows = state.store.all_coaching().unwrap_or_default();
    let items: Vec<Value> = rows
        .into_iter()
        .map(|r| {
            let fb: serde_json::Value =
                serde_json::from_str(&r.coaching_json).unwrap_or_default();
            json!({
                "utterance_id": r.utterance_id,
                "text": r.text,
                "source_agent": r.source_agent,
                "clarity_score": r.clarity_score,
                "interaction_style": r.interaction_style,
                "feedback": fb,
            })
        })
        .collect();
    Json(json!({ "coaching": items }))
}

/// GET /api/v1/coaching/summary — aggregate coaching summary.
pub async fn get_coaching_summary(State(state): State<Arc<AppState>>) -> Json<Value> {
    let engine = agentrace_analysis::AnalysisEngine::new(state.store.clone());
    match engine.coach_summary() {
        Ok(s) => Json(serde_json::to_value(s).unwrap_or_default()),
        Err(e) => Json(json!({"error": e.to_string()})),
    }
}
