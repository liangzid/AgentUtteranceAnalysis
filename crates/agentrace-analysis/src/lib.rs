// ======================================================================
// `AGENTRACE-ANALYSIS`
//
// 1. Analysis engine — the brain of agentrace.
//    Modules: safety, behavior, knowledge, self_improvement, stats.
// 2. Calling chain:
//    agentrace-cli (analyze) → AnalysisEngine::run() → reads from
//    agentrace-storage, uses agentrace-embedding → writes results back.
// 3. Modification history:
//    - 16 June 2025: Initial skeleton
//    - 16 June 2025: Phase 3 — stats analysis + storage integration
//
//     Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
//     Copyright © 2025, Zi Liang, all rights reserved.
//     Created: 16 June 2025
// ======================================================================

pub mod behavior;
pub mod graph;
pub mod knowledge;
pub mod safety;
pub mod self_improvement;
pub mod stats;

use agentrace_embedding::Embedding;
use agentrace_storage::Store;
use anyhow::Result;
use graph::KnowledgeGraph;
use serde::Serialize;
use stats::{AnalysisRow, StatsReport};

/// Consolidated analysis result returned to the CLI / API.
#[derive(Debug, Clone, Serialize)]
pub struct AnalysisResult {
    pub stats: StatsReport,
}

/// Coaching summary generated from LLM analysis of all conversations.
#[derive(Debug, Clone, Serialize)]
pub struct CoachSummary {
    pub total_coached: usize,
    pub avg_clarity: f32,
    pub dominant_style: String,
    pub common_issues: Vec<String>,
    pub top_tips: Vec<String>,
}

/// The central analysis engine.
pub struct AnalysisEngine {
    store: Store,
}

impl AnalysisEngine {
    pub fn new(store: Store) -> Self {
        Self { store }
    }

    /// Generate and store embeddings for all utterances using the given provider.
    pub fn embed_all(
        &self,
        provider: &dyn agentrace_embedding::EmbeddingProvider,
    ) -> Result<usize> {
        let rows = self.store.all_rows()?;
        let texts: Vec<&str> = rows.iter().map(|r| r.text.as_str()).collect();
        if texts.is_empty() {
            return Ok(0);
        }

        let embeddings = provider.embed(&texts)?;
        let mut stored = 0usize;

        for (row, embedding) in rows.iter().zip(embeddings.iter()) {
            self.store.insert_embedding(
                &row.id,
                agentrace_embedding::MODEL_NAME,
                provider.dimension(),
                embedding,
            )?;
            stored += 1;
        }

        Ok(stored)
    }

    /// Build a 3D knowledge graph from stored embeddings.
    /// Uses PCA to reduce 384-dim embeddings → 3D coordinates,
    /// then constructs edges between semantically similar utterances.
    pub fn build_graph(
        &self,
        provider: &dyn agentrace_embedding::EmbeddingProvider,
    ) -> Result<KnowledgeGraph> {
        // Ensure embeddings exist
        let rows = self.store.all_rows()?;
        let existing = self.store.all_embeddings(agentrace_embedding::MODEL_NAME)?;

        let embeddings: Vec<Vec<f32>> = if existing.len() == rows.len() {
            existing.iter().map(|(_, e)| e.clone()).collect()
        } else {
            // Generate missing embeddings
            let texts: Vec<&str> = rows.iter().map(|r| r.text.as_str()).collect();
            let embs = provider.embed(&texts)?;
            for (row, emb) in rows.iter().zip(embs.iter()) {
                self.store.insert_embedding(
                    &row.id,
                    agentrace_embedding::MODEL_NAME,
                    provider.dimension(),
                    emb,
                )?;
            }
            embs
        };

        if embeddings.len() < 3 {
            anyhow::bail!("need at least 3 utterances to build graph, got {}", embeddings.len());
        }

        // PCA: 384-dim → 3D
        let (coords, variance_explained) = graph::pca_reduce(&embeddings, 3)?;

        // Build nodes
        let mut nodes = Vec::new();
        for (i, row) in rows.iter().enumerate() {
            nodes.push(graph::GraphNode {
                utterance_id: row.id.clone(),
                text: row.text.clone(),
                source_agent: row.source_agent.clone(),
                x: coords[[i, 0]],
                y: coords[[i, 1]],
                z: coords[[i, 2]],
            });
        }

        // Build edges: connect nodes with cosine similarity > 0.3
        let edges = graph::build_similarity_edges(&embeddings, &nodes, 0.3, 1000);

        // Store positions
        self.store.clear_graph_positions()?;
        for node in &nodes {
            self.store
                .insert_graph_position(&node.utterance_id, node.x, node.y, node.z)?;
        }

        Ok(KnowledgeGraph {
            nodes,
            edges,
            variance_explained,
        })
    }

    /// Run LLM coaching analysis on all uncoached user utterances.
    /// Uses the DeepSeek API to analyze each conversation turn and stores
    /// per-utterance coaching feedback.
    pub async fn coach_all(
        &self,
        client: &agentrace_llm::DeepSeekClient,
    ) -> Result<usize> {
        let uncoached = self.store.uncoached_user_utterances()?;
        if uncoached.is_empty() {
            tracing::info!("All utterances already coached");
            return Ok(0);
        }

        let mut coached = 0usize;
        for row in &uncoached {
            // Find the next assistant response in the same conversation
            let ai_response = self.find_ai_response(row);

            tracing::info!("Coaching: {} — {}", &row.id[..12.min(row.id.len())], &row.text[..40.min(row.text.len())]);

            let feedback = client
                .coach_conversation(&row.text, ai_response.as_deref(), &row.source_agent)
                .await?;

            let json = serde_json::to_string(&feedback)?;
            let style = match &feedback.interaction_style {
                agentrace_llm::InteractionStyle::Direct => "direct",
                agentrace_llm::InteractionStyle::Exploratory => "exploratory",
                agentrace_llm::InteractionStyle::Helpless => "helpless",
                agentrace_llm::InteractionStyle::Vague => "vague",
                agentrace_llm::InteractionStyle::WellStructured => "well_structured",
            };

            self.store.insert_coaching(
                &row.id,
                &json,
                feedback.clarity_score,
                style,
                "deepseek-chat",
            )?;
            coached += 1;
        }

        Ok(coached)
    }

    /// Find the next assistant utterance after this user utterance in the same conversation.
    fn find_ai_response(&self, user_row: &agentrace_storage::UtteranceRow) -> Option<String> {
        let all = self.store.all_rows().ok()?;
        let mut found_user = false;
        for r in &all {
            if found_user && r.conversation_id == user_row.conversation_id && r.role == "assistant" {
                // Truncate to 300 chars for the prompt
                let truncated = if r.text.len() > 300 {
                    format!("{}…", &r.text[..300])
                } else {
                    r.text.clone()
                };
                return Some(truncated);
            }
            if r.id == user_row.id {
                found_user = true;
            }
        }
        None
    }

    /// Generate a coaching summary from all analyzed utterances.
    pub fn coach_summary(&self) -> Result<CoachSummary> {
        let coaching = self.store.all_coaching()?;
        if coaching.is_empty() {
            return Ok(CoachSummary {
                total_coached: 0,
                avg_clarity: 0.0,
                dominant_style: "none".into(),
                common_issues: vec!["No coaching data yet".into()],
                top_tips: vec![],
            });
        }

        let total = coaching.len();
        let avg_clarity: f32 = coaching.iter().map(|c| c.clarity_score as f32).sum::<f32>() / total as f32;

        // Dominant style
        let mut styles: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for c in &coaching {
            *styles.entry(c.interaction_style.clone()).or_default() += 1;
        }
        let dominant = styles.into_iter().max_by_key(|(_, v)| *v).map(|(k, _)| k).unwrap_or_default();

        // Collect issues and tips from coaching JSON
        let mut issues = Vec::new();
        let mut tips = Vec::new();
        for c in &coaching {
            if let Ok(fb) = serde_json::from_str::<serde_json::Value>(&c.coaching_json) {
                if let Some(s) = fb["could_improve"].as_str() { if !s.is_empty() { issues.push(s.to_string()); } }
                if let Some(s) = fb["hidden_tip"].as_str() { if !s.is_empty() { tips.push(s.to_string()); } }
            }
        }
        issues.truncate(10);
        tips.truncate(10);

        Ok(CoachSummary {
            total_coached: total,
            avg_clarity,
            dominant_style: dominant,
            common_issues: issues,
            top_tips: tips,
        })
    }

    /// Run all analysis modules and return consolidated results.
    pub fn run(&self) -> Result<AnalysisResult> {
        let rows = self.store.all_rows()?;
        let _utterance_count = self.store.utterance_count()?;
        let conversation_count = self.store.conversation_count()?;
        let agent_dist = self.store.agent_distribution()?;
        let month_dist = self.store.month_distribution()?;

        let analysis_rows: Vec<AnalysisRow> = rows
            .into_iter()
            .map(|r| AnalysisRow {
                text: r.text,
                source_agent: r.source_agent,
                source_path: r.source_path,
                timestamp: r.timestamp,
            })
            .collect();

        let report = stats::analyze_stats(
            &analysis_rows,
            conversation_count as usize,
            agent_dist,
            month_dist,
        );

        Ok(AnalysisResult { stats: report })
    }
}
