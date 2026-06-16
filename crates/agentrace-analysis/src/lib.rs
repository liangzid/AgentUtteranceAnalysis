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
pub mod knowledge;
pub mod safety;
pub mod self_improvement;
pub mod stats;

use agentrace_storage::Store;
use anyhow::Result;
use serde::Serialize;
use stats::{AnalysisRow, StatsReport};

/// Consolidated analysis result returned to the CLI / API.
#[derive(Debug, Clone, Serialize)]
pub struct AnalysisResult {
    pub stats: StatsReport,
}

/// The central analysis engine.
pub struct AnalysisEngine {
    store: Store,
}

impl AnalysisEngine {
    pub fn new(store: Store) -> Self {
        Self { store }
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
