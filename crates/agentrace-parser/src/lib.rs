// ======================================================================
// `AGENTRACE-PARSER`
//
// 1. Multi-format parser: extracts user utterances from JSON, JSONL, SQLite,
//    Markdown, and plain text agent conversation logs.
// 2. Called by: agentrace-cli (import subcommand), agentrace-server (daemon).
// 3. Modification history:
//    - 16 June 2025: Initial skeleton
//
//     Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
//     Copyright © 2025, Zi Liang, all rights reserved.
//     Created: 16 June 2025
// ======================================================================

use agentrace_core::models::Utterance;

pub fn parse_file(path: &str) -> anyhow::Result<Vec<Utterance>> {
    let _ = path;
    Ok(vec![])
}
