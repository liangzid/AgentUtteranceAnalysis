// ======================================================================
// `AGENTRACE-DISCOVERY`
//
// 1. Auto-discovery of AI agent conversation stores on disk.
// 2. Called by: agentrace-cli (discover subcommand), agentrace-server (daemon scanner).
// 3. Modification history:
//    - 16 June 2025: Initial skeleton
//
//     Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
//     Copyright © 2025, Zi Liang, all rights reserved.
//     Created: 16 June 2025
// ======================================================================

pub struct DiscoverySummary {
    pub roots: Vec<String>,
    pub files: Vec<String>,
}
