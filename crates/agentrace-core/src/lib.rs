// ======================================================================
// `AGENTRACE-CORE`
//
// 1. Core data models shared across all agentrace crates:
//    Utterance, SourceFile, Conversation, AgentKind, ModelKind, etc.
// 2. Calling chain: All other crates depend on this one as the type foundation.
// 3. Modification history:
//    - 16 June 2025: Initial creation — workspace skeleton for agentrace
//
//     Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
//     Copyright © 2025, Zi Liang, all rights reserved.
//     Created: 16 June 2025
// ======================================================================

pub mod models;
