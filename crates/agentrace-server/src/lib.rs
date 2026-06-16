// ======================================================================
// `AGENTRACE-SERVER` (library)
//
// 1. Re-exports the router builder and serve function for embedding in the CLI.
// 2. Modification history:
//    - 16 June 2025: Initial skeleton
//
//     Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
//     Copyright © 2025, Zi Liang, all rights reserved.
//     Created: 16 June 2025
// ======================================================================

mod router;
pub mod api;

pub use router::build_router;
pub use router::serve;
