// ======================================================================
// `AGENTRACE-SERVER` (binary entrypoint)
//
// 1. Standalone binary entry for the agentrace web server.
// 2. Modification history:
//    - 16 June 2025: Initial skeleton
//
//     Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
//     Copyright © 2025, Zi Liang, all rights reserved.
//     Created: 16 June 2025
// ======================================================================

use agentrace_storage::Store;
use std::net::SocketAddr;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let port: u16 = std::env::var("AGENTRACE_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(3000);

    let db_path = std::env::var("AGENTRACE_DB")
        .unwrap_or_else(|_| "data/agentrace.sqlite".to_string());

    let store = Store::open(&db_path)?;
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    agentrace_server::serve(addr, store).await?;

    Ok(())
}
