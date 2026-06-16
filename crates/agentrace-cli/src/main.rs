// ======================================================================
// `AGENTRACE-CLI`
//
// 1. Command-line interface for agentrace — discover, import, analyze, serve.
// 2. Entry point for all user-facing operations.
// 3. Modification history:
//    - 16 June 2025: Initial skeleton
//    - 16 June 2025: Phase 2 — wired import with discovery + parser + storage
//
//     Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
//     Copyright © 2025, Zi Liang, all rights reserved.
//     Created: 16 June 2025
// ======================================================================

use agentrace_core::models::{AgentKind, SourceFile};
use agentrace_discovery::{discover_sources, SUPPORTED_SUFFIXES};
use agentrace_parser::parse_file;
use agentrace_storage::Store;
use clap::{Parser, Subcommand};
use sha2::{Digest, Sha256};
use std::path::Path;
use walkdir::WalkDir;

#[derive(Parser)]
#[command(name = "agentrace")]
#[command(about = "Track, analyze, and visualize your AI agent interactions")]
#[command(version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,

    /// Path to the SQLite database
    #[arg(long, global = true, default_value = "data/agentrace.sqlite")]
    pub db: String,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Discover agent conversation stores on disk
    Discover {
        /// Home directory for global candidates
        #[arg(long, default_value = "~")]
        home: String,
    },
    /// Import utterances from agent logs
    Import {
        /// Paths to scan for conversation files
        paths: Vec<String>,
        /// Force re-import of unchanged files
        #[arg(long)]
        force: bool,
    },
    /// Run analysis on imported utterances
    Analyze,
    /// Start the web dashboard server
    Serve {
        #[arg(long, default_value = "3000")]
        port: u16,
    },
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Discover { home } => run_discover(&home),
        Commands::Import { paths, force } => run_import(&cli.db, &paths, force),
        Commands::Analyze => {
            println!("[stub] analyze — will run analysis engine");
            Ok(())
        }
        Commands::Serve { port } => {
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(async {
                let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
                agentrace_server::serve(addr).await
            })?;
            Ok(())
        }
    }
}

fn run_discover(home: &str) -> anyhow::Result<()> {
    let home_path = shellexpand::tilde(home).to_string();
    let home = Path::new(&home_path);
    let summary = discover_sources(home, None, true, 3);

    println!("Roots ({}):", summary.roots.len());
    for root in &summary.roots {
        println!("  {}", root.display());
    }
    println!("\nFiles ({}):", summary.files.len());
    for file in &summary.files {
        println!("  {}", file.display());
    }

    Ok(())
}

fn run_import(db_path: &str, paths: &[String], force: bool) -> anyhow::Result<()> {
    let mut store = Store::open(db_path)?;
    let mut scanned = 0u64;
    let mut imported = 0u64;
    let mut skipped = 0u64;
    let mut failed = 0u64;

    for raw_path in paths {
        let expanded = shellexpand::tilde(raw_path).to_string();
        let path = Path::new(&expanded);

        if path.is_file() {
            scanned += 1;
            match import_file(&mut store, path, force) {
                Ok(count) => {
                    imported += count;
                    println!("  imported {} utterances from {}", count, path.display());
                }
                Err(e) => {
                    failed += 1;
                    eprintln!("  error {}: {}", path.display(), e);
                }
            }
        } else if path.is_dir() {
            for entry in WalkDir::new(path)
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().is_file())
            {
                let file_path = entry.path();
                let is_supported = file_path
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(|ext| {
                        let ext_dot = format!(".{}", ext.to_lowercase());
                        SUPPORTED_SUFFIXES.contains(&ext_dot.as_str())
                    })
                    .unwrap_or(false);
                let is_opencode_db = file_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n == "opencode.db")
                    .unwrap_or(false);

                if !is_supported && !is_opencode_db {
                    continue;
                }

                scanned += 1;
                match import_file(&mut store, file_path, force) {
                    Ok(count) => {
                        imported += count;
                        if count > 0 {
                            println!("  imported {} utterances from {}", count, file_path.display());
                        } else {
                            skipped += 1;
                        }
                    }
                    Err(e) => {
                        failed += 1;
                        eprintln!("  error {}: {}", file_path.display(), e);
                    }
                }
            }
        } else {
            eprintln!("  path not found: {}", expanded);
        }
    }

    println!(
        "\nImport summary: {} scanned, {} imported, {} skipped, {} failed",
        scanned, imported, skipped, failed
    );
    Ok(())
}

/// Import a single file: compute hash, check if changed, parse, and store.
fn import_file(store: &mut Store, path: &Path, force: bool) -> anyhow::Result<u64> {
    // Compute SHA256 of file content
    let content = std::fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(&content);
    let hash = format!("{:x}", hasher.finalize());

    // Get file metadata
    let metadata = std::fs::metadata(path)?;
    let mtime_ns = metadata
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos() as i64)
        .unwrap_or(0);
    let size = metadata.len();

    // Detect agent
    let agent = AgentKind::from_path_hint(&path.to_string_lossy());

    let source = SourceFile {
        path: path.to_string_lossy().to_string(),
        agent: agent.clone(),
        sha256: hash.clone(),
        mtime_ns,
        size,
    };

    // Skip unchanged files (unless forced)
    if !force && store.source_is_current(&source)? {
        return Ok(0);
    }

    // Parse utterances
    let utterances = match parse_file(path, &agent) {
        Ok(u) => u,
        Err(e) => {
            anyhow::bail!("parse error: {}", e);
        }
    };

    if utterances.is_empty() {
        return Ok(0);
    }

    let count = utterances.len() as u64;
    store.replace_source(&source, &utterances)?;

    Ok(count)
}

// ======================================================================
// Tests
// ======================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn cli_parse_discover() {
        let cli = Cli::try_parse_from(["agentrace", "discover"]).unwrap();
        assert!(matches!(cli.command, Commands::Discover { .. }));
    }

    #[test]
    fn cli_parse_import_no_paths() {
        let cli = Cli::try_parse_from(["agentrace", "import"]).unwrap();
        assert!(matches!(cli.command, Commands::Import { ref paths, .. } if paths.is_empty()));
    }

    #[test]
    fn cli_parse_import_with_paths() {
        let cli = Cli::try_parse_from(["agentrace", "import", "/tmp/a.json", "/tmp/b.jsonl"]).unwrap();
        assert!(matches!(
            cli.command,
            Commands::Import { ref paths, .. } if paths == &vec!["/tmp/a.json".to_string(), "/tmp/b.jsonl".to_string()]
        ));
    }

    #[test]
    fn cli_parse_import_force() {
        let cli = Cli::try_parse_from(["agentrace", "import", "--force", "/tmp/a.json"]).unwrap();
        assert!(matches!(cli.command, Commands::Import { force: true, .. }));
    }

    #[test]
    fn cli_parse_analyze() {
        let cli = Cli::try_parse_from(["agentrace", "analyze"]).unwrap();
        assert!(matches!(cli.command, Commands::Analyze));
    }

    #[test]
    fn cli_parse_serve_default_port() {
        let cli = Cli::try_parse_from(["agentrace", "serve"]).unwrap();
        assert!(matches!(cli.command, Commands::Serve { port: 3000 }));
    }

    #[test]
    fn cli_parse_serve_custom_port() {
        let cli = Cli::try_parse_from(["agentrace", "serve", "--port", "8080"]).unwrap();
        assert!(matches!(cli.command, Commands::Serve { port: 8080 }));
    }

    #[test]
    fn cli_parse_with_custom_db() {
        let cli = Cli::try_parse_from([
            "agentrace",
            "--db",
            "/custom/path.sqlite",
            "import",
        ])
        .unwrap();
        assert_eq!(cli.db, "/custom/path.sqlite");
    }

    #[test]
    fn cli_parse_default_db() {
        let cli = Cli::try_parse_from(["agentrace", "discover"]).unwrap();
        assert_eq!(cli.db, "data/agentrace.sqlite");
    }

    #[test]
    fn cli_parse_help() {
        let result = Cli::try_parse_from(["agentrace", "--help"]);
        assert!(result.is_err());
    }

    #[test]
    fn cli_parse_version() {
        let result = Cli::try_parse_from(["agentrace", "--version"]);
        assert!(result.is_err());
    }

    #[test]
    fn cli_parse_invalid_command() {
        let result = Cli::try_parse_from(["agentrace", "nonexistent"]);
        assert!(result.is_err());
    }
}
