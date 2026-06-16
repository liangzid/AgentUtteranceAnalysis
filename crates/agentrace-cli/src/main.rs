// ======================================================================
// `AGENTRACE-CLI`
//
// 1. Command-line interface for agentrace — discover, import, analyze, serve.
// 2. Entry point for all user-facing operations.
// 3. Modification history:
//    - 16 June 2025: Initial skeleton with clap derive
//
//     Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
//     Copyright © 2025, Zi Liang, all rights reserved.
//     Created: 16 June 2025
// ======================================================================

use clap::{Parser, Subcommand};

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
    Discover,
    /// Import utterances from agent logs
    Import {
        /// Paths to scan for conversation files
        paths: Vec<String>,
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
        Commands::Discover => {
            println!("[stub] discover — will scan for agent stores");
        }
        Commands::Import { paths } => {
            println!("[stub] import — will scan {:?}", paths);
        }
        Commands::Analyze => {
            println!("[stub] analyze — will run analysis engine");
        }
        Commands::Serve { port } => {
            println!("[stub] serve — will start dashboard on port {}", port);
        }
    }

    Ok(())
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
        assert!(matches!(cli.command, Commands::Discover));
    }

    #[test]
    fn cli_parse_import_no_paths() {
        let cli = Cli::try_parse_from(["agentrace", "import"]).unwrap();
        assert!(matches!(cli.command, Commands::Import { paths } if paths.is_empty()));
    }

    #[test]
    fn cli_parse_import_with_paths() {
        let cli = Cli::try_parse_from(["agentrace", "import", "/tmp/a.json", "/tmp/b.jsonl"]).unwrap();
        assert!(matches!(
            cli.command,
            Commands::Import { ref paths } if paths == &vec!["/tmp/a.json".to_string(), "/tmp/b.jsonl".to_string()]
        ));
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
        assert!(result.is_err()); // help exits early in clap
    }

    #[test]
    fn cli_parse_version() {
        let result = Cli::try_parse_from(["agentrace", "--version"]);
        assert!(result.is_err()); // version exits early in clap
    }

    #[test]
    fn cli_parse_invalid_command() {
        let result = Cli::try_parse_from(["agentrace", "nonexistent"]);
        assert!(result.is_err());
    }
}
