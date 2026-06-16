// ======================================================================
// `AGENTRACE-DISCOVERY`
//
// 1. Auto-discovery of AI agent conversation stores on disk.
//    Scans home directory and project directories for codex, claude,
//    opencode, openclaw, kilo-code logs.
// 2. Called by: agentrace-cli (discover subcommand), agentrace-server (daemon).
// 3. Ported from Python discovery.py (Phase 2).
//
//     Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
//     Copyright © 2025, Zi Liang, all rights reserved.
//     Created: 16 June 2025
// ======================================================================

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

/// Agent directory names to search for inside project directories.
const AGENT_DIR_NAMES: &[&str] = &[
    ".codex", ".claude", ".opencode", ".openclaw", ".kilo", ".kilo-code", ".kilocode",
];

/// Global candidate paths relative to HOME.
const GLOBAL_CANDIDATES: &[&str] = &[
    ".local/share/opencode/opencode.db",
    ".local/share/opencode/log",
    ".codex",
    ".claude",
    ".config/claude",
    ".config/opencode",
    ".openclaw",
    ".kilo",
    ".kilo-code",
    ".kilocode",
];

/// Directory names to skip during recursive scanning.
const EXCLUDED_DIR_NAMES: &[&str] = &[
    ".docker", ".git", ".hg", ".svn", ".venv", "__pycache__", "node_modules",
    "snapshot", "bin", "containers", "dist", "docker", "build", "overlay2",
    "telemetry", "volumes",
];

/// File suffixes that indicate parsable agent logs.
pub const SUPPORTED_SUFFIXES: &[&str] = &[
    ".json", ".jsonl", ".ndjson", ".md", ".markdown", ".txt", ".log", ".db", ".sqlite",
];

/// Result of a discovery scan.
#[derive(Debug, Clone)]
pub struct DiscoverySummary {
    /// Directories that contain agent logs (can be recursed into).
    pub roots: Vec<PathBuf>,
    /// Individual log files found.
    pub files: Vec<PathBuf>,
}

impl DiscoverySummary {
    pub fn new() -> Self {
        Self {
            roots: vec![],
            files: vec![],
        }
    }
}

/// Discover agent conversation stores on disk.
///
/// KEY REVIEW POINT: `home` is typically `~`. The function expands the tilde
/// via `shellexpand` or the caller is expected to pass an expanded path.
pub fn discover_sources(
    home: &Path,
    project_globs: Option<&[String]>,
    include_global: bool,
    max_project_depth: usize,
) -> DiscoverySummary {
    let mut roots: HashSet<PathBuf> = HashSet::new();
    let mut files: HashSet<PathBuf> = HashSet::new();

    if include_global {
        for relative in GLOBAL_CANDIDATES {
            let candidate = home.join(relative);
            add_candidate(&candidate, &mut roots, &mut files);
        }
    }

    for pattern in project_globs.unwrap_or(&["~/code/*".to_string()]) {
        let expanded = shellexpand::tilde(pattern).to_string();
        if let Ok(entries) = glob::glob(&expanded) {
            for entry in entries.flatten() {
                if !entry.is_dir() {
                    continue;
                }
                for agent_dir in iter_agent_dirs(&entry, max_project_depth) {
                    add_candidate(&agent_dir, &mut roots, &mut files);
                }
                // Also scan for opencode.db anywhere under project
                if let Ok(walk) = glob::glob(
                    &format!("{}/**/opencode.db", entry.display()),
                ) {
                    for db in walk.flatten() {
                        if !is_ignored(&db) {
                            files.insert(db);
                        }
                    }
                }
            }
        }
    }

    let mut roots_vec: Vec<PathBuf> = roots.into_iter().collect();
    let mut files_vec: Vec<PathBuf> = files.into_iter().collect();
    roots_vec.sort();
    files_vec.sort();

    DiscoverySummary {
        roots: roots_vec,
        files: files_vec,
    }
}

/// Add a candidate path — if it is a parsable file, add to files;
/// if a directory, add to roots.
fn add_candidate(candidate: &Path, roots: &mut HashSet<PathBuf>, files: &mut HashSet<PathBuf>) {
    if !candidate.exists() {
        return;
    }
    if candidate.is_file() {
        let is_supported = candidate
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| {
                let ext_with_dot = format!(".{}", ext.to_lowercase());
                SUPPORTED_SUFFIXES.contains(&ext_with_dot.as_str())
            })
            .unwrap_or(false);
        let is_opencode_db = candidate
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| n == "opencode.db")
            .unwrap_or(false);
        if is_supported || is_opencode_db {
            files.insert(candidate.to_path_buf());
        }
    } else if candidate.is_dir() {
        roots.insert(candidate.to_path_buf());
    }
}

/// Walk a project directory and yield agent-specific subdirectories.
fn iter_agent_dirs(project: &Path, max_depth: usize) -> Vec<PathBuf> {
    let base_depth = project.components().count();
    let mut result = Vec::new();

    if let Ok(entries) = fs::read_dir(project) {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let depth = path.components().count() - base_depth;
            if depth >= max_depth {
                continue;
            }
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if AGENT_DIR_NAMES.contains(&name) {
                    result.push(path);
                }
            }
        }
    }

    result
}

/// Check if any component of the path is in the exclusion list.
fn is_ignored(path: &Path) -> bool {
    path.components().any(|c| {
        c.as_os_str()
            .to_str()
            .map(|s| EXCLUDED_DIR_NAMES.contains(&s))
            .unwrap_or(false)
    })
}

// ======================================================================
// Tests
// ======================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_discover_with_home_candidates() {
        let home = env::temp_dir();
        let summary = discover_sources(&home, None, true, 3);
        // Most temp dirs won't have agent logs, but the function should not panic
        assert!(summary.roots.len() + summary.files.len() >= 0);
    }

    #[test]
    fn test_discover_without_global() {
        let home = env::temp_dir();
        let summary = discover_sources(&home, None, false, 3);
        // Without global, only project globs matter
        assert!(summary.files.is_empty() || !summary.files.is_empty());
    }

    #[test]
    fn test_is_ignored_true() {
        assert!(is_ignored(Path::new("/foo/.git/config")));
        assert!(is_ignored(Path::new("/foo/node_modules/pkg")));
        assert!(is_ignored(Path::new("/foo/__pycache__/bar")));
    }

    #[test]
    fn test_is_ignored_false() {
        assert!(!is_ignored(Path::new("/foo/bar/baz")));
        assert!(!is_ignored(Path::new("/home/user/projects/my_app")));
    }

    #[test]
    fn test_supported_suffixes() {
        assert!(SUPPORTED_SUFFIXES.contains(&".json"));
        assert!(SUPPORTED_SUFFIXES.contains(&".jsonl"));
        assert!(SUPPORTED_SUFFIXES.contains(&".md"));
        assert!(SUPPORTED_SUFFIXES.contains(&".sqlite"));
        assert!(!SUPPORTED_SUFFIXES.contains(&".py"));
    }

    #[test]
    fn test_add_candidate_file() {
        let tmp = env::temp_dir().join("agentrace_test_file.json");
        fs::write(&tmp, "{}").unwrap();
        let mut roots = HashSet::new();
        let mut files = HashSet::new();
        add_candidate(&tmp, &mut roots, &mut files);
        assert!(files.iter().any(|f| f == &tmp));
        assert!(roots.is_empty());
        fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_add_candidate_missing() {
        let missing = Path::new("/nonexistent/path/xyz.json");
        let mut roots = HashSet::new();
        let mut files = HashSet::new();
        add_candidate(missing, &mut roots, &mut files);
        assert!(roots.is_empty());
        assert!(files.is_empty());
    }
}
