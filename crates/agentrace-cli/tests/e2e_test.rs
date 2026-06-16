// ======================================================================
// End-to-end integration tests for the agentrace CLI binary.
//
// Build with: cargo build -p agentrace-cli
// Then run:   cargo test -p agentrace-cli --test e2e_test
// ======================================================================

use std::path::PathBuf;
use std::process::Command;

fn binary_path() -> PathBuf {
    if let Ok(path) = std::env::var("CARGO_BIN_EXE_agentrace_cli") {
        return PathBuf::from(path);
    }
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();
    workspace_root.join("target/debug/agentrace-cli")
}

#[test]
fn e2e_cli_help_runs() {
    let output = Command::new(binary_path())
        .arg("--help")
        .output()
        .expect("binary should run");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("discover"));
    assert!(stdout.contains("import"));
    assert!(stdout.contains("analyze"));
    assert!(stdout.contains("serve"));
}

#[test]
fn e2e_cli_version_runs() {
    let output = Command::new(binary_path())
        .arg("--version")
        .output()
        .expect("binary should run");

    assert!(output.status.success());
}

#[test]
fn e2e_cli_discover_runs() {
    let output = Command::new(binary_path())
        .arg("discover")
        .output()
        .expect("binary should run");

    assert!(output.status.success());
}

#[test]
fn e2e_cli_import_runs() {
    let tmp = std::env::temp_dir().join("agentrace_e2e_test.sqlite");
    let _ = std::fs::remove_file(&tmp);

    let output = Command::new(binary_path())
        .args(["import", "--db", tmp.to_str().unwrap(), "examples/"])
        .output()
        .expect("binary should run");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("imported"));

    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn e2e_cli_analyze_runs() {
    let tmp = std::env::temp_dir().join("agentrace_e2e_analyze.sqlite");
    let _ = std::fs::remove_file(&tmp);

    // First import, then analyze
    Command::new(binary_path())
        .args(["import", "--db", tmp.to_str().unwrap(), "examples/"])
        .output()
        .unwrap();

    let output = Command::new(binary_path())
        .args(["analyze", "--db", tmp.to_str().unwrap()])
        .output()
        .expect("analyze should run");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("stats") || stdout.contains("utterance_count"));

    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn e2e_cli_invalid_command_fails() {
    let output = Command::new(binary_path())
        .arg("nonexistent")
        .output()
        .expect("binary should run");

    assert!(!output.status.success());
}
