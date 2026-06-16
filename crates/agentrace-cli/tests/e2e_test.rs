// ======================================================================
// End-to-end integration tests for the agentrace CLI binary.
//
// These tests run the actual compiled binary and verify output.
// Build with: cargo build -p agentrace-cli
// Then run:   cargo test -p agentrace-cli --test e2e_test
// ======================================================================

use std::path::PathBuf;
use std::process::Command;

fn binary_path() -> PathBuf {
    // CARGO_BIN_EXE_agentrace_cli is set by cargo when running integration tests
    if let Ok(path) = std::env::var("CARGO_BIN_EXE_agentrace_cli") {
        return PathBuf::from(path);
    }
    // Fallback: look relative to the OUT_DIR or CARGO_MANIFEST_DIR
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
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("[stub] discover"));
}

#[test]
fn e2e_cli_import_runs() {
    let output = Command::new(binary_path())
        .args(["import", "/tmp"])
        .output()
        .expect("binary should run");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("[stub] import"));
}

#[test]
fn e2e_cli_analyze_runs() {
    let output = Command::new(binary_path())
        .arg("analyze")
        .output()
        .expect("binary should run");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("[stub] analyze"));
}

#[test]
fn e2e_cli_serve_stub_runs() {
    let output = Command::new(binary_path())
        .arg("serve")
        .output()
        .expect("binary should run");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("[stub] serve"));
}

#[test]
fn e2e_cli_invalid_command_fails() {
    let output = Command::new(binary_path())
        .arg("nonexistent")
        .output()
        .expect("binary should run");

    assert!(!output.status.success());
}
