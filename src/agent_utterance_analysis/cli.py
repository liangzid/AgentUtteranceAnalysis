from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import analyze_rows
from .discovery import discover_sources
from .export import export_rows
from .importer import import_paths
from .storage import Store
from .ui import TerminalUI


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.ui = TerminalUI(theme=args.theme, plain=args.plain)
    if args.command == "import":
        return command_import(args)
    if args.command == "discover":
        return command_discover(args)
    if args.command == "export":
        return command_export(args)
    if args.command == "analyze":
        return command_analyze(args)
    if args.command == "run":
        return command_run(args)
    parser.print_help()
    return 2


def build_parser() -> argparse.ArgumentParser:
    db_parent = argparse.ArgumentParser(add_help=False)
    db_parent.add_argument("--db", default="data/utterances.sqlite", help="SQLite database path.")
    db_parent.add_argument("--theme", choices=["dark", "light", "mono"], default="dark", help="Terminal color theme.")
    db_parent.add_argument("--plain", action="store_true", help="Disable colors, tables, and spinner animations.")

    parser = argparse.ArgumentParser(
        prog="aua",
        description="Import, export, and analyze AI-agent dialogue utterances.",
        parents=[db_parent],
    )
    subparsers = parser.add_subparsers(dest="command")

    import_parser = subparsers.add_parser(
        "import", parents=[db_parent], help="Import conversation files incrementally."
    )
    add_discovery_args(import_parser)
    import_parser.add_argument("paths", nargs="*", help="Files or directories to scan. Defaults to auto-discovery.")
    import_parser.add_argument("--force", action="store_true", help="Re-import unchanged files.")

    discover_parser = subparsers.add_parser(
        "discover",
        parents=[db_parent],
        help="Find likely AI-agent dialogue stores without importing them.",
    )
    add_discovery_args(discover_parser)

    export_parser = subparsers.add_parser("export", parents=[db_parent], help="Export normalized utterances.")
    export_parser.add_argument("--format", choices=["jsonl", "csv", "markdown"], default="markdown")
    export_parser.add_argument("--output", required=True, help="Output file path.")

    analyze_parser = subparsers.add_parser("analyze", parents=[db_parent], help="Analyze normalized utterances.")
    analyze_parser.add_argument("--output", help="Optional Markdown report output path.")

    run_parser = subparsers.add_parser("run", parents=[db_parent], help="Import, export, and analyze in one command.")
    add_discovery_args(run_parser)
    run_parser.add_argument("paths", nargs="*", help="Files or directories to scan. Defaults to auto-discovery.")
    run_parser.add_argument("--force", action="store_true", help="Re-import unchanged files.")
    run_parser.add_argument("--export", default="exports/utterances.md", help="Markdown export path.")
    run_parser.add_argument("--report", default="reports/analysis.md", help="Analysis report path.")
    return parser


def add_discovery_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--home", default="~", help="Home directory used for auto-discovery.")
    parser.add_argument(
        "--project-glob",
        action="append",
        nargs="+",
        help="Project glob to search, for example '~/code/*'. Can be repeated.",
    )
    parser.add_argument(
        "--no-global",
        action="store_true",
        help="Skip global locations such as ~/.local/share/opencode.",
    )
    parser.add_argument(
        "--max-project-depth",
        type=int,
        default=3,
        help="Maximum depth below each project for hidden agent directories.",
    )
    parser.add_argument(
        "--english-only",
        action="store_true",
        help="Import only English-dominant user utterances.",
    )


def command_import(args: argparse.Namespace) -> int:
    ui: TerminalUI = args.ui
    with ui.status("Resolving dialogue sources"):
        paths = resolve_input_paths(args)
    with ui.status("Importing utterances incrementally"):
        with Store(args.db) as store:
            summary = import_paths(paths, store, force=args.force, english_only=args.english_only)
            counts = store.counts()
    ui.summary(
        "Import complete",
        {
            "scanned": summary.scanned_files,
            "imported": summary.imported_files,
            "current": summary.skipped_current_files,
            "failed": summary.failed_files,
            "new utterances": summary.utterances,
            "total utterances": counts["utterances"],
        },
        failed=summary.failed_files,
    )
    return 0 if summary.failed_files == 0 else 1


def command_discover(args: argparse.Namespace) -> int:
    ui: TerminalUI = args.ui
    with ui.status("Searching global and project-local agent stores"):
        summary = discover_sources(
            home=args.home,
            project_globs=flatten_project_globs(args.project_glob),
            include_global=not args.no_global,
            max_project_depth=args.max_project_depth,
        )
    ui.discovery(summary.files, summary.roots)
    return 0


def command_export(args: argparse.Namespace) -> int:
    ui: TerminalUI = args.ui
    with ui.status(f"Exporting utterances as {args.format}"):
        with Store(args.db) as store:
            count = export_rows(store.iter_utterances(), args.output, args.format)
    ui.summary("Export complete", {"utterances": count, "output": Path(args.output)})
    return 0


def command_analyze(args: argparse.Namespace) -> int:
    ui: TerminalUI = args.ui
    with ui.status("Analyzing utterance distribution and English naturalness"):
        with Store(args.db) as store:
            report = analyze_rows(store.iter_utterances(), args.output)
    if args.output:
        ui.summary("Analysis complete", {"report": Path(args.output)})
    else:
        ui.raw(report)
    return 0


def command_run(args: argparse.Namespace) -> int:
    ui: TerminalUI = args.ui
    with ui.status("Resolving dialogue sources"):
        paths = resolve_input_paths(args)
    with Store(args.db) as store:
        with ui.status("Importing utterances incrementally"):
            summary = import_paths(paths, store, force=args.force, english_only=args.english_only)
        with ui.status("Writing Markdown export"):
            export_count = export_rows(store.iter_utterances(), args.export, "markdown")
        with ui.status("Writing analysis report"):
            analyze_rows(store.iter_utterances(), args.report)
    ui.summary(
        "Run complete",
        {
            "scanned": summary.scanned_files,
            "imported": summary.imported_files,
            "failed": summary.failed_files,
            "exported": export_count,
            "export": Path(args.export),
            "report": Path(args.report),
        },
        failed=summary.failed_files,
    )
    return 0 if summary.failed_files == 0 else 1


def resolve_input_paths(args: argparse.Namespace) -> list[Path]:
    explicit_paths = [Path(path).expanduser() for path in getattr(args, "paths", [])]
    if explicit_paths:
        return explicit_paths
    summary = discover_sources(
        home=args.home,
        project_globs=flatten_project_globs(args.project_glob),
        include_global=not args.no_global,
        max_project_depth=args.max_project_depth,
    )
    return list(summary.files) + list(summary.roots)


def flatten_project_globs(values: list[list[str]] | None) -> list[str] | None:
    if not values:
        return None
    return [item for group in values for item in group]


if __name__ == "__main__":
    raise SystemExit(main())
