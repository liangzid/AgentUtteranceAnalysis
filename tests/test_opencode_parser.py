import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from agent_utterance_analysis.discovery import discover_sources
from agent_utterance_analysis.parsers import parse_opencode_sqlite


class OpenCodeParserTests(unittest.TestCase):
    def test_parse_opencode_sqlite_extracts_user_text_parts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "opencode.db"
            create_opencode_db(db_path)
            utterances = parse_opencode_sqlite(db_path)
        self.assertEqual(len(utterances), 1)
        self.assertEqual(utterances[0].source_agent, "opencode")
        self.assertEqual(utterances[0].conversation_id, "ses_1")
        self.assertEqual(utterances[0].text, "Please analyze this.")
        self.assertEqual(utterances[0].metadata["project_worktree"], "/home/me/code/demo")

    def test_discovery_finds_global_opencode_db_and_project_agent_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            global_dir = home / ".local" / "share" / "opencode"
            global_dir.mkdir(parents=True)
            (global_dir / "opencode.db").write_text("", encoding="utf-8")
            project = home / "code" / "demo"
            agent_dir = project / ".opencode"
            agent_dir.mkdir(parents=True)

            summary = discover_sources(home=home, project_globs=[str(home / "code" / "*")])

        self.assertIn(global_dir / "opencode.db", summary.files)
        self.assertIn(agent_dir, summary.roots)


def create_opencode_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE project (
            id text PRIMARY KEY,
            worktree text NOT NULL,
            name text
        );
        CREATE TABLE session (
            id text PRIMARY KEY,
            project_id text NOT NULL,
            title text NOT NULL,
            directory text NOT NULL
        );
        CREATE TABLE message (
            id text PRIMARY KEY,
            session_id text NOT NULL,
            time_created integer NOT NULL,
            time_updated integer NOT NULL,
            data text NOT NULL
        );
        CREATE TABLE part (
            id text PRIMARY KEY,
            message_id text NOT NULL,
            session_id text NOT NULL,
            time_created integer NOT NULL,
            time_updated integer NOT NULL,
            data text NOT NULL
        );
        """
    )
    conn.execute("INSERT INTO project VALUES (?, ?, ?)", ("proj_1", "/home/me/code/demo", "demo"))
    conn.execute("INSERT INTO session VALUES (?, ?, ?, ?)", ("ses_1", "proj_1", "Demo", "/home/me/code/demo"))
    conn.execute(
        "INSERT INTO message VALUES (?, ?, ?, ?, ?)",
        ("msg_1", "ses_1", 1772429204479, 1772429204479, json.dumps({"role": "user"})),
    )
    conn.execute(
        "INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)",
        (
            "part_1",
            "msg_1",
            "ses_1",
            1772429204480,
            1772429204480,
            json.dumps({"type": "text", "text": "Please analyze this."}),
        ),
    )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    unittest.main()
