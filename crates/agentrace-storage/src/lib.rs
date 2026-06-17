// ======================================================================
// `AGENTRACE-STORAGE`
//
// 1. SQLite + sqlite-vec persistence layer for utterances, embeddings, and
//    analysis results. Single-file database with vector search support.
// 2. Thread-safe via Arc<Mutex<Connection>> — supports Clone for sharing.
// 3. Modification history:
//    - 16 June 2025: Initial skeleton
//    - 16 June 2025: Phase 3 — Arc<Mutex> for Clone + query methods
//
//     Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
//     Copyright © 2025, Zi Liang, all rights reserved.
//     Created: 16 June 2025
// ======================================================================

use agentrace_core::models::{SourceFile, Utterance};
use anyhow::Result;
use serde::Serialize;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct Store {
    db_path: String,
    conn: Arc<Mutex<rusqlite::Connection>>,
}

impl Store {
    /// Create or open a store at the given path.
    pub fn open(db_path: &str) -> Result<Self> {
        let conn = rusqlite::Connection::open(db_path)?;
        Self::migrate(&conn)?;
        Ok(Self {
            db_path: db_path.to_string(),
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// KEY REVIEW POINT: Schema design — tables must support all analysis dimensions
    /// and cross-filtering by agent, model, and task type.
    fn migrate(conn: &rusqlite::Connection) -> Result<()> {
        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS source_files (
                path        TEXT PRIMARY KEY,
                agent       TEXT NOT NULL,
                sha256      TEXT NOT NULL,
                mtime_ns    INTEGER NOT NULL,
                size        INTEGER NOT NULL,
                imported_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS utterances (
                id              TEXT PRIMARY KEY,
                source_path     TEXT NOT NULL,
                source_agent    TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                turn_index      INTEGER NOT NULL,
                timestamp       TEXT,
                model_provider  TEXT,
                model_name      TEXT,
                text            TEXT NOT NULL,
                metadata_json   TEXT DEFAULT '{}',
                imported_at     TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (source_path) REFERENCES source_files(path)
            );

            CREATE INDEX IF NOT EXISTS idx_utterances_agent
                ON utterances(source_agent);
            CREATE INDEX IF NOT EXISTS idx_utterances_timestamp
                ON utterances(timestamp);
            CREATE INDEX IF NOT EXISTS idx_utterances_conv
                ON utterances(conversation_id);

            CREATE TABLE IF NOT EXISTS embeddings (
                utterance_id TEXT PRIMARY KEY,
                model_name   TEXT NOT NULL,
                dimensions   INTEGER NOT NULL,
                vector       BLOB NOT NULL,
                FOREIGN KEY (utterance_id) REFERENCES utterances(id)
            );

            CREATE TABLE IF NOT EXISTS graph_positions (
                utterance_id TEXT PRIMARY KEY,
                x            REAL NOT NULL,
                y            REAL NOT NULL,
                z            REAL NOT NULL,
                FOREIGN KEY (utterance_id) REFERENCES utterances(id)
            );
            ",
        )?;
        Ok(())
    }

    /// Locks the connection and runs a closure.
    fn with_conn<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&rusqlite::Connection) -> Result<T>,
    {
        let guard = self.conn.lock().unwrap();
        f(&guard)
    }

    /// Check if a source file on disk matches the stored record.
    pub fn source_is_current(&self, source: &SourceFile) -> Result<bool> {
        self.with_conn(|conn| {
            let result: Option<(String, i64, i64)> = conn
                .query_row(
                    "SELECT sha256, mtime_ns, size FROM source_files WHERE path = ?1",
                    rusqlite::params![source.path],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                )
                .ok();

            match result {
                Some((stored_hash, stored_mtime, stored_size)) => {
                    Ok(stored_hash == source.sha256
                        && stored_mtime == source.mtime_ns
                        && stored_size == source.size as i64)
                }
                None => Ok(false),
            }
        })
    }

    pub fn replace_source(&self, source: &SourceFile, utterances: &[Utterance]) -> Result<()> {
        self.with_conn(|conn| {
            let tx = conn.unchecked_transaction()?;

            tx.execute(
                "DELETE FROM utterances WHERE source_path = ?1",
                rusqlite::params![source.path],
            )?;
            tx.execute(
                "DELETE FROM source_files WHERE path = ?1",
                rusqlite::params![source.path],
            )?;

            tx.execute(
                "INSERT INTO source_files (path, agent, sha256, mtime_ns, size) VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![
                    source.path,
                    source.agent.to_string(),
                    source.sha256,
                    source.mtime_ns,
                    source.size,
                ],
            )?;

            for u in utterances {
                tx.execute(
                    "INSERT OR REPLACE INTO utterances (id, source_path, source_agent, conversation_id, turn_index, timestamp, model_provider, model_name, text, metadata_json)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                    rusqlite::params![
                        u.stable_id(),
                        u.source_path,
                        u.source_agent.to_string(),
                        u.conversation_id,
                        u.turn_index,
                        u.timestamp.map(|t| t.to_rfc3339()),
                        u.model.as_ref().map(|m| &m.provider),
                        u.model.as_ref().map(|m| &m.model_name),
                        u.text,
                        serde_json::to_string(&u.metadata).unwrap_or_default(),
                    ],
                )?;
            }

            tx.commit()?;
            Ok(())
        })
    }

    pub fn utterance_count(&self) -> Result<i64> {
        self.with_conn(|conn| {
            Ok(conn.query_row("SELECT COUNT(*) FROM utterances", [], |r| r.get(0))?)
        })
    }

    pub fn conversation_count(&self) -> Result<i64> {
        self.with_conn(|conn| {
            Ok(conn.query_row(
                "SELECT COUNT(DISTINCT conversation_id) FROM utterances",
                [],
                |r| r.get(0),
            )?)
        })
    }

    pub fn agent_distribution(&self) -> Result<Vec<(String, i64)>> {
        self.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT source_agent, COUNT(*) as cnt FROM utterances GROUP BY source_agent ORDER BY cnt DESC",
            )?;
            let rows = stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?;
            Ok(rows.filter_map(|r| r.ok()).collect())
        })
    }

    pub fn month_distribution(&self) -> Result<Vec<(String, i64)>> {
        self.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT substr(timestamp, 1, 7) as month, COUNT(*) as cnt
                 FROM utterances WHERE timestamp IS NOT NULL
                 GROUP BY month ORDER BY month",
            )?;
            let rows = stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?;
            let mut results: Vec<(String, i64)> = rows.filter_map(|r| r.ok()).collect();
            let unknown: i64 = conn.query_row(
                "SELECT COUNT(*) FROM utterances WHERE timestamp IS NULL",
                [],
                |r| r.get(0),
            )?;
            if unknown > 0 {
                results.push(("unknown".to_string(), unknown));
            }
            Ok(results)
        })
    }

    pub fn all_texts(&self) -> Result<Vec<String>> {
        self.with_conn(|conn| {
            let mut stmt = conn.prepare("SELECT text FROM utterances")?;
            let rows = stmt.query_map([], |row| row.get(0))?;
            Ok(rows.filter_map(|r| r.ok()).collect())
        })
    }

    pub fn all_rows(&self) -> Result<Vec<UtteranceRow>> {
        self.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT id, source_path, source_agent, conversation_id, turn_index, timestamp, text
                 FROM utterances ORDER BY timestamp",
            )?;
            let rows = stmt.query_map([], |row| {
                Ok(UtteranceRow {
                    id: row.get(0)?,
                    source_path: row.get(1)?,
                    source_agent: row.get(2)?,
                    conversation_id: row.get(3)?,
                    turn_index: row.get(4)?,
                    timestamp: row.get(5)?,
                    text: row.get(6)?,
                })
            })?;
            Ok(rows.filter_map(|r| r.ok()).collect())
        })
    }

    pub fn insert_embedding(
        &self,
        utterance_id: &str,
        model_name: &str,
        dimensions: usize,
        vector: &[f32],
    ) -> Result<()> {
        let bytes: Vec<u8> = vector.iter().flat_map(|f| f.to_le_bytes()).collect();
        self.with_conn(|conn| {
            conn.execute(
                "INSERT OR REPLACE INTO embeddings (utterance_id, model_name, dimensions, vector) VALUES (?1, ?2, ?3, ?4)",
                rusqlite::params![utterance_id, model_name, dimensions as i64, bytes],
            )?;
            Ok(())
        })
    }

    pub fn all_embeddings(&self, model_name: &str) -> Result<Vec<(String, Vec<f32>)>> {
        self.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT utterance_id, dimensions, vector FROM embeddings WHERE model_name = ?1",
            )?;
            let rows = stmt.query_map(rusqlite::params![model_name], |row| {
                let id: String = row.get(0)?;
                let dims: i64 = row.get(1)?;
                let blob: Vec<u8> = row.get(2)?;
                let vector: Vec<f32> = blob
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Ok((id, vector))
            })?;
            Ok(rows.filter_map(|r| r.ok()).collect())
        })
    }

    /// Store a 3D position for a graph node.
    pub fn insert_graph_position(&self, utterance_id: &str, x: f32, y: f32, z: f32) -> Result<()> {
        self.with_conn(|conn| {
            conn.execute(
                "INSERT OR REPLACE INTO graph_positions (utterance_id, x, y, z) VALUES (?1, ?2, ?3, ?4)",
                rusqlite::params![utterance_id, x, y, z],
            )?;
            Ok(())
        })
    }

    /// Retrieve all graph positions with utterance metadata.
    pub fn all_graph_positions(&self) -> Result<Vec<GraphPositionRow>> {
        self.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT g.utterance_id, g.x, g.y, g.z, u.text, u.source_agent
                 FROM graph_positions g
                 JOIN utterances u ON u.id = g.utterance_id
                 ORDER BY g.utterance_id",
            )?;
            let rows = stmt.query_map([], |row| {
                Ok(GraphPositionRow {
                    utterance_id: row.get(0)?,
                    x: row.get(1)?,
                    y: row.get(2)?,
                    z: row.get(3)?,
                    text: row.get(4)?,
                    source_agent: row.get(5)?,
                })
            })?;
            Ok(rows.filter_map(|r| r.ok()).collect())
        })
    }

    /// Clear all graph positions (re-run graph build).
    pub fn clear_graph_positions(&self) -> Result<()> {
        self.with_conn(|conn| {
            conn.execute("DELETE FROM graph_positions", [])?;
            Ok(())
        })
    }
}

/// A row from graph_positions joined with utterances.
#[derive(Debug, Clone, Serialize)]
pub struct GraphPositionRow {
    pub utterance_id: String,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub text: String,
    pub source_agent: String,
}

/// A lightweight row from the utterances table.
#[derive(Debug, Clone)]
pub struct UtteranceRow {
    pub id: String,
    pub source_path: String,
    pub source_agent: String,
    pub conversation_id: String,
    pub turn_index: u32,
    pub timestamp: Option<String>,
    pub text: String,
}

// ======================================================================
// Tests
// ======================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_source(path: &str) -> SourceFile {
        SourceFile {
            path: path.into(),
            agent: agentrace_core::models::AgentKind::Codex,
            sha256: "deadbeef".into(),
            mtime_ns: 1_700_000_000_000_000_000,
            size: 1024,
        }
    }

    fn make_test_utterance(text: &str, turn: u32) -> Utterance {
        Utterance {
            source_path: "/data/test.jsonl".into(),
            source_agent: agentrace_core::models::AgentKind::Codex,
            conversation_id: "conv-001".into(),
            turn_index: turn,
            text: text.into(),
            timestamp: None,
            model: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn store_open_succeeds() {
        let store = Store::open(":memory:").unwrap();
        assert!(store.utterance_count().unwrap() == 0);
    }

    #[test]
    fn replace_and_query() {
        let store = Store::open(":memory:").unwrap();
        let source = make_test_source("/data/test.jsonl");
        let utterances = vec![
            make_test_utterance("hello", 1),
            make_test_utterance("world", 2),
        ];
        store.replace_source(&source, &utterances).unwrap();
        assert_eq!(store.utterance_count().unwrap(), 2);
        assert_eq!(store.conversation_count().unwrap(), 1);
    }

    #[test]
    fn source_is_current_works() {
        let store = Store::open(":memory:").unwrap();
        let source = make_test_source("/data/test.jsonl");
        assert!(!store.source_is_current(&source).unwrap());
        store.replace_source(&source, &[]).unwrap();
        assert!(store.source_is_current(&source).unwrap());
    }

    #[test]
    fn store_is_clone() {
        let store = Store::open(":memory:").unwrap();
        let store2 = store.clone();
        assert_eq!(store.utterance_count().unwrap(), store2.utterance_count().unwrap());
    }
}
