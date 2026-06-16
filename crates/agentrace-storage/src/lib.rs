// ======================================================================
// `AGENTRACE-STORAGE`
//
// 1. SQLite + sqlite-vec persistence layer for utterances, embeddings, and
//    analysis results. Single-file database with vector search support.
// 2. Called by: agentrace-cli (import), agentrace-server (API queries),
//    agentrace-analysis (read/write results).
// 3. Modification history:
//    - 16 June 2025: Initial skeleton
//
//     Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
//     Copyright © 2025, Zi Liang, all rights reserved.
//     Created: 16 June 2025
// ======================================================================

use agentrace_core::models::{SourceFile, Utterance};
use anyhow::Result;

pub struct Store {
    #[allow(dead_code)]
    db_path: String,
    conn: Option<rusqlite::Connection>,
}

impl Store {
    /// Create or open a store at the given path.
    pub fn open(db_path: &str) -> Result<Self> {
        let conn = rusqlite::Connection::open(db_path)?;
        Self::migrate(&conn)?;
        Ok(Self {
            db_path: db_path.to_string(),
            conn: Some(conn),
        })
    }

    /// Return a reference to the underlying SQLite connection.
    pub fn conn(&self) -> &rusqlite::Connection {
        self.conn.as_ref().expect("Store not opened")
    }

    /// KEY REVIEW POINT: Schema design — tables must support all analysis dimensions

    /// Check if a source file on disk matches the stored record.
    /// Returns true if hash, mtime, and size are all unchanged.
    pub fn source_is_current(&self, source: &SourceFile) -> Result<bool> {
        let conn = self.conn();
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
            ",
        )?;
        Ok(())
    }

    pub fn replace_source(&mut self, source: &SourceFile, utterances: &[Utterance]) -> Result<()> {
        let conn = self.conn.as_ref().expect("Store not opened");
        let tx = conn.unchecked_transaction()?;

        // Delete old utterances for this source
        tx.execute(
            "DELETE FROM utterances WHERE source_path = ?1",
            rusqlite::params![source.path],
        )?;
        tx.execute(
            "DELETE FROM source_files WHERE path = ?1",
            rusqlite::params![source.path],
        )?;

        // Insert source record
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

        // Insert utterances
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
    }
}
