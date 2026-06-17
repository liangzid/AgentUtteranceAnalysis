// ======================================================================
// `AGENTRACE-EMBEDDING`
//
// 1. Text embedding engine for semantic search and clustering.
// 2. EmbeddingProvider trait + StubEmbeddingProvider + CandleEmbeddingProvider.
// 3. Candle backend: pure-Rust BERT, zero C++ deps, works on any glibc.
// 4. Modification history:
//    - 16 June 2025: Initial skeleton
//    - 17 June 2025: Phase 5 — candle-based real inference
//
//     Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
//     Copyright © 2025, Zi Liang, all rights reserved.
//     Created: 16 June 2025
// ======================================================================

use anyhow::Result;

pub mod candle;

pub const EMBEDDING_DIM: usize = 384;
pub const MODEL_NAME: &str = "all-MiniLM-L6-v2";

pub type Embedding = Vec<f32>;

/// Trait abstracting over different embedding backends.
pub trait EmbeddingProvider: Send + Sync {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Embedding>>;
    fn dimension(&self) -> usize;
}

/// Simple stub provider for testing and development.
pub struct StubEmbeddingProvider {
    dimension: usize,
}

impl StubEmbeddingProvider {
    pub fn new() -> Self {
        Self {
            dimension: EMBEDDING_DIM,
        }
    }
}

impl EmbeddingProvider for StubEmbeddingProvider {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Embedding>> {
        Ok(texts
            .iter()
            .map(|_| vec![0.0f32; self.dimension])
            .collect())
    }
    fn dimension(&self) -> usize {
        self.dimension
    }
}

// Re-export candle provider
pub use candle::OnnxEmbeddingProvider;

// ======================================================================
// Tests
// ======================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_dimension() {
        assert_eq!(StubEmbeddingProvider::new().dimension(), 384);
    }

    #[test]
    fn stub_embed_count() {
        let e = StubEmbeddingProvider::new().embed(&["a", "b", "c"]).unwrap();
        assert_eq!(e.len(), 3);
    }

    #[test]
    fn stub_embed_empty() {
        assert!(StubEmbeddingProvider::new().embed(&[]).unwrap().is_empty());
    }
}
