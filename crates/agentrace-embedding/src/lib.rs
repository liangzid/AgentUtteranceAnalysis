// ======================================================================
// `AGENTRACE-EMBEDDING`
//
// 1. Text embedding engine using ONNX Runtime with all-MiniLM-L6-v2.
//    Produces 384-dimensional embeddings for semantic search and clustering.
// 2. Called by: agentrace-analysis (for clustering/similarity),
//    agentrace-cli (import pipeline), agentrace-server (semantic search API).
// 3. Modification history:
//    - 16 June 2025: Initial skeleton
//
//     Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
//     Copyright © 2025, Zi Liang, all rights reserved.
//     Created: 16 June 2025
// ======================================================================

use anyhow::Result;

pub const EMBEDDING_DIM: usize = 384;

/// A text embedding vector.
pub type Embedding = Vec<f32>;

/// Trait abstracting over different embedding backends.
pub trait EmbeddingProvider: Send + Sync {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Embedding>>;
    fn dimension(&self) -> usize;
}

/// ONNX-based embedding provider using all-MiniLM-L6-v2.
pub struct OnnxEmbeddingProvider {
    // TODO: ort session will go here
    dimension: usize,
}

impl OnnxEmbeddingProvider {
    /// Load the ONNX model from the given path.
    pub fn load(model_path: &str) -> Result<Self> {
        let _ = model_path;
        Ok(Self {
            dimension: EMBEDDING_DIM,
        })
    }
}

impl EmbeddingProvider for OnnxEmbeddingProvider {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Embedding>> {
        // Stub: return zero vectors of correct dimension
        Ok(texts
            .iter()
            .map(|_| vec![0.0f32; self.dimension])
            .collect())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

// ======================================================================
// Tests
// ======================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_provider_dimension_is_384() {
        let provider = OnnxEmbeddingProvider::load("dummy").unwrap();
        assert_eq!(provider.dimension(), 384);
    }

    #[test]
    fn embedding_provider_returns_correct_count() {
        let provider = OnnxEmbeddingProvider::load("dummy").unwrap();
        let texts = ["hello", "world", "test"];
        let embeddings = provider.embed(&texts).unwrap();
        assert_eq!(embeddings.len(), 3);
    }

    #[test]
    fn embedding_provider_returns_correct_dimension_per_vector() {
        let provider = OnnxEmbeddingProvider::load("dummy").unwrap();
        let texts = ["single"];
        let embeddings = provider.embed(&texts).unwrap();
        assert_eq!(embeddings[0].len(), 384);
    }

    #[test]
    fn embedding_provider_empty_input() {
        let provider = OnnxEmbeddingProvider::load("dummy").unwrap();
        let texts: [&str; 0] = [];
        let embeddings = provider.embed(&texts).unwrap();
        assert!(embeddings.is_empty());
    }
}
