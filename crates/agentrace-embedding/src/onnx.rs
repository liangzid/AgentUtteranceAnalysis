// ======================================================================
// ONNX EMBEDDING PROVIDER (behind "onnx" feature)
//
// Real ONNX inference with all-MiniLM-L6-v2 via ort.
// Downloads model from HuggingFace Hub on first use.
// ======================================================================

use crate::{Embedding, EmbeddingProvider, EMBEDDING_DIM, MODEL_NAME};
use anyhow::Result;
use ort::ep::CPUExecutionProvider;
use ort::session::Session;
use ort::value::Tensor;
use std::path::PathBuf;
use std::sync::Mutex;
use tokenizers::Tokenizer;

/// Real ONNX-based embedding provider.
pub struct OnnxEmbeddingProvider {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    dimension: usize,
}

impl OnnxEmbeddingProvider {
    pub fn load() -> Result<Self> {
        let model_dir = ensure_model_downloaded()?;
        let model_path = model_dir.join("model.onnx");
        let tokenizer_path = model_dir.join("tokenizer.json");

        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_execution_providers([CPUExecutionProvider::default().build()])
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .commit_from_file(model_path)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        let tokenizer =
            Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow::anyhow!("{e}"))?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            dimension: EMBEDDING_DIM,
        })
    }

    fn infer_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>> {
        let batch_size = texts.len();
        if batch_size == 0 {
            return Ok(vec![]);
        }

        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        let max_len = encodings.iter().map(|e| e.len()).max().unwrap_or(0);
        let mut input_ids = vec![0i64; batch_size * max_len];
        let mut attention_mask = vec![0i64; batch_size * max_len];

        for (i, enc) in encodings.iter().enumerate() {
            for (j, &id) in enc.get_ids().iter().enumerate() {
                input_ids[i * max_len + j] = id as i64;
            }
            for (j, &m) in enc.get_attention_mask().iter().enumerate() {
                attention_mask[i * max_len + j] = m as i64;
            }
        }

        let ids_tensor = Tensor::from_array(([batch_size, max_len], input_ids))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let mask_tensor =
            Tensor::from_array(([batch_size, max_len], attention_mask.clone()))
                .map_err(|e| anyhow::anyhow!("{e}"))?;

        let mut session = self.session.lock().unwrap();
        let outputs = session
            .run(ort::inputs![
                "input_ids" => ids_tensor,
                "attention_mask" => mask_tensor,
            ])
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        let (shape, data) = outputs["token_embeddings"]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        let hidden_dim = shape.get(2).copied().unwrap_or(EMBEDDING_DIM as i64) as usize;

        let mut embeddings = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let mut pooled = vec![0.0f32; hidden_dim];
            let mut count = 0f32;
            for j in 0..max_len {
                if attention_mask[i * max_len + j] == 1 {
                    let offset = (i * max_len + j) * hidden_dim;
                    for k in 0..hidden_dim {
                        pooled[k] += data[offset + k];
                    }
                    count += 1.0;
                }
            }
            let norm: f32 = pooled.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 && count > 0.0 {
                for v in &mut pooled {
                    *v = (*v / count) / norm;
                }
            }
            embeddings.push(pooled);
        }
        Ok(embeddings)
    }
}

impl EmbeddingProvider for OnnxEmbeddingProvider {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Embedding>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        let mut results = Vec::new();
        for chunk in texts.chunks(32) {
            results.extend(self.infer_batch(chunk)?);
        }
        Ok(results)
    }
    fn dimension(&self) -> usize {
        self.dimension
    }
}

// --- Model download --------------------------------------------------

const HF_BASE: &str =
    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main";

fn ensure_model_downloaded() -> Result<PathBuf> {
    let cache_dir = model_cache_dir();
    let model_path = cache_dir.join("model.onnx");
    let tokenizer_path = cache_dir.join("tokenizer.json");

    if model_path.exists() && tokenizer_path.exists() {
        return Ok(cache_dir);
    }

    std::fs::create_dir_all(&cache_dir)?;
    tracing::info!("Downloading all-MiniLM-L6-v2 from HuggingFace…");

    download_file(&format!("{HF_BASE}/onnx/model.onnx"), &model_path)?;
    download_file(&format!("{HF_BASE}/tokenizer.json"), &tokenizer_path)?;

    tracing::info!("Model cached at {}", cache_dir.display());
    Ok(cache_dir)
}

fn model_cache_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("agentrace")
        .join("models")
        .join(MODEL_NAME)
}

fn download_file(url: &str, dest: &std::path::Path) -> Result<()> {
    let resp = reqwest::blocking::get(url)?;
    if !resp.status().is_success() {
        anyhow::bail!("HTTP {} downloading {url}", resp.status());
    }
    let bytes = resp.bytes()?;
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(dest, &bytes)?;
    Ok(())
}
