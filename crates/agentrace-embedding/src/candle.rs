// ======================================================================
// CANDLE EMBEDDING PROVIDER
//
// Pure-Rust BERT embedding via HuggingFace candle framework.
// Uses all-MiniLM-L6-v2 model, downloaded from HuggingFace Hub on first use.
// Zero C++ dependencies — works on any glibc version.
// ======================================================================

use crate::{Embedding, EmbeddingProvider, EMBEDDING_DIM};
use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use hf_hub::api::sync::Api;
use std::path::PathBuf;
use tokenizers::Tokenizer;

pub struct OnnxEmbeddingProvider {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    dimension: usize,
}

// Safety: BertModel + Tokenizer are both Send + Sync
unsafe impl Send for OnnxEmbeddingProvider {}
unsafe impl Sync for OnnxEmbeddingProvider {}

impl OnnxEmbeddingProvider {
    /// Load the model, downloading from HuggingFace Hub if needed (~90MB).
    pub fn load() -> Result<Self> {
        let device = Device::Cpu;
        let model_dir = ensure_model_downloaded()?;

        // Load tokenizer
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer =
            Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow::anyhow!("{e}"))?;

        // Load BERT config
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)?;
        let config: Config = serde_json::from_str(&config_str)?;

        // Load model weights
        let model_path = model_dir.join("model.safetensors");
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], candle_core::DType::F32, &device)? };
        let model = BertModel::load(vb, &config)?;

        Ok(Self {
            model,
            tokenizer,
            device,
            dimension: EMBEDDING_DIM,
        })
    }

    fn infer_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>> {
        let batch_size = texts.len();
        if batch_size == 0 {
            return Ok(vec![]);
        }

        // Tokenize
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        let max_len = encodings.iter().map(|e| e.len()).max().unwrap_or(0);
        let mut input_ids = vec![0u32; batch_size * max_len];
        let mut attention_mask = vec![0u32; batch_size * max_len];
        let mut token_type_ids = vec![0u32; batch_size * max_len];

        for (i, enc) in encodings.iter().enumerate() {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            let type_ids = enc.get_type_ids();
            for (j, &id) in ids.iter().enumerate() {
                input_ids[i * max_len + j] = id;
            }
            for (j, &m) in mask.iter().enumerate() {
                attention_mask[i * max_len + j] = m;
            }
            for (j, &t) in type_ids.iter().enumerate() {
                token_type_ids[i * max_len + j] = t;
            }
        }

        // Convert to candle tensors
        let ids_tensor = Tensor::from_vec(input_ids, (batch_size, max_len), &self.device)?;
        let mask_tensor = Tensor::from_vec(attention_mask, (batch_size, max_len), &self.device)?;
        let ttype_tensor = Tensor::from_vec(token_type_ids, (batch_size, max_len), &self.device)?;

        // Run BERT forward
        let output = self.model.forward(
            &ids_tensor,
            &ttype_tensor,
            Some(&mask_tensor),
        )?;

        // Mean pool across sequence (accounting for attention mask)
        // output shape: [batch, seq, hidden]
        let hidden_dim = output.dims()[2];
        let mask_f32 = mask_tensor.to_dtype(candle_core::DType::F32)?;
        let mask_expanded = mask_f32.unsqueeze(2)? // [batch, seq, 1]
            .broadcast_as((batch_size, max_len, hidden_dim))?;

        let masked_output = output.mul(&mask_expanded)?;
        let summed = masked_output.sum(1)?; // [batch, hidden]
        let counts = mask_f32.sum(1)?; // [batch]
        let counts_unsqueezed = counts.unsqueeze(1)?; // [batch, 1]
        let counts_safe = (counts_unsqueezed + 1e-9)?;
        let mean_pooled = summed.broadcast_div(&counts_safe)?; // [batch, hidden] / [batch, 1] ✓

        // L2 normalize
        let norms = mean_pooled.sqr()?.sum(1)?.sqrt()?; // [batch]
        let norms_unsqueezed = norms.unsqueeze(1)?; // [batch, 1]
        let normalized = mean_pooled.broadcast_div(&(norms_unsqueezed + 1e-9)?)?;

        // Convert to Vec<Vec<f32>>
        let flat: Vec<f32> = normalized.flatten_all()?.to_vec1()?;
        let embeddings: Vec<Vec<f32>> = flat
            .chunks(hidden_dim)
            .map(|c| c.to_vec())
            .collect();

        Ok(embeddings)
    }
}

impl EmbeddingProvider for OnnxEmbeddingProvider {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Embedding>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        // BERT max input is 512 tokens; process in batches of 16 to stay within memory
        let mut results = Vec::new();
        for chunk in texts.chunks(16) {
            results.extend(self.infer_batch(chunk)?);
        }
        Ok(results)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

// --- Model download via hf-hub ---------------------------------------

const MODEL_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";

fn ensure_model_downloaded() -> Result<PathBuf> {
    let api = Api::new()?;
    let repo = api.model(MODEL_ID.to_string());

    // Download the three required files
    let files = ["config.json", "tokenizer.json", "model.safetensors"];
    let mut paths = Vec::new();

    for file in &files {
        let path = repo.get(file).with_context(|| format!("download {file}"))?;
        paths.push(path);
    }

    // Return the directory (all files are in the same cache dir)
    Ok(paths[0].parent().unwrap().to_path_buf())
}
