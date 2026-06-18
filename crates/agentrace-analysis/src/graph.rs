// ======================================================================
// KNOWLEDGE GRAPH BUILDER
//
// 1. PCA dimensionality reduction: 384-dim embeddings → 3D coordinates.
// 2. Graph edge construction: connect semantically similar utterances.
// 3. Uses power iteration (no external LA library needed).
// ======================================================================

use anyhow::Result;
use ndarray::{Array1, Array2, ArrayView1, Axis};
use serde::Serialize;

/// A node in the 3D knowledge graph.
#[derive(Debug, Clone, Serialize)]
pub struct GraphNode {
    pub utterance_id: String,
    pub text: String,
    pub source_agent: String,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub cluster_id: u32,
}

/// An edge connecting two semantically similar nodes.
#[derive(Debug, Clone, Serialize)]
pub struct GraphEdge {
    pub source: usize,
    pub target: usize,
    pub similarity: f32,
}

/// Complete 3D knowledge graph.
#[derive(Debug, Clone, Serialize)]
pub struct KnowledgeGraph {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
    pub variance_explained: [f32; 3],
}

/// Reduce embeddings to 3D via PCA using power iteration.
pub fn pca_reduce(embeddings: &[Vec<f32>], n_components: usize) -> Result<(Array2<f32>, [f32; 3])> {
    let n = embeddings.len();
    let d = embeddings[0].len();
    if n < n_components {
        anyhow::bail!("need at least {} embeddings, got {}", n_components, n);
    }

    // Build data matrix X: [n × d]
    let flat: Vec<f32> = embeddings.iter().flat_map(|e| e.iter().copied()).collect();
    let x = Array2::from_shape_vec((n, d), flat)?;

    // Center the data
    let mean = x.mean_axis(Axis(0)).unwrap();
    let x_centered = x - &mean.insert_axis(Axis(0));

    // Covariance matrix C = X^T X / (n-1): [d × d]
    let xt = x_centered.t();
    let c = xt.dot(&x_centered) / (n as f32 - 1.0);

    // Power iteration for top k eigenvectors
    let mut components = Vec::new();
    let mut eigenvalues = Vec::new();
    let mut residual = c.clone();

    for _ in 0..n_components {
        // Random initialization
        let mut v = Array1::from_vec(vec![1.0f32 / (d as f32).sqrt(); d]);
        // Power iteration
        for _ in 0..100 {
            let v_new = residual.dot(&v);
            let norm = v_new.dot(&v_new).sqrt();
            if norm < 1e-10 {
                break;
            }
            v = &v_new / norm;
        }
        // Compute eigenvalue
        let av = residual.dot(&v);
        let lambda = v.dot(&av);
        eigenvalues.push(lambda);
        components.push(v.clone());
        // Deflate
        let outer = v.clone().into_shape((d, 1))?.dot(&v.clone().into_shape((1, d))?);
        residual = &residual - &(lambda * outer);
    }

    // Project centered data onto components: X_centered @ V^T → [n × k]
    let v_stack: Vec<f32> = components.iter().flat_map(|c| c.iter().copied()).collect();
    let v_mat = Array2::from_shape_vec((n_components, d), v_stack)?; // [k × d]
    let projection = x_centered.dot(&v_mat.t()); // [n × k]

    // Compute variance explained
    let total_var: f32 = (0..d).map(|i| c[[i, i]]).sum();
    let mut var_exp = [0.0f32; 3];
    for i in 0..n_components.min(3) {
        var_exp[i] = eigenvalues[i] / total_var;
    }

    Ok((projection, var_exp))
}

/// Build edges by connecting nodes with cosine similarity above a threshold.
pub fn build_similarity_edges(
    embeddings: &[Vec<f32>],
    nodes: &[GraphNode],
    threshold: f32,
    max_edges: usize,
) -> Vec<GraphEdge> {
    let mut edges = Vec::new();
    let n = nodes.len();
    // Only connect nearby pairs to keep the graph sparse
    for i in 0..n {
        for j in (i + 1)..n {
            let sim = cosine_similarity(&embeddings[i], &embeddings[j]);
            if sim >= threshold {
                edges.push(GraphEdge {
                    source: i,
                    target: j,
                    similarity: sim,
                });
                if edges.len() >= max_edges {
                    return edges;
                }
            }
        }
    }
    // Sort by similarity descending
    edges.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
    edges.truncate(max_edges);
    edges
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Simple k-means clustering of embeddings.
/// Returns (cluster_assignments, k) where assignments[i] is the cluster index.
pub fn kmeans_cluster(embeddings: &[Vec<f32>], k: usize, max_iters: usize) -> Vec<usize> {
    let n = embeddings.len();
    let d = embeddings[0].len();
    if n <= k {
        return (0..n).collect();
    }

    // Initialize centroids: pick k evenly spaced embeddings
    let mut centroids: Vec<Vec<f32>> = (0..k)
        .map(|i| embeddings[i * n / k].clone())
        .collect();
    let mut assignments = vec![0usize; n];

    for _ in 0..max_iters {
        // Assign each point to nearest centroid
        let mut changed = false;
        for (i, emb) in embeddings.iter().enumerate() {
            let best = centroids.iter()
                .enumerate()
                .map(|(j, c)| (j, cosine_similarity(emb, c)))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(j, _)| j)
                .unwrap_or(0);
            if assignments[i] != best {
                assignments[i] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }

        // Update centroids: mean of assigned points
        for j in 0..k {
            let members: Vec<&Vec<f32>> = assignments.iter().enumerate()
                .filter(|(_, c)| **c == j)
                .map(|(i, _)| &embeddings[i])
                .collect();
            if !members.is_empty() {
                let mut mean = vec![0.0f32; d];
                for m in &members {
                    for (dim, &val) in m.iter().enumerate() {
                        mean[dim] += val;
                    }
                }
                for dim in 0..d {
                    mean[dim] /= members.len() as f32;
                }
                centroids[j] = mean;
            }
        }
    }

    assignments
}

// ======================================================================
// Tests
// ======================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pca_reduces_dimensions() {
        // 10 random 384-dim vectors → 3-dim
        let embeddings: Vec<Vec<f32>> = (0..10)
            .map(|_| (0..384).map(|_| rand::random::<f32>() * 0.1).collect())
            .collect();

        let (proj, var) = pca_reduce(&embeddings, 3).unwrap();
        assert_eq!(proj.shape(), &[10, 3]);
        // Variance explained should sum to ~1.0
        let total: f32 = var.iter().sum();
        assert!(total > 0.0 && total <= 1.1);
    }

    #[test]
    fn pca_preserves_structure() {
        // Two clusters in high-dim: PCA should separate them in 3D
        let mut embeddings = Vec::new();
        for _ in 0..5 {
            let mut v = vec![0.0f32; 384];
            v[0] = 1.0;
            v[1] = rand::random::<f32>() * 0.01;
            embeddings.push(v);
        }
        for _ in 0..5 {
            let mut v = vec![0.0f32; 384];
            v[1] = 1.0;
            v[0] = rand::random::<f32>() * 0.01;
            embeddings.push(v);
        }

        let (proj, _) = pca_reduce(&embeddings, 2).unwrap();
        // First 5 should cluster together, last 5 should cluster separately
        let c0_avg = proj.slice(ndarray::s![0..5, 0]).mean().unwrap();
        let c1_avg = proj.slice(ndarray::s![5..10, 0]).mean().unwrap();
        // The two clusters should be separated along PC1
        assert!((c0_avg - c1_avg).abs() > 0.1);
    }
}
