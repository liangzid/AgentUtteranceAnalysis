/// Integration test: real BERT embedding via candle.
/// Tests semantic similarity: same-meaning texts cluster, different-meaning texts separate.
#[test]
fn real_embedding_semantic_clustering() {
    use agentrace_embedding::candle::OnnxEmbeddingProvider;
    use agentrace_embedding::EmbeddingProvider;

    let provider = OnnxEmbeddingProvider::load().expect("load model");
    assert_eq!(provider.dimension(), 384);

    // Group A: coding-related
    let coding = [
        "How to fix a Rust compilation error",
        "Troubleshooting Python import issues",
        "Debugging a segmentation fault in C",
    ];
    // Group B: cooking-related
    let cooking = [
        "How to make Italian pasta at home",
        "Best recipes for chocolate cake",
        "Cooking techniques for tender steak",
    ];

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na == 0.0 || nb == 0.0 { return 0.0; }
        dot / (na * nb)
    }

    fn avg_pairwise(embeddings: &[Vec<f32>]) -> f32 {
        let mut sum = 0.0f32;
        let mut count = 0u32;
        for i in 0..embeddings.len() {
            for j in (i + 1)..embeddings.len() {
                sum += cosine(&embeddings[i], &embeddings[j]);
                count += 1;
            }
        }
        sum / count as f32
    }

    fn cross_avg(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
        let mut sum = 0.0f32;
        let mut count = 0u32;
        for ea in a {
            for eb in b {
                sum += cosine(ea, eb);
                count += 1;
            }
        }
        sum / count as f32
    }

    let coding_emb = provider.embed(&coding).expect("coding embed");
    let cooking_emb = provider.embed(&cooking).expect("cooking embed");

    let intra_coding = avg_pairwise(&coding_emb);
    let intra_cooking = avg_pairwise(&cooking_emb);
    let cross = cross_avg(&coding_emb, &cooking_emb);

    println!("Intra-coding similarity:  {:.4}", intra_coding);
    println!("Intra-cooking similarity: {:.4}", intra_cooking);
    println!("Cross-domain similarity:  {:.4}", cross);

    // Same-domain texts should be more similar than cross-domain
    assert!(intra_coding > cross,
        "Coding texts should cluster together (intra={:.4} > cross={:.4})",
        intra_coding, cross);
    assert!(intra_cooking > cross,
        "Cooking texts should cluster together (intra={:.4} > cross={:.4})",
        intra_cooking, cross);

    // Assertion passed: the model produces meaningful semantic embeddings!
    println!("\n✅ all-MiniLM-L6-v2 embeddings are semantically meaningful!");
}
