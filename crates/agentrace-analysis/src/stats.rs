// ======================================================================
// `ANALYSIS — STATS`
//
// 1. Basic statistical analysis of utterances: word/char counts,
//    agent/time distributions, heuristic property detection,
//    English naturalness warnings.
// 2. Ported from Python analysis.py.
//
//     Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
//     Copyright © 2025, Zi Liang, all rights reserved.
//     Created: 16 June 2025
// ======================================================================

use regex::Regex;
use serde::Serialize;
use std::collections::HashMap;

const QUESTION_WORDS: &[&str] = &[
    "what", "why", "how", "when", "where", "who", "which", "can", "could",
    "would", "should", "do", "does", "is", "are",
];
const REQUEST_MARKERS: &[&str] = &[
    "please", "want", "need", "create", "write", "fix", "implement",
    "analyze", "explain", "review", "help",
];
const POLITENESS_MARKERS: &[&str] = &["please", "thanks", "thank", "could", "would"];

/// A single style warning found in an utterance.
#[derive(Debug, Clone, Serialize)]
pub struct StyleWarning {
    pub source: String,
    pub message: String,
    pub excerpt: String,
}

/// Complete analysis report.
#[derive(Debug, Clone, Serialize)]
pub struct StatsReport {
    pub utterance_count: usize,
    pub conversation_count: usize,
    pub avg_words: f64,
    pub median_words: f64,
    pub max_words: usize,
    pub avg_chars: f64,
    pub agent_distribution: HashMap<String, usize>,
    pub month_distribution: HashMap<String, usize>,
    pub properties: HashMap<String, usize>,
    pub style_warnings: Vec<StyleWarning>,
}

/// Row from the database for analysis.
#[derive(Debug, Clone)]
pub struct AnalysisRow {
    pub text: String,
    pub source_agent: String,
    pub source_path: String,
    pub timestamp: Option<String>,
}

/// Run the full stats analysis.
pub fn analyze_stats(
    rows: &[AnalysisRow],
    conversation_count: usize,
    agent_dist: Vec<(String, i64)>,
    month_dist: Vec<(String, i64)>,
) -> StatsReport {
    let texts: Vec<&str> = rows.iter().map(|r| r.text.as_str()).collect();
    let words_per: Vec<usize> = texts.iter().map(|t| word_count(t)).collect();
    let chars_per: Vec<usize> = texts.iter().map(|t| t.len()).collect();

    let avg_words = if words_per.is_empty() {
        0.0
    } else {
        words_per.iter().sum::<usize>() as f64 / words_per.len() as f64
    };

    let median_words = median(&words_per);
    let max_words = words_per.iter().copied().max().unwrap_or(0);
    let avg_chars = if chars_per.is_empty() {
        0.0
    } else {
        chars_per.iter().sum::<usize>() as f64 / chars_per.len() as f64
    };

    let agent_distribution: HashMap<String, usize> = agent_dist
        .into_iter()
        .map(|(k, v)| (k, v as usize))
        .collect();
    let month_distribution: HashMap<String, usize> = month_dist
        .into_iter()
        .map(|(k, v)| (k, v as usize))
        .collect();

    let properties = property_distribution(&texts);
    let style_warnings = collect_style_warnings(rows);

    StatsReport {
        utterance_count: rows.len(),
        conversation_count,
        avg_words,
        median_words,
        max_words,
        avg_chars,
        agent_distribution,
        month_distribution,
        properties,
        style_warnings,
    }
}

/// Count words in text.
fn word_count(text: &str) -> usize {
    let re = Regex::new(r"\b[\w']+\b").unwrap();
    re.find_iter(text).count()
}

/// Compute median of a slice of usize.
fn median(values: &[usize]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<usize> = values.to_vec();
    sorted.sort();
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) as f64 / 2.0
    } else {
        sorted[mid] as f64
    }
}

/// Detect heuristic properties in utterance texts.
fn property_distribution(texts: &[&str]) -> HashMap<String, usize> {
    let mut counter: HashMap<String, usize> = HashMap::new();
    for text in texts {
        let lower = text.to_lowercase();
        let words: Vec<&str> = Regex::new(r"[A-Za-z']+")
            .unwrap()
            .find_iter(text)
            .map(|m| m.as_str())
            .collect();
        let lower_words: std::collections::HashSet<&str> =
            words.iter().map(|w| *w).collect();

        if text.contains('?')
            || QUESTION_WORDS
                .iter()
                .any(|qw| lower_words.contains(qw))
        {
            *counter.entry("question_or_inquiry".into()).or_default() += 1;
        }
        if REQUEST_MARKERS
            .iter()
            .any(|rm| lower_words.contains(rm))
        {
            *counter.entry("request_or_instruction".into()).or_default() += 1;
        }
        if POLITENESS_MARKERS
            .iter()
            .any(|pm| lower_words.contains(pm))
        {
            *counter.entry("politeness_marker".into()).or_default() += 1;
        }
        if word_count(text) >= 80 {
            *counter.entry("long_context_prompt".into()).or_default() += 1;
        }
        if Regex::new(r"\b\d+\.").unwrap().is_match(text) {
            *counter.entry("numbered_requirements".into()).or_default() += 1;
        }
        if text.contains("```") {
            *counter.entry("contains_code_block".into()).or_default() += 1;
        }
    }
    counter
}

/// Collect English naturalness warnings using regex heuristics.
fn collect_style_warnings(rows: &[AnalysisRow]) -> Vec<StyleWarning> {
    let style_patterns: Vec<(Regex, &str)> = vec![
        (
            Regex::new(r"(?i)\b(batchly)\b").unwrap(),
            "Use 'in batches' or 'batch' instead of 'batchly'.",
        ),
        (
            Regex::new(r"(?i)\b(?:utternaces|utternace|utternances|utternance|uterances|utterences)\b").unwrap(),
            "Check spelling: 'utterances'.",
        ),
        (
            Regex::new(r"(?i)\bdisplay\b").unwrap(),
            "When discussing language, 'expression' or 'wording' is usually more natural than 'display'.",
        ),
        (
            Regex::new(r"(?i)\bcorrect or not\b").unwrap(),
            "More natural: 'whether ... is natural and correct'.",
        ),
        (
            Regex::new(r"(?i)\band so on\b").unwrap(),
            "'and so on' is understandable, but 'and similar tools' is often more precise.",
        ),
        (
            Regex::new(r"(?i)\bI want to know whether\b").unwrap(),
            "Natural, but often shorter as 'I want to know if'.",
        ),
    ];

    let mut warnings = Vec::new();
    for row in rows {
        for (pattern, message) in &style_patterns {
            if pattern.is_match(&row.text) {
                warnings.push(StyleWarning {
                    source: row.source_path.clone(),
                    message: message.to_string(),
                    excerpt: excerpt(&row.text, 180),
                });
            }
        }
        // Sentence complexity check
        let complexity = sentence_complexity(&row.text);
        if complexity > 38.0 {
            warnings.push(StyleWarning {
                source: row.source_path.clone(),
                message:
                    "This sentence may be hard to read; consider splitting it into shorter requests."
                        .into(),
                excerpt: excerpt(&row.text, 180),
            });
        }
    }
    warnings
}

/// Sentence complexity score: words / sqrt(sentence_count).
fn sentence_complexity(text: &str) -> f64 {
    let sentences: Vec<&str> = text.split(|c| c == '.' || c == '!' || c == '?')
        .filter(|s| !s.trim().is_empty())
        .collect();
    if sentences.is_empty() {
        return 0.0;
    }
    let words = word_count(text) as f64;
    words / (sentences.len() as f64).sqrt()
}

/// Truncate text for excerpt display.
fn excerpt(text: &str, limit: usize) -> String {
    let compact = Regex::new(r"\s+").unwrap().replace_all(text.trim(), " ");
    if compact.chars().count() <= limit {
        compact.to_string()
    } else {
        format!("{}...", compact.chars().take(limit.saturating_sub(3)).collect::<String>())
    }
}

// ======================================================================
// Tests
// ======================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_row(text: &str) -> AnalysisRow {
        AnalysisRow {
            text: text.into(),
            source_agent: "codex".into(),
            source_path: "/test.json".into(),
            timestamp: None,
        }
    }

    #[test]
    fn word_count_english() {
        assert_eq!(word_count("hello world"), 2);
        assert_eq!(word_count(""), 0);
        assert_eq!(word_count("one two three four"), 4);
    }

    #[test]
    fn median_odd() {
        assert_eq!(median(&[1, 3, 2]), 2.0);
    }

    #[test]
    fn median_even() {
        assert_eq!(median(&[1, 2, 3, 4]), 2.5);
    }

    #[test]
    fn median_empty() {
        assert_eq!(median(&[]), 0.0);
    }

    #[test]
    fn analyze_stats_basic() {
        let rows = vec![
            make_row("hello world"),
            make_row("fix the bug please"),
            make_row("what is rust?"),
        ];
        let report = analyze_stats(
            &rows,
            2,
            vec![("codex".into(), 3)],
            vec![("2024-06".into(), 3)],
        );

        assert_eq!(report.utterance_count, 3);
        assert_eq!(report.conversation_count, 2);
        assert_eq!(report.avg_words, (2.0 + 4.0 + 3.0) / 3.0);
        assert_eq!(report.max_words, 4);
        assert_eq!(
            *report.agent_distribution.get("codex").unwrap(),
            3usize
        );
    }

    #[test]
    fn property_detection_question() {
        let texts: Vec<&str> = vec!["what is rust?", "hello", "can you help?"];
        let props = property_distribution(&texts);
        assert_eq!(props.get("question_or_inquiry"), Some(&2usize));
    }

    #[test]
    fn property_detection_request() {
        let texts: Vec<&str> = vec!["please fix this", "create a file", "hello"];
        let props = property_distribution(&texts);
        assert_eq!(props.get("request_or_instruction"), Some(&2usize));
    }

    #[test]
    fn property_detection_code_block() {
        let texts: Vec<&str> = vec!["here is code: ```rust\nfn main() {}\n```"];
        let props = property_distribution(&texts);
        assert_eq!(props.get("contains_code_block"), Some(&1usize));
    }

    #[test]
    fn property_detection_long_context() {
        let long = "word ".repeat(80);
        let texts: Vec<&str> = vec![&long];
        let props = property_distribution(&texts);
        assert_eq!(props.get("long_context_prompt"), Some(&1usize));
    }

    #[test]
    fn style_warning_batchly() {
        let rows = vec![make_row("Can you batchly export these?")];
        let warnings = collect_style_warnings(&rows);
        assert!(!warnings.is_empty());
        assert!(warnings[0].message.contains("batchly"));
    }

    #[test]
    fn style_warning_utterances_spelling() {
        let rows = vec![make_row("export these utternaces")];
        let warnings = collect_style_warnings(&rows);
        assert!(!warnings.is_empty());
        assert!(warnings[0].message.contains("spelling"));
    }

    #[test]
    fn excerpt_truncation() {
        let long = "a".repeat(200);
        let ex = excerpt(&long, 100);
        assert!(ex.len() <= 100);
        assert!(ex.ends_with("..."));
    }
}
