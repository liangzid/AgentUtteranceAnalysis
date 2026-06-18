// ======================================================================
// `AGENTRACE-LLM`
//
// 1. DeepSeek API client for conversation-level coaching analysis.
// 2. OpenAI-compatible chat completions with JSON structured output.
// 3. Uses DEEPSEEK_API_KEY environment variable.
//
//     Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
//     Copyright © 2025, Zi Liang, all rights reserved.
//     Created: 17 June 2025
// ======================================================================

use anyhow::Result;
use serde::{Deserialize, Serialize};

const DEEPSEEK_BASE: &str = "https://api.deepseek.com";
const CHAT_ENDPOINT: &str = "/v1/chat/completions";
const DEFAULT_MODEL: &str = "deepseek-chat";

/// Coaching feedback for a single user-AI conversation turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoachingFeedback {
    /// One-sentence summary of what the user was trying to do.
    pub intent: String,
    /// What the user did well (positive reinforcement).
    pub what_worked: String,
    /// What could be improved.
    pub could_improve: String,
    /// A better version of the prompt the user could have used.
    pub better_prompt: String,
    /// A tip about a tool/feature/command that could have solved the problem directly.
    pub hidden_tip: String,
    /// Knowledge area the user might want to study.
    pub knowledge_gap: String,
    /// Interaction style category.
    pub interaction_style: InteractionStyle,
    /// 1-5 clarity score.
    pub clarity_score: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InteractionStyle {
    Direct,
    Exploratory,
    Helpless,
    Vague,
    WellStructured,
}

/// Summary of an entire conversation session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    pub session_id: String,
    pub title: String,
    pub topics: Vec<String>,
    pub summary: String,
    pub dominant_language: String,
}

/// DeepSeek API client.
pub struct DeepSeekClient {
    client: reqwest::Client,
    api_key: String,
    model: String,
}

impl DeepSeekClient {
    /// Create a new client, reading DEEPSEEK_API_KEY from environment.
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("DEEPSEEK_API_KEY")
            .map_err(|_| anyhow::anyhow!("DEEPSEEK_API_KEY not set"))?;
        Ok(Self {
            client: reqwest::Client::new(),
            api_key,
            model: DEFAULT_MODEL.to_string(),
        })
    }

    /// Analyze a batch of conversation turns in a single API call.
    /// Each item is (user_text, ai_response, agent_name).
    /// Returns one CoachingFeedback per input item, in the same order.
    pub async fn coach_batch(
        &self,
        turns: &[(String, Option<String>, String)],
    ) -> Result<Vec<CoachingFeedback>> {
        if turns.is_empty() {
            return Ok(vec![]);
        }

        // Build the prompt: list all turns with indices
        let mut turns_text = String::new();
        for (i, (user_text, ai_response, agent)) in turns.iter().enumerate() {
            let response_text = ai_response.as_deref().unwrap_or("(no response)");
            turns_text.push_str(&format!(
                "--- Turn {idx} (agent: {agent}) ---\n\
                 User: \"{user}\"\n\
                 AI response: \"{ai}\"\n\n",
                idx = i,
                agent = agent,
                user = user_text,
                ai = response_text,
            ));
        }

        let user_prompt = format!(
            "Analyze the following {count} conversation turns between a user and AI coding assistants. \
             Return coaching feedback for each turn.\n\n{turns}",
            count = turns.len(),
            turns = turns_text,
        );

        let request = ChatRequest {
            model: self.model.clone(),
            messages: vec![
                Message {
                    role: "system".into(),
                    content: COACH_BATCH_SYSTEM_PROMPT.into(),
                },
                Message {
                    role: "user".into(),
                    content: user_prompt,
                },
            ],
            response_format: Some(ResponseFormat {
                r#type: "json_object".into(),
            }),
            temperature: Some(0.3),
        };

        let resp = self
            .client
            .post(format!("{}{}", DEEPSEEK_BASE, CHAT_ENDPOINT))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("DeepSeek API error {status}: {body}");
        }

        let chat_resp: ChatResponse = resp.json().await?;
        let content = chat_resp
            .choices
            .first()
            .map(|c| &c.message.content)
            .unwrap_or(&String::new())
            .clone();

        // Parse the batch response: {"feedback": [{...}, {...}]}
        // LLMs sometimes return truncated/invalid JSON — try recovery.
        let sanitized = sanitize_json_str(&content);
        let parsed: serde_json::Value = match serde_json::from_str(&sanitized) {
            Ok(v) => v,
            Err(e) => {
                // Try to recover truncated arrays by closing them
                let repaired = repair_truncated_json(&sanitized);
                match serde_json::from_str(&repaired) {
                    Ok(v) => {
                        tracing::warn!("JSON repaired after truncation: {}", e);
                        v
                    }
                    Err(e2) => {
                        tracing::warn!(
                            "Batch JSON unparseable ({}). First 300 chars: {}",
                            e2,
                            sanitized.chars().take(300).collect::<String>()
                        );
                        return Ok(vec![]);
                    }
                }
            }
        };

        let feedbacks: Vec<CoachingFeedback> = parsed
            .get("feedback")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|item| serde_json::from_value(item.clone()).ok())
                    .collect()
            })
            .unwrap_or_default();

        if feedbacks.len() != turns.len() {
            tracing::warn!(
                "Batch coaching: expected {} feedbacks, got {}. Partial result.",
                turns.len(),
                feedbacks.len()
            );
        }

        Ok(feedbacks)
    }

    /// Summarize multiple conversation sessions — extract topics, purpose, and patterns.
    /// Each session is (session_id, turns_text) where turns_text is the concatenated
    /// user + assistant messages for that session.
    pub async fn summarize_sessions(
        &self,
        sessions: &[(String, String)],
    ) -> Result<Vec<SessionSummary>> {
        if sessions.is_empty() {
            return Ok(vec![]);
        }

        let mut sessions_text = String::new();
        for (i, (id, turns)) in sessions.iter().enumerate() {
            let preview: String = turns.chars().take(500).collect();
            sessions_text.push_str(&format!(
                "--- Session {idx} (id: {id}) ---\n{preview}\n\n",
                idx = i,
                id = id,
                preview = preview,
            ));
        }

        let request = ChatRequest {
            model: self.model.clone(),
            messages: vec![
                Message { role: "system".into(), content: SESSION_SUMMARY_SYSTEM_PROMPT.into() },
                Message {
                    role: "user".into(),
                    content: format!(
                        "Summarize these {count} AI coding sessions:\n\n{sessions}",
                        count = sessions.len(),
                        sessions = sessions_text,
                    ),
                },
            ],
            response_format: Some(ResponseFormat { r#type: "json_object".into() }),
            temperature: Some(0.3),
        };

        let resp = self.client
            .post(format!("{}{}", DEEPSEEK_BASE, CHAT_ENDPOINT))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("DeepSeek API error {status}: {body}");
        }

        let chat_resp: ChatResponse = resp.json().await?;
        let content = chat_resp.choices.first()
            .map(|c| &c.message.content)
            .unwrap_or(&String::new())
            .clone();

        let sanitized = sanitize_json_str(&content);
        let parsed: serde_json::Value = serde_json::from_str(&sanitized)
            .unwrap_or_else(|_| serde_json::json!({ "sessions": [] }));

        let summaries: Vec<SessionSummary> = parsed
            .get("sessions")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter()
                .filter_map(|item| serde_json::from_value(item.clone()).ok())
                .collect())
            .unwrap_or_default();

        Ok(summaries)
    }
}

// --- OpenAI-compatible API types ---

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ResponseFormat {
    r#type: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: ChoiceMessage,
}

#[derive(Deserialize)]
struct ChoiceMessage {
    content: String,
}

// --- Prompt ---

/// Clean up LLM-generated JSON that may contain invalid escape sequences.
/// Replaces lone backslashes that are not part of valid JSON escapes.
fn sanitize_json_str(raw: &str) -> String {
    let mut result = String::with_capacity(raw.len());
    let chars: Vec<char> = raw.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '\\' && i + 1 < chars.len() {
            let next = chars[i + 1];
            if matches!(next, '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' | 'u') {
                result.push('\\');
                result.push(next);
                i += 2;
            } else {
                result.push_str("\\\\");
                result.push(next);
                i += 2;
            }
        } else {
            result.push(chars[i]);
            i += 1;
        }
    }
    result
}

/// Try to repair JSON that was truncated mid-response by the LLM.
/// Closes unclosed strings, arrays, and objects.
fn repair_truncated_json(raw: &str) -> String {
    let trimmed = raw.trim();
    let mut result = trimmed.to_string();

    // Count brackets
    let open_braces: usize = result.chars().filter(|&c| c == '{').count();
    let close_braces: usize = result.chars().filter(|&c| c == '}').count();
    let open_brackets: usize = result.chars().filter(|&c| c == '[').count();
    let close_brackets: usize = result.chars().filter(|&c| c == ']').count();

    // Close unclosed strings: if we have an odd number of quotes outside brackets,
    // add a closing quote
    let quote_count: usize = result.chars().filter(|&c| c == '"').count();
    if quote_count % 2 != 0 {
        result.push('"');
    }

    // Close unclosed arrays/objects
    for _ in 0..(open_brackets.saturating_sub(close_brackets)) {
        result.push(']');
    }
    for _ in 0..(open_braces.saturating_sub(close_braces)) {
        result.push('}');
    }

    result
}

const COACH_SYSTEM_PROMPT: &str = r#"You are an expert AI interaction coach. You analyze conversations between users and AI coding assistants (Claude, Codex, OpenCode, etc.) to help users improve how they interact with AI.

For each conversation turn, provide coaching feedback in this exact JSON format:
{
  "intent": "one sentence describing what the user wanted",
  "what_worked": "what the user did well (be specific, positive)",
  "could_improve": "what the user could have done better (be constructive)",
  "better_prompt": "a concrete example of a better way to phrase this request",
  "hidden_tip": "if there's a tool/command/feature that could have solved this directly without AI, mention it",
  "knowledge_gap": "what concept or skill the user might want to learn to avoid this kind of question in future",
  "interaction_style": "one of: direct, exploratory, helpless, vague, well_structured",
  "clarity_score": a number 1-5 where 5 is perfect clarity
}

Rules:
- Be specific, not generic. Reference actual technologies, commands, or patterns.
- what_worked must always contain genuine praise — never leave it empty.
- hidden_tip: think about whether the problem could be solved by `cargo check`, `git log`, `npm docs`, IDE features, etc.
- clarity_score 5 = complete context, specific error messages, clear goal. 1 = "it doesn't work" with no details.
- If the AI responded with a clarification question, that signals the user could have been clearer.
- Write in the user's language (match their input language)."#;

const COACH_BATCH_SYSTEM_PROMPT: &str = r#"You are an expert AI interaction coach. You analyze conversations between users and AI coding assistants (Claude, Codex, OpenCode, Reasonix, etc.) to help users improve their communication.

You will receive multiple conversation turns, each labeled "Turn N". For each turn, provide coaching feedback.

Return your response as a JSON object with a "feedback" array:

{
  "feedback": [
    {
      "turn_index": 0,
      "intent": "one sentence describing what the user wanted",
      "what_worked": "what the user did well (be specific, positive)",
      "could_improve": "what could be improved (be constructive)",
      "better_prompt": "a concrete example of a better way to phrase this request",
      "hidden_tip": "a tool/command/feature that could have solved this directly without AI (if applicable)",
      "knowledge_gap": "what concept or skill to learn to improve",
      "interaction_style": "direct | exploratory | helpless | vague | well_structured",
      "clarity_score": 1-5
    },
    ...
  ]
}

Rules:
- The "feedback" array must have exactly one entry per turn, in the same order.
- If a turn is a system/harness message (e.g. "Plan mode", "Plan approved", "Host final-answer readiness check", "<background-jobs>..."), mark it with intent="[SYSTEM_MESSAGE]", clarity_score=0, interaction_style="direct", and leave all other fields as empty strings. Do NOT coach system messages.
- Be specific, reference actual technologies/commands/patterns.
- what_worked: always include genuine praise — never empty (except for system messages).
- hidden_tip: think about `cargo check`, `git log`, IDE features, shell commands, etc.
- clarity_score: 5 = complete context + specific error + clear goal. 1 = "it doesn't work" with no details.
- If the AI responded with a clarification question, that signals the user could have been clearer (score < 4).
- Write in the user's language (match their input language).
- Keep each field concise — 1-2 sentences."#;

const SESSION_SUMMARY_SYSTEM_PROMPT: &str = r#"You are an expert at analyzing AI coding assistant conversation sessions. You will receive conversation transcripts between a user and AI coding assistants.

For each session, provide a summary in this JSON format:

{
  "sessions": [
    {
      "session_id": "the-session-id-provided",
      "title": "a short (3-8 word) descriptive title for this session",
      "topics": ["topic1", "topic2", "topic3"],
      "summary": "a 2-4 sentence summary of what the user worked on, what was accomplished, and the overall nature of the conversation",
      "dominant_language": "en | zh | mixed"
    },
    ...
  ]
}

Rules:
- sessions array must have exactly one entry per session, matched by session_id.
- title: concise and descriptive, like a good commit message.
- topics: 2-5 key technical areas discussed (e.g. "Rust compiler errors", "NixOS configuration", "Emacs Lisp").
- summary: capture the overall goal and outcome of the session.
- dominant_language: use "zh" if primarily Chinese, "en" if primarily English, "mixed" if both."#;
