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

    /// Analyze a single conversation turn (user utterance + AI response).
    pub async fn coach_conversation(
        &self,
        user_text: &str,
        ai_response: Option<&str>,
        agent: &str,
    ) -> Result<CoachingFeedback> {
        let response_text = ai_response.unwrap_or("(no response)");
        let system_prompt = COACH_SYSTEM_PROMPT;

        let user_prompt = format!(
            "User asked the AI agent '{agent}':\n\"{user_text}\"\n\nThe AI responded:\n\"{response_text}\""
        );

        let request = ChatRequest {
            model: self.model.clone(),
            messages: vec![
                Message {
                    role: "system".into(),
                    content: system_prompt.into(),
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

        let feedback: CoachingFeedback = serde_json::from_str(&content)
            .map_err(|e| anyhow::anyhow!("Failed to parse coaching JSON: {e}\nContent: {content}"))?;

        Ok(feedback)
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
