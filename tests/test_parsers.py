from pathlib import Path
import unittest

from agent_utterance_analysis.parsers import parse_json, parse_jsonl, parse_labeled_text


class ParserTests(unittest.TestCase):
    def test_parse_json_extracts_user_messages(self) -> None:
        raw = """
        {
          "id": "abc",
          "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "human", "content": [{"text": "Please help"}]}
          ]
        }
        """
        utterances = parse_json(raw, Path("codex.json"), "codex")
        self.assertEqual([item.text for item in utterances], ["Hello", "Please help"])
        self.assertEqual(utterances[0].conversation_id, "abc")


    def test_parse_markdown_speaker_blocks(self) -> None:
        raw = """
        ## User
        Please fix this.

        ## Assistant
        Sure.

        User: Can you explain it?
        """
        utterances = parse_labeled_text(raw, Path("claude.md"), "claude-code", markdown=True)
        self.assertEqual([item.text for item in utterances], ["Please fix this.", "Can you explain it?"])

    def test_parse_codex_history_jsonl(self) -> None:
        raw = '{"session_id":"abc","ts":1776734600,"text":"Please review this change."}'

        utterances = parse_jsonl(raw, Path("history.jsonl"), "codex")

        self.assertEqual([item.text for item in utterances], ["Please review this change."])
        self.assertEqual(utterances[0].conversation_id, "abc")

    def test_parse_claude_nested_user_message(self) -> None:
        raw = (
            '{"type":"user","message":{"role":"user","content":"Read the file and complete the task."},'
            '"sessionId":"session-1","timestamp":"2026-04-20T08:46:37.702Z"}'
        )

        utterances = parse_jsonl(raw, Path("claude.jsonl"), "claude-code")

        self.assertEqual([item.text for item in utterances], ["Read the file and complete the task."])

    def test_parse_codex_session_payload_user_message(self) -> None:
        raw = (
            '{"timestamp":"2026-05-24T09:11:35.210Z","type":"response_item",'
            '"payload":{"type":"message","role":"user","content":[{"type":"input_text",'
            '"text":"Can you run the next experiment group?"}]}}'
        )

        utterances = parse_jsonl(raw, Path("rollout.jsonl"), "codex")

        self.assertEqual([item.text for item in utterances], ["Can you run the next experiment group?"])


if __name__ == "__main__":
    unittest.main()
