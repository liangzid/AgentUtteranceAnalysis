from pathlib import Path
import unittest

from agent_utterance_analysis.parsers import parse_json, parse_labeled_text


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


if __name__ == "__main__":
    unittest.main()
