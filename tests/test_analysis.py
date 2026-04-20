import unittest

from agent_utterance_analysis.analysis import build_report


class AnalysisTests(unittest.TestCase):
    def test_analysis_flags_common_wording_issues(self) -> None:
        report = build_report(
            [
                {
                    "source_agent": "codex",
                    "timestamp": "2026-04-20T10:00:00+08:00",
                    "source_path": "sample.json",
                    "text": "Can you batchly export these utternaces and tell me whether my display is correct or not?",
                }
            ]
        )
        self.assertIn("batchly", report)
        self.assertIn("utterances", report)
        self.assertIn("Distribution by Agent", report)


if __name__ == "__main__":
    unittest.main()
