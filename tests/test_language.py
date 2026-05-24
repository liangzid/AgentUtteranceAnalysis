import unittest

from agent_utterance_analysis.language import is_probably_english


class LanguageTests(unittest.TestCase):
    def test_english_detection_keeps_english_dominant_text(self) -> None:
        self.assertTrue(is_probably_english("Please review this PR and explain the failing test."))

    def test_english_detection_rejects_chinese_dominant_text_with_terms(self) -> None:
        self.assertFalse(is_probably_english("帮我分析 Docker 里面的 logs，但是 docker 里的不算。"))

    def test_english_detection_rejects_mixed_chinese_and_english_text(self) -> None:
        self.assertFalse(is_probably_english("Great. 我提几个点: please make the method clearer."))


if __name__ == "__main__":
    unittest.main()
