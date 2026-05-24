from __future__ import annotations

import re


CJK_RE = re.compile(r"[\u3400-\u9fff]")
LATIN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")


def is_probably_english(text: str) -> bool:
    """Return true for utterances that are primarily English prose or commands."""
    stripped = text.strip()
    if not stripped:
        return False

    latin_words = LATIN_WORD_RE.findall(stripped)
    if not latin_words:
        return False

    cjk_chars = CJK_RE.findall(stripped)
    if cjk_chars:
        return False
    letters = [char for char in stripped if char.isalpha()]
    latin_letters = sum(1 for char in stripped if ("A" <= char <= "Z") or ("a" <= char <= "z"))
    if not letters:
        return False

    latin_ratio = latin_letters / len(letters)
    return latin_ratio >= 0.75
