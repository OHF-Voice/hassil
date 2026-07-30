"""Tests for number formatting engine lookup."""

import pytest

from hassil.numbers import get_rbnf_engine


def _words(language: str, number: int) -> set:
    """Return every spelled-out form of a number in a language."""
    return set(get_rbnf_engine(language).format_number(number).text_by_ruleset.values())


def test_exact_language() -> None:
    """Test a language that unicode-rbnf supports directly."""
    assert "two" in _words("en", 2)


def test_language_family_fallback() -> None:
    """Test that a region falls back to the language family."""
    assert _words("en-US", 2) == _words("en", 2)


def test_separator_is_normalized() -> None:
    """Test that "-" is accepted where unicode-rbnf uses "_"."""
    # Serbian written in Latin script, not the Cyrillic of the "sr" family.
    assert "dva" in _words("sr-Latn", 2)
    assert "два" not in _words("sr-Latn", 2)


@pytest.mark.parametrize("language", ["zh-TW", "zh-HK", "zh-MO"])
def test_traditional_chinese_regions(language: str) -> None:
    """Test that regions using Traditional Chinese do not get Simplified rules."""
    words = _words(language, 2)

    # 兩 is how the quantity 2 is spoken; the Simplified 两 must not be used.
    assert "兩" in words
    assert "两" not in words

    # 二 is written the same in both scripts and stays available.
    assert "二" in words


def test_simplified_chinese_is_unchanged() -> None:
    """Test that Simplified regions still get the Simplified rules."""
    words = _words("zh-CN", 2)
    assert "两" in words
    assert "兩" not in words


def test_unsupported_language() -> None:
    """Test that an unsupported language still raises."""
    with pytest.raises(ValueError):
        get_rbnf_engine("nonexistent-language")


def test_engine_is_cached() -> None:
    """Test that repeated lookups return the same engine."""
    assert get_rbnf_engine("zh-TW") is get_rbnf_engine("zh-TW")
