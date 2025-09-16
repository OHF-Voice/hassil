"""Utility methods."""

import re
import unicodedata
from collections.abc import (
    Collection,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)
from typing import Any, Dict, Iterable, Optional

WHITESPACE = re.compile(r"\s+")
WHITESPACE_CAPTURE = re.compile(r"(\s+)")
WHITESPACE_SEPARATOR = " "

TEMPLATE_SYNTAX = re.compile(r".*[(){}<>\[\]|@].*")

PUNCTUATION_STR_NO_PERIOD = "。,，?¿？؟!¡！;；:：’"
PUNCTUATION_PATTERN_NO_PERIOD = rf"[{re.escape(PUNCTUATION_STR_NO_PERIOD)}]+"
PUNCTUATION_STR = f".{PUNCTUATION_STR_NO_PERIOD}"
PUNCTUATION_PATTERN = rf"[{re.escape(PUNCTUATION_STR)}]+"
PUNCTUATION_START = re.compile(rf"^{PUNCTUATION_PATTERN}")
PUNCTUATION_END = re.compile(rf"{PUNCTUATION_PATTERN}$")
PUNCTUATION_END_SPACE = re.compile(rf"{PUNCTUATION_PATTERN}\s*$")
PUNCTUATION_START_WORD = re.compile(rf"(?<=\W){PUNCTUATION_PATTERN}(?=\w)")
PUNCTUATION_END_WORD = re.compile(rf"(?<=\w){PUNCTUATION_PATTERN_NO_PERIOD}(?=\W)")
PUNCTUATION_END_PERIOD = re.compile(r"(?<!\w\.\w)\.(?=\W)")  # ignore initialisms
PUNCTUATION_WORD = re.compile(rf"(?<=\W){PUNCTUATION_PATTERN}(?=\W)")

INITIALISM_DOTS_AT_END = re.compile(r"\b(?:\w\.){2,}$")


def merge_dict(
    base_dict: MutableMapping[Any, Any], new_dict: Mapping[Any, Any]
) -> None:
    """Merge new_dict into base_dict."""
    for key, value in new_dict.items():
        if key in base_dict:
            old_value = base_dict[key]
            if isinstance(old_value, MutableMapping):
                # Combine dictionary
                assert isinstance(value, Mapping), f"Not a dict: {value}"
                merge_dict(old_value, value)
            elif isinstance(old_value, MutableSequence):
                # Combine list
                assert isinstance(value, Sequence), f"Not a list: {value}"
                old_value.extend(value)
            else:
                # Overwrite
                base_dict[key] = value
        else:
            base_dict[key] = value


def remove_escapes(text: str) -> str:
    """Remove backslash escape sequences."""
    return re.sub(r"\\(.)", r"\1", text)


def normalize_whitespace(text: str) -> str:
    """Make all whitespace inside a string single spaced."""
    return WHITESPACE_CAPTURE.sub(WHITESPACE_SEPARATOR, text)


def normalize_text(text: str) -> str:
    """Normalize whitespace and unicode forms."""
    text = normalize_whitespace(text)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("’", "'")

    return text


def is_template(text: str) -> bool:
    """Return True if text contains template syntax."""
    return TEMPLATE_SYNTAX.match(text) is not None


def check_required_context(
    required_context: Dict[str, Any],
    match_context: Optional[Dict[str, Any]],
    allow_missing_keys: bool = False,
) -> bool:
    """Return True if match context does not violate required context.

    Setting allow_missing_keys to True only checks existing keys in match
    context.
    """
    for (
        required_key,
        required_value,
    ) in required_context.items():
        if (not match_context) or (required_key not in match_context):
            # Match is missing key
            if allow_missing_keys:
                # Only checking existing keys
                continue

            return False

        if isinstance(required_value, Mapping):
            # Unpack dict
            # <context_key>:
            #   value: ...
            required_value = required_value.get("value")

        # Ensure value matches
        actual_value = match_context[required_key]

        if isinstance(actual_value, Mapping):
            # Unpack dict
            # <context_key>:
            #   value: ...
            actual_value = actual_value.get("value")

        if (not isinstance(required_value, str)) and isinstance(
            required_value, Collection
        ):
            if actual_value not in required_value:
                # Match value not in required list
                return False
        elif (required_value is not None) and (actual_value != required_value):
            # Match value doesn't equal required value
            return False

    return True


def check_excluded_context(
    excluded_context: Dict[str, Any], match_context: Optional[Dict[str, Any]]
) -> bool:
    """Return True if match context does not violate excluded context."""
    for (
        excluded_key,
        excluded_value,
    ) in excluded_context.items():
        if (not match_context) or (excluded_key not in match_context):
            continue

        if isinstance(excluded_value, Mapping):
            # Unpack dict
            # <context_key>:
            #   value: ...
            excluded_value = excluded_value.get("value")

        # Ensure value does not match
        actual_value = match_context[excluded_key]

        if isinstance(actual_value, Mapping):
            # Unpack dict
            # <context_key>:
            #   value: ...
            actual_value = actual_value.get("value")

        if (not isinstance(excluded_value, str)) and isinstance(
            excluded_value, Collection
        ):
            if actual_value in excluded_value:
                # Match value is in excluded list
                return False
        elif actual_value == excluded_value:
            # Match value equals excluded value
            return False

    return True


def remove_skip_words(
    text: str, skip_words: Iterable[str], ignore_whitespace: bool
) -> str:
    """Remove all skip words from text."""
    if not skip_words:
        return text

    if ignore_whitespace:
        skip_words_pattern = re.compile(
            r"("
            + "|".join(
                re.escape(w.strip()) for w in sorted(skip_words, key=len, reverse=True)
            )
            + r")",
            re.IGNORECASE,
        )
        return skip_words_pattern.sub("", text)

    skip_words_pattern = re.compile(
        r"(?<=\W)("
        + "|".join(
            re.escape(w.strip()) for w in sorted(skip_words, key=len, reverse=True)
        )
        + r")(?=\W)",
        re.IGNORECASE,
    )
    text = skip_words_pattern.sub(" ", f" {text} ").strip()
    return normalize_whitespace(text)


def remove_punctuation(text: str) -> str:
    """Remove punctuation from start/end of words and entire text."""
    text = PUNCTUATION_START.sub("", text)

    if not INITIALISM_DOTS_AT_END.match(text):
        # Don't remove final "." from "A.C.", etc.
        text = PUNCTUATION_END.sub("", text)

    text = PUNCTUATION_START_WORD.sub("", text)
    text = PUNCTUATION_END_WORD.sub("", text)
    text = PUNCTUATION_END_PERIOD.sub("", text)
    text = PUNCTUATION_WORD.sub("", text)

    return text


def match_start(text: str, prefix: str) -> Optional[int]:
    """Match prefix at start of text and return end of match position."""
    match = re.match(rf"^{re.escape(prefix)}", text, re.IGNORECASE)
    if match is None:
        return None

    return match.end()


def match_first(
    text: str, prefix: str, start_idx: int = 0, start_of_word: bool = False
) -> int:
    """Match prefix at text or word boundary and return start of match position."""
    if start_idx > 0:
        text = text[start_idx:]

    if start_of_word:
        boundary = r"\b"
    else:
        boundary = ""

    match = re.search(rf"{boundary}{re.escape(prefix)}", text, re.IGNORECASE)
    if match is None:
        return -1

    return start_idx + match.start()
