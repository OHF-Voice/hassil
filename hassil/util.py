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
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional

WHITESPACE = re.compile(r"\s+")
WHITESPACE_CAPTURE = re.compile(r"(\s+)")
WHITESPACE_SEPARATOR = " "

# Scripts without inter-word spaces (CJK ideographs, Japanese kana, Korean
# hangul). When ``ignore_whitespace`` is set, whitespace is only removed
# *between* two of these characters. Whitespace at a CJK/non-CJK boundary or
# between non-CJK characters is preserved so that values like song titles
# ("Taylor Swift") keep their internal spaces instead of being glued together.
CJK = (
    "一-鿿"  # CJK Unified Ideographs
    "㐀-䶿"  # CJK Unified Ideographs Extension A
    "぀-ヿ"  # Hiragana + Katakana
    "가-힯"  # Hangul Syllables
)
CJK_WHITESPACE = re.compile(rf"(?<=[{CJK}])\s+(?=[{CJK}])")

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


class TrackedText:
    """Text plus a map from each character back to its index in the source text.

    Matching happens on text that has had punctuation and skip words removed and
    whitespace normalized, so entity spans come out in the coordinates of that
    derived text. Tracking offsets while the text is transformed is the only
    reliable way to report spans against what the user actually typed: realigning
    afterwards is ambiguous, because a derived word can also occur earlier in the
    source (removing "the" from "run the test" leaves "run test", whose "test"
    aligns just as well onto "the").
    """

    __slots__ = ("text", "offsets")

    def __init__(self, text: str, offsets: Optional[List[int]] = None) -> None:
        self.text = text
        self.offsets = list(range(len(text))) if offsets is None else offsets

    def copy(self) -> "TrackedText":
        """Return an independent copy."""
        return TrackedText(self.text, list(self.offsets))

    def sub(self, pattern: "re.Pattern[str]", repl: str = "") -> None:
        """Replace every match, keeping offsets aligned.

        A non-empty replacement is attributed to the start of the text it
        replaced, which is what a caller highlighting the source wants.
        """
        parts: List[str] = []
        offsets: List[int] = []
        pos = 0

        for match in pattern.finditer(self.text):
            start, end = match.span()
            parts.append(self.text[pos:start])
            offsets.extend(self.offsets[pos:start])

            if repl:
                source_idx = self.offsets[start] if start < len(self.offsets) else start
                parts.append(repl)
                offsets.extend([source_idx] * len(repl))

            pos = end

        if not parts:
            # No matches, nothing to rebuild.
            return

        parts.append(self.text[pos:])
        offsets.extend(self.offsets[pos:])

        self.text = "".join(parts)
        self.offsets = offsets

    def replace_char(self, old: str, new: str) -> None:
        """Replace single characters one-for-one (offsets are unaffected)."""
        assert len(old) == len(new) == 1
        self.text = self.text.replace(old, new)

    def strip(self) -> None:
        """Strip leading/trailing whitespace."""
        stripped = self.text.strip()
        if stripped == self.text:
            return

        lead = len(self.text) - len(self.text.lstrip())
        self.offsets = self.offsets[lead : lead + len(stripped)]
        self.text = stripped

    def append(self, extra: str) -> None:
        """Append text that has no counterpart in the source."""
        if not extra:
            return

        end = (self.offsets[-1] + 1) if self.offsets else 0
        self.offsets = self.offsets + [end] * len(extra)
        self.text += extra

    def normalize_unicode(self) -> None:
        """Apply NFC normalization."""
        new_text = unicodedata.normalize("NFC", self.text)
        if new_text == self.text:
            return

        if len(new_text) != len(self.text):
            # Composition merged characters. This is vanishingly rare for command
            # text; keep the map usable by padding/truncating rather than letting
            # it desynchronize with the text length.
            offsets = self.offsets[: len(new_text)]
            last = (offsets[-1] + 1) if offsets else 0
            offsets.extend([last] * (len(new_text) - len(offsets)))
            self.offsets = offsets

        self.text = new_text


def normalize_whitespace(text: str) -> str:
    """Make all whitespace inside a string single spaced."""
    return WHITESPACE_CAPTURE.sub(WHITESPACE_SEPARATOR, text)


def _normalize_text(tracked: TrackedText) -> None:
    """Normalize whitespace and unicode forms in place."""
    tracked.sub(WHITESPACE_CAPTURE, WHITESPACE_SEPARATOR)
    tracked.normalize_unicode()
    tracked.replace_char("’", "'")


def normalize_text(text: str) -> str:
    """Normalize whitespace and unicode forms."""
    tracked = TrackedText(text)
    _normalize_text(tracked)

    return tracked.text


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


def remove_skip_words_tracked(
    tracked: TrackedText,
    skip_words: Iterable[str],
    ignore_whitespace: bool,
    start: bool = True,
    end: bool = True,
) -> None:
    """Remove all skip words from tracked text, in place."""
    words = sorted({w.strip() for w in skip_words if w.strip()}, key=len, reverse=True)
    if not words:
        return

    skip_words_str = "|".join(re.escape(w) for w in words)

    if ignore_whitespace:
        if start and end:
            tracked.sub(re.compile(rf"(?:{skip_words_str})", re.IGNORECASE))
            return

        if start:
            pattern = re.compile(rf"^(?:{skip_words_str})", re.IGNORECASE)
        else:
            pattern = re.compile(rf"(?:{skip_words_str})$", re.IGNORECASE)

        while True:
            previous_text = tracked.text
            tracked.sub(pattern)
            if tracked.text == previous_text:
                break

        return

    # Whitespace-sensitive mode.
    if start and end:
        # Remove skip words anywhere, but only as separated words/phrases.
        tracked.sub(
            re.compile(rf"(?<!\w)(?:{skip_words_str})(?!\w)", re.IGNORECASE),
            WHITESPACE_SEPARATOR,
        )
        tracked.sub(WHITESPACE_CAPTURE, WHITESPACE_SEPARATOR)
        tracked.strip()
        return

    if start:
        pattern = re.compile(
            rf"^\s*(?:{skip_words_str})(?=\s|$|[^\w])\s*",
            re.IGNORECASE,
        )
    else:
        pattern = re.compile(
            rf"\s*(?<!\w)(?:{skip_words_str})\s*$",
            re.IGNORECASE,
        )

    while True:
        previous_text = tracked.text
        tracked.sub(pattern)
        tracked.sub(WHITESPACE_CAPTURE, WHITESPACE_SEPARATOR)
        tracked.strip()

        if tracked.text == previous_text:
            break


def remove_skip_words(
    text: str,
    skip_words: Iterable[str],
    ignore_whitespace: bool,
    start: bool = True,
    end: bool = True,
) -> str:
    """Remove all skip words from text."""
    tracked = TrackedText(text)
    remove_skip_words_tracked(
        tracked, skip_words, ignore_whitespace, start=start, end=end
    )

    return tracked.text


def _remove_punctuation(tracked: TrackedText) -> None:
    """Remove punctuation from start/end of words and entire text, in place."""
    tracked.sub(PUNCTUATION_START)

    if not INITIALISM_DOTS_AT_END.search(tracked.text):
        # Don't remove final "." from "A.C.", etc. Use search (not match) so a
        # trailing initialism is preserved even at the end of a longer string,
        # keeping normalization consistent with a standalone initialism.
        tracked.sub(PUNCTUATION_END)

    tracked.sub(PUNCTUATION_START_WORD)
    tracked.sub(PUNCTUATION_END_WORD)
    tracked.sub(PUNCTUATION_END_PERIOD)
    tracked.sub(PUNCTUATION_WORD)


def remove_punctuation(text: str) -> str:
    """Remove punctuation from start/end of words and entire text."""
    tracked = TrackedText(text)
    _remove_punctuation(tracked)

    return tracked.text


def normalize_for_matching(text: str) -> TrackedText:
    """Return match-ready text with a map back into the original text."""
    tracked = TrackedText(text)
    _remove_punctuation(tracked)
    _normalize_text(tracked)
    tracked.strip()

    return tracked


def _is_same_source_char(original_char: str, derived_char: str) -> bool:
    """Return True if derived_char could have come from original_char."""
    if original_char == derived_char:
        return True

    # normalize_whitespace() collapses any whitespace run to a single space, and
    # remove_skip_words() substitutes a space for the words it drops.
    if (derived_char == WHITESPACE_SEPARATOR) and original_char.isspace():
        return True

    # normalize_text() rewrites the typographic apostrophe.
    return (derived_char == "'") and (original_char == "’")


def build_offset_map(original: str, derived: str) -> List[int]:
    """Map each index in derived text to the index in original it came from.

    Matching runs against text that has been punctuation-stripped, whitespace
    normalized and had skip words removed, so entity spans are in the coordinates
    of that derived text. Callers that report spans against the text the user
    actually typed need to translate them back.

    Every step that produces the derived text either deletes characters, collapses
    a whitespace run to a single space, or substitutes a character of equal length.
    Under those operations a greedy left-to-right alignment recovers the source
    position of each derived character.

    Trailing entries point at len(original) if alignment runs out of input (for
    example the artificial word boundary appended to the match text).
    """
    offsets: List[int] = []
    original_len = len(original)
    original_idx = 0

    for derived_char in derived:
        while (original_idx < original_len) and (
            not _is_same_source_char(original[original_idx], derived_char)
        ):
            original_idx += 1

        if original_idx >= original_len:
            # Ran out of original text; pin the remainder to the end.
            offsets.extend([original_len] * (len(derived) - len(offsets)))
            return offsets

        offsets.append(original_idx)
        original_idx += 1

    return offsets


@lru_cache(maxsize=8192)
def _compiled_prefix(prefix: str, boundary: str) -> "re.Pattern[str]":
    """Cache the compiled prefix pattern to avoid re-escaping/re-compiling per match."""
    return re.compile(rf"{boundary}{re.escape(prefix)}", re.IGNORECASE)


def match_start(text: str, prefix: str) -> Optional[int]:
    """Match prefix at start of text and return end of match position."""
    match = _compiled_prefix(prefix, "^").match(text)
    if match is None:
        return None

    return match.end()


def match_first(
    text: str, prefix: str, start_idx: int = 0, start_of_word: bool = False
) -> int:
    """Match prefix at text or word boundary and return start of match position."""
    if start_idx > 0:
        text = text[start_idx:]

    boundary = r"\b" if start_of_word else ""

    match = _compiled_prefix(prefix, boundary).search(text)
    if match is None:
        return -1

    return start_idx + match.start()
