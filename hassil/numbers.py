"""Number formatting engines for range lists."""

from typing import Dict, Iterable

from unicode_rbnf import RbnfEngine

_ENGINE_CACHE: Dict[str, RbnfEngine] = {}

# Languages whose script is implied by their region, following CLDR
# likelySubtags. Without these, a region code falls back to the bare language
# and picks up the wrong script: zh-TW would load Simplified rules and only
# recognize 两 instead of 兩.
#
# Keys are lower-cased and use "_" as the separator.
_SCRIPT_ALIASES = {
    "zh_hk": "zh_Hant",
    "zh_mo": "zh_Hant",
    "zh_tw": "zh_Hant",
}


def _candidate_languages(language: str) -> Iterable[str]:
    """Generate language codes to try, from most to least specific."""
    yield language

    # unicode-rbnf names its files with "_" (sr_Latn, de_CH)
    normalized = language.replace("-", "_")
    if normalized != language:
        yield normalized

    alias = _SCRIPT_ALIASES.get(normalized.lower())
    if alias is not None:
        yield alias

    # Fall back to the language family, e.g. "en" for "en-US"
    family = normalized.split("_", maxsplit=1)[0]
    if family != normalized:
        yield family


def get_rbnf_engine(language: str) -> RbnfEngine:
    """Get a number formatting engine for a language.

    Raises ValueError if no engine is available.
    """
    engine = _ENGINE_CACHE.get(language)
    if engine is not None:
        return engine

    last_error: ValueError = ValueError(f"{language} is not supported")
    for candidate in _candidate_languages(language):
        try:
            engine = RbnfEngine.for_language(candidate)
        except ValueError as err:
            last_error = err
            continue

        _ENGINE_CACHE[language] = engine
        return engine

    raise last_error
