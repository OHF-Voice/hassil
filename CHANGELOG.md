# Changelog

## 3.1.0

- Add fuzzy n-gram matcher (beta)

## 3.0.1

- Use faster permutation iteration

## 3.0.0

- Rename `Sequence` to `Group`
- Remove `SequenceType` and instead of types for each `Group`:
    - `Sequence` for `(a b c)`
    - `Alternative` for `(a|b|c)`
    - `Permutation` for `(a;b;c)`
- `Sentence` now contains an `expression` field with the parsed expression
- `Permutation` is represented explicitly instead of being expanded internally into an alternative
- Everything used by downstream projects has been moved into a top-level import (`from hassil import Intents`). The internal structure of of hassil (e.g., `hassil.intents` or `hassil.util`) should be expected to change.
- Drop support for Python 3.8 (EOL October 7, 2024)

## 2.2.3

- Fix behavior with wildcards inside and outside words

## 2.2.2

- Allow "," as a decimal separator for fractional ranges

## 2.2.1

- Allow list values with "in" but no "out"

## 2.2.0

- Add "fractions" to number ranges with halves and tenths
- Don't remove punctuation within words in `text_clean` (e.g., "2.5")

## 2.1.1

- Allow number ranges to have the same start/stop (single number)

## 2.1.0

- Upgrade to `unicode-rbnf` 2.2
- Transition to pyproject.toml

## 2.0.4

- Trie values are accumulated on `insert` instead of being overwritten

## 2.0.3

- Make trie more restrictive (`two` will not match `t|wo`)

## 2.0.2

- Require `unicode-rbnf>=2.1` which includes important bugfixes

## 2.0.1

- Count stripped text in `text_chunks_matched`

## 2.0.0

- Allow wildcards to be followed by expansion rules and lists
- Use regular expressions to filter sentence templates
- Add `filter_with_regex` to intent settings and intent data (`false` disables regex filtering)
- Filter text slot list values by required/excluded context during matching
- Use a trie to filter range slot list values based on remaining text to be matched
- Add `required_keywords` section to intent data to skip sentences without specific keywords
- Preserve case during matching 
- Strip punctuation before text processing
- Remove extraneous whitespace from the end of wildcards
- Refactor string matching code into `string_matcher.py`

## 1.8.0

- Bump `unicode-rbnf` to 2.0.0
- Use multiple texts for numbers, e.g. for German 1 `ein`, `eins`, etc.
- Remove `words_ruleset` for ranges

## 1.7.4

- Loosen `unicode-rbnf` version

## 1.7.3

- Cache number words

## 1.7.2

- Add apostrophe to punctuation list

## 1.7.1

- Fix `is_wildcard` in match entities
- Fix `wildcard_text` initialization
- Bump pylint to 3.1.0

## 1.7.0

- Add multiplier to range

## 1.6.1

- Allow context values to be dicts

## 1.6.0

- Add metadata to sentences
- Add metadata to list items

## 1.5.3

- Restrict unmatched entities to contiguous blocks of non-literal text
- Automatically use intents language for number words if supported

## 1.5.2

- Add local slots (under data sentences)
- Add literal text chunk count and matched sentence template to results

## 1.5.1

- Expand `requires_context` to allow copying value to a slot

## 1.5.0

- Add fuzzy matching using edit distance

## 1.4.0

- Sort wildcard sentences so they are processed by most literal text chunks first

## 1.3.0

- Add number to word generation using [unicode-rbnf](https://github.com/rhasspy/unicode-rbnf) for range lists

## 1.2.5

- Fix degenerate wildcard case
