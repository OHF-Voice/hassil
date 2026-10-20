"""Specialized implementation of a trie.

See: https://en.wikipedia.org/wiki/Trie
"""

import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# re.IGNORECASE treats "I", "i", dotless "ı" (U+0131) and dotted "İ" (U+0130) as
# the same letter, but str.lower maps "I" to "i" (never "ı") and expands "İ" to
# two characters ("i" + U+0307). See _fold_case.
_TURKISH_I_TABLE = str.maketrans({"\u0130": "i", "\u0131": "i"})


@dataclass
class TrieNode:
    """Node in trie."""

    id: int
    text: Optional[str] = None
    values: Optional[List[Any]] = None
    children: "Optional[Dict[str, TrieNode]]" = None


class Trie:
    """A specialized trie data structure that finds all known words in a string."""

    def __init__(self) -> None:
        self.roots: Dict[str, TrieNode] = {}
        self._next_id = 0

        # The same words keyed on _fold_case, built on demand by find
        self._folded_roots: Optional[Dict[str, TrieNode]] = None

    def insert(self, text: str, value: Any) -> None:
        """Insert a word and value into the trie."""
        self._folded_roots = None
        self._insert(self.roots, text, text, value)

    def _insert(
        self, roots: Dict[str, TrieNode], key: str, text: str, value: Any
    ) -> None:
        """Insert value under key, recording text as the word that was found."""
        current_node: Optional[TrieNode] = None
        current_children: Optional[Dict[str, TrieNode]] = roots

        last_idx = len(key) - 1
        for i, c in enumerate(key):
            if current_children is None:
                assert current_node is not None
                current_node.children = current_children = {}

            current_node = current_children.get(c)
            if current_node is None:
                current_node = TrieNode(id=self.next_id())
                current_children[c] = current_node

            if i == last_idx:
                current_node.text = text
                if current_node.values is None:
                    current_node.values = [value]
                else:
                    current_node.values.append(value)

            current_children = current_node.children

    def find(
        self,
        text: str,
        unique: bool = True,
        word_boundaries: bool = False,
        ignore_case: bool = False,
    ) -> Iterable[Tuple[int, str, Any]]:
        """Yield (end_pos, text, value) pairs of all words found in the string.

        With ignore_case, both the words and the string are folded with
        _fold_case first. Since that never changes the length, end_pos is still a
        position in the original string, and text is the word as it was inserted.
        """
        roots = self.roots
        if ignore_case:
            roots = self._get_folded_roots()
            text = _fold_case(text)

        visited: Set[int] = set()
        for i in range(len(text)):
            if word_boundaries and (not _is_word_boundary(text, i)):
                continue

            current_children = roots
            current_position = i

            while current_position < len(text):
                node = current_children.get(text[current_position])
                if node is None:
                    break

                match_end = current_position + 1

                if (
                    (node.text is not None)
                    and ((not word_boundaries) or _is_end_boundary(text, match_end))
                    and ((not unique) or (node.id not in visited))
                ):
                    if unique:
                        visited.add(node.id)

                    for value in node.values or [None]:
                        yield (match_end, node.text, value)

                current_children = node.children or {}
                current_position += 1

    def find_prefixes(self, text: str) -> Iterable[Tuple[int, str, Any]]:
        """Yield (end_pos, text, value) for every key that is a prefix of text.

        Unlike ``find``, this only walks from the start of the string, so it costs
        O(length of the longest matching key) instead of scanning the whole text.
        """
        current_children: Optional[Dict[str, TrieNode]] = self.roots

        for i, current_char in enumerate(text):
            if current_children is None:
                return

            node = current_children.get(current_char)
            if node is None:
                return

            if node.text is not None:
                for value in node.values or [None]:
                    yield (i + 1, node.text, value)

            current_children = node.children

    def _get_folded_roots(self) -> Dict[str, TrieNode]:
        """Return roots with every word in the trie inserted under _fold_case."""
        folded_roots = self._folded_roots
        if folded_roots is None:
            folded_roots = {}
            nodes = list(self.roots.values())
            while nodes:
                node = nodes.pop()
                if node.text is not None:
                    folded_text = _fold_case(node.text)
                    for value in node.values or []:
                        self._insert(folded_roots, folded_text, node.text, value)

                if node.children:
                    nodes.extend(node.children.values())

            self._folded_roots = folded_roots

        return folded_roots

    def next_id(self) -> int:
        current_id = self._next_id
        self._next_id += 1
        return current_id


def _fold_case(text: str) -> str:
    """Fold case for find(ignore_case=True) without changing the length of text.

    find reports positions in the text it was given, so the folded text must line
    up with the original character for character. Once the Turkish "İ" and "ı"
    are mapped to "i", str.lower keeps the length of every string (str.casefold
    does not: "ß" becomes "ss").
    """
    return text.translate(_TURKISH_I_TABLE).lower()


def _is_boundary_category(text: str) -> bool:
    """Return True if text Unicode category is a word boundary."""
    text_category = unicodedata.category(text)
    if not text_category:
        return False

    # punctuation
    return text_category[0] == "P"


def _is_word_boundary(text: str, index: int) -> bool:
    """Return True if the character at `index` is at a valid word boundary."""
    if index == 0:
        # Start of text
        return True

    prev = text[index - 1]
    return prev.isspace() or _is_boundary_category(prev)


def _is_end_boundary(text: str, index: int) -> bool:
    """Return True if the character at `index` is at a valid end word boundary."""
    return (
        (index == len(text))  # end of text
        or text[index].isspace()
        or _is_boundary_category(text[index])
    )
