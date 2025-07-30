"""Specialized implementation of a trie.

See: https://en.wikipedia.org/wiki/Trie
"""

import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


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

    def insert(self, text: str, value: Any) -> None:
        """Insert a word and value into the trie."""
        current_node: Optional[TrieNode] = None
        current_children: Optional[Dict[str, TrieNode]] = self.roots

        last_idx = len(text) - 1
        for i, c in enumerate(text):
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
        self, text: str, unique: bool = True, word_boundaries: bool = False
    ) -> Iterable[Tuple[int, str, Any]]:
        """Yield (end_pos, text, value) pairs of all words found in the string."""
        visited: Set[int] = set()

        for i in range(len(text)):
            if word_boundaries and (not _is_word_boundary(text, i)):
                continue

            current_children = self.roots
            current_position = i

            while current_position < len(text):
                current_char = text[current_position]
                node = current_children.get(current_char)
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

    def next_id(self) -> int:
        current_id = self._next_id
        self._next_id += 1
        return current_id


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
