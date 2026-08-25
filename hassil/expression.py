"""Classes for representing sentence templates."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, Iterator, List, Optional, Set, Tuple

from .util import PUNCTUATION_STR

INLINE_RANGE_PATTERN = re.compile(r"^(\d+)..(\d+)(?:,(\d+))?$")

_PUNCTUATION_CHARS = frozenset(PUNCTUATION_STR)

# Guard against expansion rules that reference each other.
_MAX_CLAUSE_DEPTH = 24


@dataclass
class Expression:
    """Base class for expressions."""


@dataclass
class TextChunk(Expression):
    """Contiguous chunk of text (with whitespace)."""

    # Text with casing/whitespace normalized
    text: str = ""

    # Set in __post_init__
    original_text: str = None  # type: ignore

    parent: "Optional[Group]" = None

    def __post_init__(self):
        if self.original_text is None:
            self.original_text = self.text

    @property
    def is_empty(self) -> bool:
        """True if the chunk is empty"""
        return self.text == ""

    @staticmethod
    def empty() -> TextChunk:
        """Returns an empty text chunk"""
        return TextChunk()


@dataclass
class Group(Expression):
    """Ordered group of expressions. Supports sequences, optionals, and alternatives."""

    # Items in the group
    items: List[Expression] = field(default_factory=list)

    def text_chunk_count(self) -> int:
        """Return the number of TextChunk expressions in this group (recursive)."""
        num_text_chunks = 0
        for item in self.items:
            if isinstance(item, TextChunk):
                num_text_chunks += 1
            elif isinstance(item, Group):
                grp: Group = item
                num_text_chunks += grp.text_chunk_count()

        return num_text_chunks

    def list_references(
        self,
        expansion_rules: Optional[Dict[str, Sentence]] = None,
    ) -> Iterator[ListReference]:
        """Return names of list references (recursive)."""
        for item in self.items:
            yield from self._list_refs(item, expansion_rules)

    def list_names(
        self,
        expansion_rules: Optional[Dict[str, Sentence]] = None,
    ) -> Iterator[str]:
        """Return names of list references (recursive)."""
        for list_ref in self.list_references(expansion_rules):
            yield list_ref.list_name

    def _list_refs(
        self,
        item: Expression,
        expansion_rules: Optional[Dict[str, Sentence]] = None,
    ) -> Iterator[ListReference]:
        """Return names of list references (recursive)."""
        if isinstance(item, ListReference):
            list_ref: ListReference = item
            yield list_ref
        elif isinstance(item, Group):
            grp: Group = item
            yield from grp.list_references(expansion_rules)
        elif isinstance(item, RuleReference):
            rule_ref: RuleReference = item
            if expansion_rules and (rule_ref.rule_name in expansion_rules):
                rule_body = expansion_rules[rule_ref.rule_name].expression
                yield from self._list_refs(rule_body, expansion_rules)


@dataclass
class Sequence(Group):
    """Sequence of expressions."""


@dataclass
class Alternative(Group):
    """Expressions where only one will be recognized."""

    is_optional: bool = False


@dataclass
class Permutation(Group):
    """Permutations of a set of expressions."""

    def iterate_permutations(self) -> Iterable[Tuple[Expression, Permutation]]:
        """Iterate over all permutations."""
        for i, item in enumerate(self.items):
            items = self.items.copy()
            del items[i]
            rest = Permutation(items=items)
            yield (item, rest)


@dataclass
class RuleReference(Expression):
    """Reference to an expansion rule by <name>."""

    # Name of referenced rule
    rule_name: str = ""


@dataclass
class ListReference(Expression):
    """Reference to a list by {name}."""

    list_name: str = ""
    prefix: Optional[str] = None
    suffix: Optional[str] = None
    is_end_of_word: bool = True
    is_capture: bool = False
    _slot_name: Optional[str] = None
    _inline_range_match: Optional[re.Match] = None

    def __post_init__(self):
        if ":" in self.list_name:
            # list_name:slot_name
            self.list_name, self._slot_name = self.list_name.split(":", maxsplit=1)
            if self._slot_name.startswith("@"):
                # Capture (only available in response)
                self.is_capture = True
                self._slot_name = self._slot_name[1:]

            self._inline_range_match = INLINE_RANGE_PATTERN.match(self.list_name)
        elif self.list_name.startswith("@"):
            # Compact capture syntax
            # {@x} is the same as {x:@x}
            self.list_name = self.list_name[1:]
            self._slot_name = self.list_name
            self.is_capture = True
        else:
            self._slot_name = self.list_name

    @property
    def slot_name(self) -> str:
        """Name of slot to put list value into."""
        assert self._slot_name is not None
        return self._slot_name

    @property
    def is_inline_range(self) -> bool:
        return self._inline_range_match is not None

    def get_inline_range(self) -> Optional[Tuple[int, int, int]]:
        if not self._inline_range_match:
            return None

        start, stop, step = self._inline_range_match.groups()
        return (int(start), int(stop), 1 if step is None else int(step))


@dataclass
class Sentence:
    """A complete sentence template."""

    expression: Expression
    text: Optional[str] = None
    pattern: Optional[re.Pattern] = None

    required_clauses: Optional[FrozenSet[FrozenSet[str]]] = field(
        default=None, repr=False, compare=False
    )
    """Literal fragments this template requires (see get_required_clauses)."""

    def get_required_clauses(
        self, expansion_rules: Dict[str, Sentence]
    ) -> FrozenSet[FrozenSet[str]]:
        """Return literal fragments that any matching input text must contain.

        The result is in conjunctive normal form: a set of clauses, where the
        input must contain at least one fragment from *every* clause. An empty
        result means the template cannot be pre-filtered.

        This is a necessary (not sufficient) condition for a match, so it is only
        useful to skip templates that cannot possibly match. It is deliberately
        conservative -- a fragment is only emitted when its presence is
        guaranteed:

        - Fragments are split on whitespace, so the matcher's break-words
          fallback ("living-room" matching "living room") cannot cause a miss.
        - Fragments containing punctuation that ``remove_punctuation`` may strip
          from the input are dropped.
        - Fragments are casefolded. ``str.casefold`` is at least as permissive as
          the ``re.IGNORECASE`` matching used for text chunks, so the test can
          only ever be too lenient, never too strict.

        Note: like ``compile``, the result is cached on first use, so a given
        Sentence object is assumed to always be used with the same expansion
        rules.
        """
        if self.required_clauses is None:
            self.required_clauses = frozenset(
                _required_clauses(self.expression, expansion_rules)
            )

        return self.required_clauses

    def text_chunk_count(self) -> int:
        """Return the number of TextChunk expressions in this sentence."""
        assert isinstance(self.expression, Group)
        return self.expression.text_chunk_count()  # pylint: disable=no-member

    def list_names(
        self,
        expansion_rules: Optional[Dict[str, Sentence]] = None,
    ) -> Iterator[str]:
        """Return names of list references in this sentence."""
        assert isinstance(self.expression, Group)
        return self.expression.list_names(expansion_rules)  # pylint: disable=no-member

    def compile(self, expansion_rules: Dict[str, Sentence]) -> None:
        if self.pattern is not None:
            # Already compiled
            return

        pattern_chunks: List[str] = []
        self._compile_expression(self.expression, pattern_chunks, expansion_rules)
        pattern_str = "".join(pattern_chunks).replace(r"\ ", r"[ ]*")
        self.pattern = re.compile(f"^{pattern_str}$", re.IGNORECASE)

    def _compile_expression(
        self, exp: Expression, pattern_chunks: List[str], rules: Dict[str, Sentence]
    ) -> None:
        if isinstance(exp, TextChunk):
            # Literal text
            chunk: TextChunk = exp
            if chunk.text:
                escaped_text = re.escape(chunk.text)
                pattern_chunks.append(escaped_text)
        elif isinstance(exp, Group):
            grp: Group = exp
            if isinstance(grp, Sequence):
                for item in grp.items:
                    self._compile_expression(item, pattern_chunks, rules)
            elif isinstance(grp, Alternative):
                if grp.items:
                    pattern_chunks.append("(?:")
                    for item in grp.items:
                        self._compile_expression(item, pattern_chunks, rules)
                        pattern_chunks.append("|")
                    pattern_chunks[-1] = ")"
            elif isinstance(grp, Permutation):
                if grp.items:
                    pattern_chunks.append("(?:")
                    for item in grp.items:
                        self._compile_expression(item, pattern_chunks, rules)
                        pattern_chunks.append("|")
                    pattern_chunks[-1] = f"){{{len(grp.items)}}}"
            else:
                raise ValueError(grp)
        elif isinstance(exp, ListReference):
            # Slot list
            pattern_chunks.append("(?:.+)")

        elif isinstance(exp, RuleReference):
            # Expansion rule
            rule_ref: RuleReference = exp
            if rule_ref.rule_name not in rules:
                raise ValueError(rule_ref)

            e_rule = rules[rule_ref.rule_name]
            self._compile_expression(e_rule.expression, pattern_chunks, rules)
        else:
            raise ValueError(exp)


def _text_clauses(text: str) -> Set[FrozenSet[str]]:
    """Return one single-fragment clause per usable whitespace-separated piece."""
    clauses: Set[FrozenSet[str]] = set()
    for piece in text.split():
        if _PUNCTUATION_CHARS.isdisjoint(piece):
            clauses.add(frozenset((piece.casefold(),)))

    return clauses


def _required_clauses(
    exp: Expression, rules: Dict[str, "Sentence"], depth: int = 0
) -> Set[FrozenSet[str]]:
    """Compute required literal fragments in CNF. See Sentence.get_required_clauses."""
    if depth > _MAX_CLAUSE_DEPTH:
        return set()

    if isinstance(exp, TextChunk):
        return _text_clauses(exp.text)

    if isinstance(exp, Alternative):
        # [optional] is (optional|), which constrains nothing.
        if exp.is_optional or (not exp.items):
            return set()

        per_branch = [_required_clauses(item, rules, depth + 1) for item in exp.items]
        if any(not branch for branch in per_branch):
            # A branch requires nothing, so it could match without any fragment.
            return set()

        # Exactly one branch matches, so at least one fragment from one branch is
        # present. Pick the smallest clause per branch and OR them together.
        merged: Set[str] = set()
        for branch in per_branch:
            smallest = next(iter(branch))
            for clause in branch:
                if len(clause) < len(smallest):
                    smallest = clause

            merged.update(smallest)

        return {frozenset(merged)}

    if isinstance(exp, Group):
        # Sequence and Permutation both require every item to match.
        clauses: Set[FrozenSet[str]] = set()
        for item in exp.items:
            clauses.update(_required_clauses(item, rules, depth + 1))

        return clauses

    if isinstance(exp, RuleReference):
        rule = rules.get(exp.rule_name)
        if rule is None:
            return set()

        return _required_clauses(rule.expression, rules, depth + 1)

    # ListReference and anything else: contents are unknown.
    return set()
