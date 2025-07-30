"""Convert hassil templates to finite state transducers (FSTs)."""

import itertools
from collections import defaultdict, deque
from collections.abc import Collection
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Deque, Dict, List, Optional, Set, TextIO, Tuple

from hassil import (
    Alternative,
    Expression,
    Group,
    IntentData,
    Intents,
    ListReference,
    Permutation,
    RangeSlotList,
    RuleReference,
    Sentence,
    Sequence,
    SlotList,
    TextChunk,
    TextSlotList,
)

EPS = "<eps>"
SPACE = "<space>"
UNK = "<unk>"


class SuppressOutput(Enum):
    DISABLED = auto()
    UNTIL_END = auto()
    UNTIL_SPACE = auto()


@dataclass
class ListReferenceNode:
    list_names: List[Tuple[str, Optional[Set[str]]]] = field(default_factory=list)
    children: "List[ListReferenceNode]" = field(default_factory=list)


@dataclass
class FstArc:
    to_state: int
    in_label: str = EPS
    out_label: str = EPS
    log_prob: Optional[float] = None


@dataclass
class Fst:
    arcs: Dict[int, List[FstArc]] = field(default_factory=lambda: defaultdict(list))
    states: Set[int] = field(default_factory=lambda: {0})
    final_states: Set[int] = field(default_factory=set)
    words: Set[str] = field(default_factory=set)
    output_words: Set[str] = field(default_factory=set)
    start: int = 0
    current_state: int = 0

    def next_state(self) -> int:
        self.states.add(self.current_state)
        self.current_state += 1
        return self.current_state

    def next_edge(
        self,
        from_state: int,
        in_label: Optional[str] = None,
        out_label: Optional[str] = None,
        log_prob: Optional[float] = None,
    ) -> int:
        to_state = self.next_state()
        self.add_edge(from_state, to_state, in_label, out_label, log_prob)
        return to_state

    def add_edge(
        self,
        from_state: int,
        to_state: int,
        in_label: Optional[str] = None,
        out_label: Optional[str] = None,
        log_prob: Optional[float] = None,
    ) -> None:
        if in_label is None:
            in_label = EPS

        if out_label is None:
            out_label = in_label

        if (" " in in_label) or (" " in out_label):
            raise ValueError(
                f"Cannot have white space in labels: from={in_label}, to={out_label}"
            )

        if (not in_label) or (not out_label):
            raise ValueError(f"Labels cannot be empty: from={in_label}, to={out_label}")

        if in_label != EPS:
            self.words.add(in_label)

        if out_label != EPS:
            self.output_words.add(out_label)

        self.states.add(from_state)
        self.states.add(to_state)
        self.arcs[from_state].append(FstArc(to_state, in_label, out_label, log_prob))

    def accept(self, state: int) -> None:
        self.states.add(state)
        self.final_states.add(state)

    def write(self, fst_file: TextIO, symbols_file: Optional[TextIO] = None) -> None:
        symbols = {EPS: 0, UNK: 1}

        for state, arcs in self.arcs.items():
            for arc in arcs:
                if arc.in_label not in symbols:
                    symbols[arc.in_label] = len(symbols)

                if arc.out_label not in symbols:
                    symbols[arc.out_label] = len(symbols)

                if arc.log_prob is None:
                    print(
                        state, arc.to_state, arc.in_label, arc.out_label, file=fst_file
                    )
                else:
                    print(
                        state,
                        arc.to_state,
                        arc.in_label,
                        arc.out_label,
                        arc.log_prob,
                        file=fst_file,
                    )

        for state in self.final_states:
            print(state, file=fst_file)

        if symbols_file is not None:
            for symbol, symbol_id in symbols.items():
                print(symbol, symbol_id, file=symbols_file)

    def remove_spaces(self) -> "Fst":
        """Remove <space> tokens and merge partial word labels."""
        visited: Dict[Tuple[int, int, int], int] = {}

        fst_without_spaces = Fst()
        for arc in self.arcs[self.start]:
            # Copy initial weighted intent arc
            output_state = fst_without_spaces.next_edge(
                fst_without_spaces.start, log_prob=arc.log_prob
            )

            for next_arc_idx, next_arc in enumerate(self.arcs[arc.to_state]):
                self._remove_spaces(
                    arc.to_state,
                    next_arc,
                    next_arc_idx,
                    "",
                    None,
                    visited,
                    fst_without_spaces,
                    output_state,
                )

        return fst_without_spaces

    def _remove_spaces(
        self,
        state: int,
        arc: FstArc,
        arc_idx: int,
        word: str,
        output_word: Optional[str],
        visited: Dict[Tuple[int, int, int], int],
        fst_without_spaces: "Fst",
        output_state: int,
        suppress_output: SuppressOutput = SuppressOutput.DISABLED,
    ) -> None:
        if arc.in_label == SPACE:
            key = (state, arc.to_state, arc_idx)
            cached_state = visited.get(key)
            input_symbol = word or EPS
            output_symbol = input_symbol

            if suppress_output in (
                SuppressOutput.UNTIL_END,
                SuppressOutput.UNTIL_SPACE,
            ):
                # Suppress output
                output_symbol = output_word or EPS
                output_word = None  # consume
            elif output_word is not None:
                # Override output
                output_symbol = output_word
                output_word = None  # consume

            if cached_state is not None:
                fst_without_spaces.add_edge(
                    output_state,
                    cached_state,
                    input_symbol,
                    output_symbol,
                    # log_prob=WORD_PENALTY if input_symbol != EPS else None,
                )
                return

            output_state = fst_without_spaces.next_edge(
                output_state,
                input_symbol,
                output_symbol,
                # log_prob=WORD_PENALTY if input_symbol != EPS else None,
            )
            visited[key] = output_state

            if arc.to_state in self.final_states:
                fst_without_spaces.final_states.add(output_state)

            word = ""

            if suppress_output == SuppressOutput.UNTIL_SPACE:
                suppress_output = SuppressOutput.DISABLED
        elif arc.in_label != EPS:
            word += arc.in_label

            if (
                (suppress_output == SuppressOutput.DISABLED)
                and (arc.out_label != EPS)
                and (arc.out_label != arc.in_label)
            ):
                # Short-term output override
                suppress_output = SuppressOutput.UNTIL_SPACE
                output_word = arc.out_label

        for next_arc_idx, next_arc in enumerate(self.arcs[arc.to_state]):
            self._remove_spaces(
                arc.to_state,
                next_arc,
                next_arc_idx,
                word,
                output_word,
                visited,
                fst_without_spaces,
                output_state,
                suppress_output=suppress_output,
            )

    def prune(self) -> None:
        """Remove paths not connected to a final state."""
        while True:
            states_to_prune: Set[int] = set()

            for state in self.states:
                if (not self.arcs[state]) and (state not in self.final_states):
                    states_to_prune.add(state)

            if not states_to_prune:
                break

            self.states.difference_update(states_to_prune)

            # Prune outgoing arcs
            for state in states_to_prune:
                self.arcs.pop(state, None)

            # Prune incoming arcs
            for state in self.states:
                needs_pruning = any(
                    arc.to_state in states_to_prune for arc in self.arcs[state]
                )
                if needs_pruning:
                    self.arcs[state] = [
                        arc
                        for arc in self.arcs[state]
                        if arc.to_state not in states_to_prune
                    ]

    def to_strings(self, add_spaces: bool) -> List[str]:
        strings: List[str] = []
        self._to_strings("", strings, self.start, add_spaces)

        return strings

    def _to_strings(self, text: str, strings: List[str], state: int, add_spaces: bool):
        if state in self.final_states:
            text_norm = " ".join(text.strip().split())
            if text_norm:
                strings.append(text_norm)

        for arc in self.arcs[state]:
            if arc.in_label == SPACE:
                arc_text = text + " "
            elif arc.in_label != EPS:
                if add_spaces:
                    arc_text = text + " " + arc.in_label
                else:
                    arc_text = text + arc.in_label
            else:
                # Skip <eps>
                arc_text = text

            self._to_strings(arc_text, strings, arc.to_state, add_spaces)

    def to_tokens(self, only_connected: bool = True) -> List[List[str]]:
        tokens: List[List[str]] = []
        self._to_tokens([], tokens, self.start, only_connected)

        # Remove final spaces
        for path in tokens:
            if path and (path[-1] == SPACE):
                path.pop()

        return tokens

    def _to_tokens(
        self,
        path: List[str],
        tokens: List[List[str]],
        state: int,
        only_connected: bool,
    ):
        if (state in self.final_states) and path:
            tokens.append(path)

        has_arcs = False
        for arc in self.arcs[state]:
            has_arcs = True

            # Skip <eps> and initial <space>
            if (arc.in_label == EPS) or (arc.in_label == SPACE and (not path)):
                arc_path = path
            else:
                arc_path = path + [arc.in_label.strip()]

            self._to_tokens(arc_path, tokens, arc.to_state, only_connected)

        if path and (not has_arcs) and (not only_connected):
            # Dead path
            tokens.append(path)


def expression_to_fst(
    expression: Expression,
    state: int,
    fst: Fst,
    intent_data: IntentData,
    intents: Intents,
    list_ref_node: ListReferenceNode,
    slot_lists: Optional[Dict[str, SlotList]] = None,
) -> Optional[int]:
    if isinstance(expression, TextChunk):
        chunk: TextChunk = expression

        space_before = False
        space_after = False

        if chunk.original_text == " ":
            return fst.next_edge(state, SPACE)

        if chunk.original_text.startswith(" "):
            space_before = True

        if chunk.original_text.endswith(" "):
            space_after = True

        word = chunk.original_text.strip()
        if not word:
            return state

        if space_before:
            state = fst.next_edge(state, SPACE)

        sub_words = word.split()

        last_sub_word_idx = len(sub_words) - 1
        for sub_word_idx, sub_word in enumerate(sub_words):
            if isinstance(sub_word, str):
                sub_output_word: Optional[str] = sub_word
            else:
                sub_word, sub_output_word = sub_word
                sub_output_word = sub_output_word or EPS

            state = fst.next_edge(state, sub_word, sub_output_word)
            if sub_word_idx != last_sub_word_idx:
                # Add spaces between words
                state = fst.next_edge(state, SPACE)

        if space_after:
            state = fst.next_edge(state, SPACE)

        return state

    if isinstance(expression, Group):
        grp: Group = expression
        if isinstance(grp, Alternative):
            start = state
            end = fst.next_state()

            for item in grp.items:
                item_list_ref_node = ListReferenceNode()
                list_ref_node.children.append(item_list_ref_node)

                maybe_state = expression_to_fst(
                    item,
                    start,
                    fst,
                    intent_data,
                    intents,
                    item_list_ref_node,
                    slot_lists,
                )
                if maybe_state is None:
                    # Dead branch
                    continue

                state = maybe_state
                if state == start:
                    # Empty item
                    continue

                fst.add_edge(state, end)

            if grp.is_optional:
                fst.add_edge(start, end)

            return end

        if isinstance(grp, Sequence):
            for item in grp.items:
                maybe_state = expression_to_fst(
                    item,
                    state,
                    fst,
                    intent_data,
                    intents,
                    list_ref_node,
                    slot_lists,
                )

                if maybe_state is None:
                    # Dead branch
                    return None

                state = maybe_state

            return state

        if isinstance(grp, Permutation):
            # a;b -> (a b|b a)
            return expression_to_fst(
                Alternative(
                    [
                        Sequence(perm_items)  # type: ignore[arg-type]
                        for perm_items in itertools.permutations(grp.items)
                    ]
                ),
                state,
                fst,
                intent_data,
                intents,
                list_ref_node,
                slot_lists,
            )

        raise ValueError(f"Unexpected group type: {grp}")

    if isinstance(expression, ListReference):
        # {list}
        list_ref: ListReference = expression

        slot_list: Optional[SlotList] = None
        if slot_lists is not None:
            slot_list = slot_lists.get(list_ref.list_name)

        if slot_list is None:
            slot_list = intent_data.slot_lists.get(list_ref.list_name)

        if slot_list is None:
            slot_list = intents.slot_lists.get(list_ref.list_name)

        if isinstance(slot_list, TextSlotList) or (
            (slot_list is None) and (list_ref.slot_name in ("name", "area", "floor"))
        ):
            list_name = list_ref.slot_name
            domains: Optional[Set[str]] = None
            if (list_name == "name") and (intent_data.requires_context is not None):
                requires_domain = intent_data.requires_context.get("domain")
                if requires_domain:
                    if isinstance(requires_domain, str):
                        requires_domain = [requires_domain]

                    domains = set(requires_domain)

            list_ref_node.list_names.append((list_name, domains))

            return expression_to_fst(
                TextChunk(f"{{{list_name}}}"),
                state,
                fst,
                intent_data,
                intents,
                list_ref_node,
                slot_lists,
            )

        if isinstance(slot_list, RangeSlotList):
            list_name = list_ref.slot_name
            list_ref_node.list_names.append((list_name, None))
            return expression_to_fst(
                TextChunk(f"{{{list_name}}}"),
                state,
                fst,
                intent_data,
                intents,
                list_ref_node,
                slot_lists,
            )

        word = f"{{{list_ref.list_name}}}"
        fst.next_edge(state, word, word)
        return None

    if isinstance(expression, RuleReference):
        # <rule>
        rule_ref: RuleReference = expression

        rule_body: Optional[Sentence] = intent_data.expansion_rules.get(
            rule_ref.rule_name
        )
        if rule_body is None:
            rule_body = intents.expansion_rules.get(rule_ref.rule_name)

        if rule_body is None:
            raise ValueError(f"Missing expansion rule <{rule_ref.rule_name}>")

        return expression_to_fst(
            rule_body.expression,
            state,
            fst,
            intent_data,
            intents,
            list_ref_node,
            slot_lists,
        )

    return state


def intents_to_fst(
    intents: Intents,
    slot_lists: Optional[Dict[str, SlotList]] = None,
    intent_names: Optional[Collection[str]] = None,
) -> Fst:
    fst_with_spaces = Fst()
    final = fst_with_spaces.next_state()

    skip_alt_start = Alternative(
        items=[TextChunk("<skip> "), TextChunk("")], is_optional=True
    )
    skip_alt_end = Alternative(
        items=[TextChunk(" <skip>"), TextChunk("")], is_optional=True
    )
    intent_slot_combos = defaultdict(set)
    for intent_name, intent in intents.intents.items():
        if intent_names and (intent_name not in intent_names):
            continue

        for data in intent.data:
            for sentence in data.sentences:
                sentence_state = fst_with_spaces.next_edge(
                    fst_with_spaces.start,
                    SPACE,
                    SPACE,
                )

                list_ref_node = ListReferenceNode()
                state = expression_to_fst(
                    Sequence(items=[skip_alt_start, sentence.expression, skip_alt_end]),
                    # sentence.expression,
                    sentence_state,
                    fst_with_spaces,
                    data,
                    intents,
                    list_ref_node,
                    slot_lists,
                )

                if state is None:
                    # Dead branch
                    continue

                fst_with_spaces.add_edge(state, final, SPACE, SPACE)

                q: Deque[Tuple[ListReferenceNode, List[str]]] = deque(
                    [(list_ref_node, [])]
                )
                while q:
                    node, slots = q.popleft()
                    if node.list_names:
                        slots.extend(n[0] for n in node.list_names)

                    if not node.children:
                        intent_slot_combos[intent_name].add(tuple(sorted(slots)))
                        continue

                    for child_node in node.children:
                        q.append((child_node, list(slots)))

    fst_with_spaces.accept(final)

    return fst_with_spaces
