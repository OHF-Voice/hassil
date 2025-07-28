"""Fuzzy matching using n-grams."""

import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Collection, Dict, Final, List, Optional, Set, Tuple, Union

from unicode_rbnf import RbnfEngine

from .intents import Intents, RangeSlotList, SlotList, TextSlotList
from .ngram import BOS, EOS, NgramModel, NgramProbCache
from .sample import sample_expression
from .trie import Trie
from .util import normalize_text, remove_punctuation

MIN_SCORE: Final = -20.0
CLOSE_SCORE: Final = 1.0
MIN_DIFF_SCORE: Final = 0.2


@dataclass
class SpanSlotValue:
    value: Any
    name_domain: Optional[str] = None
    suffix: Optional[str] = None


@dataclass
class SpanValue:
    text: str
    slots: Dict[str, SpanSlotValue] = field(default_factory=dict)
    inferred_domain: Optional[str] = None


@dataclass
class SlotCombinationInfo:
    name_domains: Optional[Set[str]] = None


@dataclass
class FuzzySlotValue:
    value: Any
    text: str


@dataclass
class FuzzyResult:
    intent_name: str
    slots: Dict[str, FuzzySlotValue]
    score: Optional[float] = None
    name_domain: Optional[str] = None


# -----------------------------------------------------------------------------


class FuzzyNgramMatcher:

    def __init__(
        self,
        intents: Intents,
        intent_models: Dict[str, NgramModel],
        intent_slot_list_names: Dict[str, Collection[str]],
        slot_combinations: Dict[str, Dict[Tuple[str, ...], List[SlotCombinationInfo]]],
        domain_keywords: Dict[str, Collection[str]],
        slot_lists: Optional[Dict[str, SlotList]] = None,
    ) -> None:
        self.intents = intents
        self.intent_models = intent_models
        self.intent_slot_list_names = intent_slot_list_names
        self.slot_combinations = slot_combinations
        self.domain_keywords = domain_keywords

        self._slot_combo_intents: Dict[Tuple[str, ...], Set[str]] = defaultdict(set)
        for intent_name, intent_combos in self.slot_combinations.items():
            for slot_combo in intent_combos:
                self._slot_combo_intents[slot_combo].add(intent_name)

        self._trie = self._build_trie(slot_lists)

    def match(
        self,
        text: str,
        min_score: Optional[float] = MIN_SCORE,
        close_score: Optional[float] = CLOSE_SCORE,
    ):
        text_norm = remove_punctuation(normalize_text(text)).lower()

        # (start, end) -> value
        span_map: Dict[Tuple[int, int], SpanValue] = {}
        tokens = text_norm.split()
        spans = self._trie.find(text_norm, unique=False)

        # Get values for spans in text
        for end_idx, span_text, span_value in spans:
            start_idx = end_idx - len(span_text)
            token_start_idx = len(text_norm[:start_idx].split())
            token_end_idx = token_start_idx + len(text_norm[start_idx:end_idx].split())
            span_map[(token_start_idx, token_end_idx)] = span_value

        # Determine best intent match
        best_intent_name: Optional[str] = None
        best_score: Optional[float] = None
        best_slots: Optional[Dict[str, Any]] = None
        best_name_domain: Optional[str] = None

        # (intent name, score)
        best_scores: List[Tuple[str, float]] = []

        # intent -> prob cache
        logprob_cache: Dict[str, NgramProbCache] = defaultdict(dict)

        for pos_and_values in self._find_interpretations(tokens, span_map):
            # Multiples possible values may exist for each token, each one
            # representing a different interpretation.
            #
            # For example, "garage door" may be an entity {name} or a cover {device_class}.
            values: List[List[Tuple[Optional[str], Optional[str], Any, str]]] = []

            for _start_idx, _end_idx, value in pos_and_values:
                if isinstance(value, str):
                    values.append([(value, None, None, value)])
                elif isinstance(value, SpanValue):
                    span_value = value

                    # (token, slot name, slot value, text)
                    sub_values: List[Tuple[Optional[str], Optional[str], Any, str]] = []

                    if span_value.inferred_domain:
                        # Inferred domain is separate
                        values.append(
                            [
                                (
                                    None,
                                    "domain",
                                    span_value.inferred_domain,
                                    span_value.text,
                                )
                            ]
                        )

                    if span_value.slots:
                        # Possible slot interpretations
                        sub_values.extend(
                            (f"{{{slot_name}}}", slot_name, slot_value, span_value.text)
                            for slot_name, slot_value in span_value.slots.items()
                            if slot_name != "domain"
                        )
                    else:
                        sub_values.append(
                            (span_value.text, None, None, span_value.text)
                        )

                    if sub_values:
                        values.append(sub_values)

            # Iterate over possible interpretations, each one made up of
            # (token, slot name, slot value) tuples.
            for tokens_and_values in itertools.product(*values):
                interp_tokens = [BOS]
                slot_names: List[str] = []
                slot_values: Dict[str, Tuple[Any, str]] = {}
                name_domain: Optional[str] = None
                for token, slot_name, slot_value, slot_text in tokens_and_values:
                    if token:
                        interp_tokens.append(token)

                    if slot_name:
                        slot_names.append(slot_name)
                        if isinstance(slot_value, SpanSlotValue):
                            slot_values[slot_name] = (slot_value.value, slot_text)
                            if slot_value.name_domain:
                                # Interpretation is restricted by domain of {name}
                                name_domain = slot_value.name_domain
                        else:
                            slot_values[slot_name] = (slot_value, slot_text)

                combo_key = tuple(sorted(slot_names))
                intents_to_check: Optional[Collection[str]] = (
                    self._slot_combo_intents.get(combo_key)
                )
                if not intents_to_check:
                    # Slot combination is not valid for any intent
                    continue

                if name_domain:
                    # Filter intents by slot combination and name domain
                    intents_to_check = [
                        intent_name
                        for intent_name in intents_to_check
                        if any(
                            combo_info.name_domains
                            and (name_domain in combo_info.name_domains)
                            for combo_info in self.slot_combinations[intent_name][
                                combo_key
                            ]
                        )
                    ]

                if not intents_to_check:
                    # Not a valid slot combination
                    continue

                interp_tokens.append(EOS)

                # Score token string for each intent
                for intent_name in intents_to_check:
                    intent_ngram_model = self.intent_models.get(intent_name)
                    if intent_ngram_model is None:
                        continue

                    intent_score = intent_ngram_model.get_log_prob(
                        interp_tokens, cache=logprob_cache[intent_name]
                    ) / len(tokens)

                    # print(intent_score, intent_name, interp_tokens)

                    if (min_score is not None) and (intent_score < min_score):
                        # Below minimum score
                        continue

                    if (
                        (best_score is None)
                        or (intent_score > best_score)
                        or (
                            (close_score is not None)
                            # prefer more slots matched and "name" slots
                            and (
                                (abs(intent_score - best_score) < close_score)
                                and slot_names
                                and (
                                    (not best_slots)
                                    or (len(slot_values) > len(best_slots))
                                    or ("name" in slot_values)
                                )
                            )
                        )
                    ):
                        best_intent_name = intent_name
                        best_score = intent_score
                        best_slots = slot_values
                        best_name_domain = name_domain
                        best_scores.append((best_intent_name, best_score))
                        # print("Best:", best_score, best_intent_name, best_slots)

        if not best_intent_name:
            return None

        if len(best_scores) > 1:
            best_scores = sorted(
                best_scores, reverse=True, key=lambda intent_score: intent_score[1]
            )
            # print(best_scores)

            # Different intents but close scores
            if (best_scores[0][0] != best_scores[1][0]) and (
                (best_scores[0][1] - best_scores[1][1]) < MIN_DIFF_SCORE
            ):
                # Not enough difference between top 2 scores indicates uncertainty
                return None

        return FuzzyResult(
            intent_name=best_intent_name,
            slots=(
                {
                    slot_name: FuzzySlotValue(value=slot_value, text=slot_text)
                    for slot_name, (slot_value, slot_text) in best_slots.items()
                }
                if best_slots is not None
                else {}
            ),
            score=best_score,
            name_domain=best_name_domain,
        )

    # -------------------------------------------------------------------------

    def _find_interpretations(self, tokens, span_map, pos: int = 0, cache=None):
        if cache is None:
            cache = {}

        if pos == len(tokens):
            return [[]]

        if pos in cache:
            return cache[pos]

        interpretations = []

        # Option 1: Keep original token
        for rest in self._find_interpretations(tokens, span_map, pos + 1, cache):
            interpretations.append([(pos, pos + 1, tokens[pos])] + rest)

        # Option 2: Replace with a slot or skip if span exists
        for end in range(pos + 1, len(tokens) + 1):
            replacement = span_map.get((pos, end))
            if replacement is None:
                continue

            for rest in self._find_interpretations(tokens, span_map, end, cache):
                interpretations.append([(pos, end, replacement)] + rest)

        cache[pos] = interpretations
        return interpretations

    def _build_trie(self, slot_lists: Optional[Dict[str, SlotList]] = None) -> Trie:
        if slot_lists is None:
            slot_lists = {}

        trie = Trie()

        number_engine: Optional[RbnfEngine] = None
        try:
            number_engine = RbnfEngine.for_language(self.intents.language)
        except ValueError:
            # Number words will not be available
            pass

        number_cache: Dict[Union[int, float], str] = {}
        span_values: Dict[str, SpanValue] = {}

        for list_name, slot_list in itertools.chain(
            self.intents.slot_lists.items(), slot_lists.items()
        ):
            slot_names = self.intent_slot_list_names.get(list_name)
            if not slot_names:
                continue

            for slot_name in slot_names:
                if isinstance(slot_list, TextSlotList):
                    text_list: TextSlotList = slot_list
                    for value in text_list.values:
                        for value_text in sample_expression(value.text_in):
                            span_value = span_values.get(value_text)
                            if span_value is None:
                                span_value = SpanValue(text=value_text)
                                span_values[value_text] = span_value
                                trie.insert(value_text.lower(), span_value)

                            span_value.slots[slot_name] = SpanSlotValue(
                                value=value.value_out,
                                name_domain=(
                                    value.context.get("domain")
                                    if value.context
                                    else None
                                ),
                            )
                elif isinstance(slot_list, RangeSlotList):
                    range_list: RangeSlotList = slot_list
                    suffix: Optional[str] = None
                    if range_list.type == "percentage":
                        suffix = "%"
                    elif range_list.type == "temperature":
                        suffix = "°"

                    for num in range_list.get_numbers():
                        num_strs = [str(num)]
                        if suffix:
                            # Add % or °
                            num_strs.append(f"{num_strs[0]}{suffix}")

                        if number_engine is not None:
                            # 1 -> one
                            num_words = number_cache.get(num)
                            if num_words is None:
                                num_words = number_engine.format_number(num).text
                                number_cache[num] = num_words

                            num_strs.append(num_words)

                        for num_str in num_strs:
                            span_value = span_values.get(num_str)
                            if span_value is None:
                                span_value = SpanValue(text=num_str)
                                span_values[num_str] = span_value
                                trie.insert(num_str.lower(), span_value)

                            span_value.slots[slot_name] = SpanSlotValue(
                                value=num, suffix=suffix
                            )

        # "Skip" words/phrases like "please" and "could you"
        for skip_word in self.intents.skip_words:
            trie.insert(skip_word, SpanValue(text="<skip>"))

        # Map keywords to inferred domain, e.g. "lights" in "turn on the lights"
        for domain, keywords in self.domain_keywords.items():
            for keyword in keywords:
                span_value = span_values.get(keyword)
                if span_value is None:
                    span_value = SpanValue(text=keyword)
                    span_values[keyword] = span_value
                    trie.insert(keyword, span_value)

                span_value.inferred_domain = domain

        return trie
