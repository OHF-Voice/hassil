"""Fuzzy matching using n-grams."""

import itertools
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Collection, Dict, Final, List, Optional, Set, Tuple, Union

from unicode_rbnf import RbnfEngine

from .intents import Intents, RangeSlotList, SlotList, TextSlotList
from .ngram import BOS, EOS, UNK_LOG_PROB, NgramProbCache, Sqlite3NgramModel
from .sample import sample_expression
from .trie import Trie
from .util import normalize_text, remove_punctuation

MIN_SCORE: Final = -13.0
CLOSE_SCORE: Final = 1.0
EQUAL_SCORE: Final = 0.1
MIN_DIFF_SCORE: Final = 0.2
UNK_LOG_PROB_SCALE: Final = -20
STOP_WORD_LOG_PROB: Final = -10
KEYWORD_BOOST_SCORE: Final = 2.0
SKIP: Final = "<skip>"


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
        intent_models: Dict[str, Sqlite3NgramModel],
        intent_slot_list_names: Mapping[str, Collection[str]],
        slot_combinations: Mapping[
            str, Mapping[Tuple[str, ...], Sequence[SlotCombinationInfo]]
        ],
        domain_keywords: Mapping[str, Collection[str]],
        stop_words: Optional[Collection[str]] = None,
        slot_lists: Optional[Dict[str, SlotList]] = None,
    ) -> None:
        self.intents = intents
        self.intent_models = intent_models
        self.intent_slot_list_names = intent_slot_list_names
        self.slot_combinations = slot_combinations
        self.domain_keywords = domain_keywords
        self.stop_words = set(stop_words or [])

        self._slot_combo_intents: Dict[Tuple[str, ...], Set[str]] = defaultdict(set)
        for intent_name, intent_combos in self.slot_combinations.items():
            for slot_combo in intent_combos:
                self._slot_combo_intents[slot_combo].add(intent_name)

        self._trie = self._build_trie(slot_lists)

        # intent -> word -> score
        self._keyword_scores: Optional[Dict[str, Dict[str, float]]] = None

    def match(
        self,
        text: str,
        min_score: Optional[float] = MIN_SCORE,
        close_score: Optional[float] = CLOSE_SCORE,
        equal_score: Optional[float] = EQUAL_SCORE,
    ):
        text_norm = remove_punctuation(normalize_text(text)).lower()

        # (start, end) -> value
        span_map: Dict[Tuple[int, int], SpanValue] = {}
        tokens = text_norm.split()
        keyword_boosts = self._get_keyword_boosts(tokens)
        spans = self._trie.find(text_norm, unique=False, word_boundaries=True)

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

        unk_logprob_cache: Dict[str, float] = {}

        def unk_log_prob(word: str) -> float:
            word_prob = unk_logprob_cache.get(word)
            if word_prob is None:
                best_prob = None
                for intent_ngram_model in self.intent_models.values():
                    if word not in intent_ngram_model.words:
                        continue

                    model_word_prob = intent_ngram_model.get_log_prob([word])
                    if best_prob is None:
                        best_prob = model_word_prob
                    else:
                        best_prob = max(best_prob, model_word_prob)

                if best_prob is None:
                    word_prob = UNK_LOG_PROB
                else:
                    scale = 1 - (math.exp(best_prob))
                    word_prob = UNK_LOG_PROB + (scale * UNK_LOG_PROB_SCALE)

                if word in self.stop_words:
                    word_prob = max(word_prob, STOP_WORD_LOG_PROB)

                unk_logprob_cache[word] = word_prob

            return word_prob

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
                        if interp_tokens and (
                            (token != SKIP) or (interp_tokens[-1] != SKIP)
                        ):
                            # Combine multiple "<skip>"
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

                if (len(interp_tokens) == 2) and (interp_tokens[1] == "{name}"):
                    # Don't try to interpret entity names only
                    continue

                interp_tokens.append(EOS)

                model_domain = name_domain
                if (not model_domain) and (
                    domain_slot_value := slot_values.get("domain")
                ):
                    model_domain = domain_slot_value[0]

                # Score token string for each intent
                intent_models_to_check = [
                    (model_name, model)
                    for intent_name in intents_to_check
                    for model_name, model in self.intent_models.items()
                    if model_name.endswith(f"_{intent_name}")
                ]

                for model_name, intent_ngram_model in intent_models_to_check:
                    intent_domain, intent_name = model_name.rsplit("_", maxsplit=1)
                    if (
                        model_domain
                        and (model_domain != intent_domain)
                        and (f"{model_domain}_{intent_name}" in self.intent_models)
                    ):
                        # Skip if a domain-specific model exists
                        continue

                    intent_score = intent_ngram_model.get_log_prob(
                        interp_tokens,
                        unk_log_prob=unk_log_prob,
                        cache=logprob_cache[model_name],
                    ) / len(tokens)

                    intent_score += (
                        keyword_boosts.get(model_name, 0.0) * KEYWORD_BOOST_SCORE
                    )

                    if (min_score is not None) and (intent_score < min_score):
                        # Below minimum score
                        continue

                    if (
                        (best_score is not None)
                        and (intent_score <= best_score)
                        and (equal_score is not None)
                        and (abs(best_score - intent_score) <= EQUAL_SCORE)
                    ):
                        # Keep for uncertainty check below
                        best_scores.append((intent_name, intent_score))
                        continue

                    if (
                        (best_score is None)
                        or (intent_score > best_score)
                        or (
                            (close_score is not None)
                            and (
                                # prefer more slots matched and "name" slots
                                (abs(intent_score - best_score) < close_score)
                                and (
                                    ((not best_slots) and slot_values)
                                    or (
                                        best_slots
                                        and (
                                            (len(slot_values) > len(best_slots))
                                            or (
                                                ("name" in slot_values)
                                                and (
                                                    "name"
                                                    not in best_slots  # pylint: disable=unsupported-membership-test
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    ):
                        best_intent_name = intent_name
                        best_score = intent_score
                        best_slots = slot_values
                        best_name_domain = name_domain
                        best_scores.append((best_intent_name, best_score))

        if not best_intent_name:
            return None

        if (best_score is not None) and (len(best_scores) > 1):
            for other_intent_name, other_score in best_scores:
                if best_intent_name == other_intent_name:
                    continue

                if abs(other_score - best_score) < MIN_DIFF_SCORE:
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

    def _get_keyword_boosts(self, words: Collection[str]) -> Dict[str, float]:
        word_boosts: Dict[str, float] = defaultdict(lambda: 0.0)

        if not self._keyword_scores:
            self._build_keyword_scores()

        assert self._keyword_scores is not None

        for intent_name, word_scores in self._keyword_scores.items():
            for word in words:
                word_boosts[intent_name] += word_scores.get(word, 0.0)

        if not word_boosts:
            return word_boosts

        max_boost = max(word_boosts.items(), key=lambda kv: kv[1])[1]
        if max_boost > 0:
            for intent_name in word_boosts:
                word_boosts[intent_name] /= max_boost

        return word_boosts

    def _build_keyword_scores(self) -> None:
        self._keyword_scores = defaultdict(dict)

        num_intents = len(self.intent_models)
        word_logprobs: Dict[str, Dict[str, float]] = defaultdict(dict)
        for intent_name, model in self.intent_models.items():
            for word in model.words:
                if (
                    word.startswith("<")
                    or word.startswith("{")
                    or (word in self.stop_words)
                ):
                    continue

                word_logprobs[word][intent_name] = model.get_log_prob([word])

        for word, intent_info in word_logprobs.items():
            max_prob = None
            for intent_name, log_prob in intent_info.items():
                prob = 10**log_prob
                self._keyword_scores[intent_name][word] = prob
                if (max_prob is None) or (prob > max_prob):
                    max_prob = prob

            df = len(intent_info)
            idf = math.log(num_intents / (1 + df))
            if max_prob is None:
                max_prob = 1.0

            for intent_name in intent_info:
                score = (self._keyword_scores[intent_name][word] / max_prob) * idf
                self._keyword_scores[intent_name][word] = score

        for intent_name, word_scores in self._keyword_scores.items():
            max_score = max(word_scores.values())
            for word, score in word_scores.items():
                word_scores[word] = score / max_score
