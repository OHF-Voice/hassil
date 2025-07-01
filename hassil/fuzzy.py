import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Collection, TextIO
from pathlib import Path
from collections.abc import Iterable
from typing import Tuple, Dict, Any, TypedDict, Optional, Union, List, Set

from unicode_rbnf import RbnfEngine

from .intents import Intents, TextSlotList, RangeSlotList
from .trie import Trie
from .sample import sample_expression
from .util import normalize_text, remove_punctuation


class NgramProb(TypedDict):
    log_prob: float
    backoff: Optional[float]


@dataclass
class SpanValue:
    text: str
    value: Any


@dataclass
class ListSpanValue(SpanValue):
    list_name: str
    slot_name: str
    domain: Optional[str]


@dataclass
class RangeSpanValue(SpanValue):
    list_name: str
    slot_name: str


@dataclass
class SlotCombinationInfo:
    name_domains: Optional[Set[str]] = None


NgramModelType = Dict[Tuple[str, ...], NgramProb]


class FuzzyNgramMatcher:

    def __init__(
        self,
        intents: Intents,
        arpa_dir: Union[str, Path],
        intent_slot_list_names: Dict[str, str],
        slot_combinations: Dict[str, Dict[Tuple[str, ...], List[SlotCombinationInfo]]],
        domain_keywords: Dict[str, Collection[str]],
    ) -> None:
        self.intents = intents
        self.arpa_dir = Path(arpa_dir)
        self.intent_slot_list_names = intent_slot_list_names
        self.slot_combinations = slot_combinations
        self.domain_keywords = domain_keywords

        self._trie = self._build_trie()
        self._intent_models: Dict[str, NgramModelType] = self._load_models()

    def match(
        self,
        text: str,
        min_score: Optional[float] = -20,
        close_score: Optional[float] = 1,
    ):
        text_norm = remove_punctuation(normalize_text(text).casefold())

        span_map = defaultdict(list)
        tokens = text_norm.split()
        spans = self._trie.find(text_norm, unique=False)
        for end_idx, span_text, span_value in spans:
            start_idx = end_idx - len(span_text)
            token_start_idx = len(text_norm[:start_idx].split())
            token_end_idx = token_start_idx + len(text_norm[start_idx:end_idx].split())

            span_map[(token_start_idx, token_end_idx)].append(span_value)

        best_intent = None
        best_score = None
        best_interp = None
        best_slot_names = None

        # TODO: cache probabilities of prefixes
        logprob_cache = defaultdict(dict)
        for pos_and_values in self._find_interpretations(tokens, span_map):
            scores = {}
            # tokens = []
            tokens = ["<s>"]
            for _start_idx, _end_idx, value in pos_and_values:
                if isinstance(value, list):
                    value = value[0]

                if isinstance(value, SpanValue):
                    tokens.append(value.text)
                else:
                    tokens.append(value)

            tokens.append("</s>")
            # print(pos_and_values)

            if not tokens:
                continue

            for intent_name, model in self._intent_models.items():
                scores[intent_name] = get_log_prob_cached(
                    tokens,
                    model,
                    cache=logprob_cache[intent_name],
                ) / len(tokens)

            # print(scores)
            # print("")

            for intent_name, intent_score in sorted(
                scores.items(), key=lambda kv: kv[1], reverse=True
            ):
                if (min_score is not None) and (intent_score < min_score):
                    continue

                if (best_score is not None) and (intent_score < best_score):
                    continue

                slot_names = []
                name_domain: Optional[str] = None
                for _start_idx, _end_idx, values in pos_and_values:
                    if not isinstance(values, list):
                        values = [values]

                    for value in values:
                        if isinstance(value, ListSpanValue):
                            slot_names.append(value.slot_name)
                            if value.slot_name == "name":
                                name_domain = value.domain
                        elif isinstance(value, RangeSpanValue):
                            slot_names.append(value.slot_name)

                combo_key = tuple(sorted(slot_names))
                slot_combos = self.slot_combinations[intent_name].get(combo_key)
                # print(intent_name, combo_key)
                if not slot_combos:
                    # Invalid slot combination
                    # print(
                    #     "Skipping invalid slot combination:",
                    #     intent_name,
                    #     slot_names,
                    #     pos_and_values,
                    # )
                    continue

                name_domain_match = True
                if name_domain:
                    name_domain_match = False
                    for slot_combo in slot_combos:
                        if (not slot_combo.name_domains) or (
                            name_domain in slot_combo.name_domains
                        ):
                            name_domain_match = True
                            break

                if not name_domain_match:
                    # Invalid name domain
                    continue

                if (
                    (best_score is None)
                    or (intent_score > best_score)
                    or (
                        (close_score is not None)
                        and (
                            (abs(intent_score - best_score) < close_score)
                            and slot_names
                            and (
                                (not best_slot_names)
                                or (len(slot_names) > len(best_slot_names))
                                or (
                                    (len(slot_names) == len(best_slot_names))
                                    and ("name" in slot_names)
                                )
                            )
                        )
                    )
                ):
                    best_intent = intent_name
                    best_score = intent_score
                    best_interp = pos_and_values
                    best_slot_names = slot_names
                    # print(best_score, best_intent, best_slot_names)
                    break

        best_slots = {}
        if best_interp:
            for _start_idx, _end_idx, values in best_interp:
                if not isinstance(values, list):
                    values = [values]
                for value in values:
                    if isinstance(value, (ListSpanValue, RangeSpanValue)):
                        best_slots[value.slot_name] = value.value

        print(best_score, best_intent, best_slots)

    def _find_interpretations(self, tokens, span_map, pos=0, cache=None):
        if cache is None:
            cache = {}

        if pos == len(tokens):
            return [[]]

        if pos in cache:
            return cache[pos]

        # TODO: handle {range}%
        interpretations = []

        # Option 1: Keep original token
        for rest in self._find_interpretations(tokens, span_map, pos + 1, cache):
            interpretations.append([(pos, pos + 1, tokens[pos])] + rest)

        # Option 2: Replace with a slot or skip if span exists
        for end in range(pos + 1, len(tokens) + 1):
            if (pos, end) not in span_map:
                continue
            replacements = span_map[(pos, end)]
            domain_replacement = next(
                (
                    r
                    for r in replacements
                    if isinstance(r, ListSpanValue) and r.slot_name == "domain"
                ),
                None,
            )
            for repl in replacements:
                if repl == domain_replacement:
                    continue

                if domain_replacement:
                    repl = [repl, domain_replacement]

                for rest in self._find_interpretations(tokens, span_map, end, cache):
                    interpretations.append([(pos, end, repl)] + rest)

        cache[pos] = interpretations
        return interpretations

    def _build_trie(self) -> Trie:
        trie = Trie()

        number_engine: Optional[RbnfEngine] = None
        try:
            number_engine = RbnfEngine.for_language(self.intents.language)
        except ValueError:
            # Number words will not be available
            pass

        number_cache: Dict[Union[int, float], str] = {}

        # used_range_list_names = set()
        for list_name, slot_list in self.intents.slot_lists.items():
            slot_name = self.intent_slot_list_names.get(list_name)
            if not slot_name:
                continue

            if isinstance(slot_list, TextSlotList):
                text_list: TextSlotList = slot_list
                trie_list_name = f"{{{list_name}}}"
                for value in text_list.values:
                    for value_text in sample_expression(value.text_in):
                        trie.insert(
                            value_text.lower(),
                            ListSpanValue(
                                text=value_text,
                                value=value.value_out,
                                list_name=trie_list_name,
                                slot_name=slot_name,
                                domain=(
                                    value.context.get("domain")
                                    if value.context
                                    else None
                                ),
                            ),
                        )
            elif isinstance(slot_list, RangeSlotList):
                range_list: RangeSlotList = slot_list
                trie_list_name = (
                    f"{{range_{range_list.start},{range_list.stop},{range_list.step}}}"
                )
                # if trie_list_name in used_range_list_names:
                #     continue

                for num in range_list.get_numbers():
                    num_str = str(num)
                    trie.insert(
                        num_str,
                        RangeSpanValue(
                            text=num_str,
                            value=num,
                            list_name=trie_list_name,
                            slot_name=slot_name,
                        ),
                    )

                    if number_engine is not None:
                        num_words = number_cache.get(num)
                        if num_words is None:
                            num_words = number_engine.format_number(num).text
                            number_cache[num] = num_words

                        trie.insert(
                            num_words,
                            RangeSpanValue(
                                text=num_words,
                                value=num,
                                list_name=trie_list_name,
                                slot_name=slot_name,
                            ),
                        )

                # used_range_list_names.add(trie_list_name)

        for skip_word in self.intents.skip_words:
            trie.insert(skip_word, SpanValue(text="<skip>", value=None))

        # TODO
        for domain, keywords in self.domain_keywords.items():
            for keyword in keywords:
                trie.insert(
                    keyword,
                    ListSpanValue(
                        text=keyword,
                        value=domain,
                        list_name="domain",
                        slot_name="domain",
                        domain=None,
                    ),
                )

        return trie

    def _load_models(self) -> Dict[str, NgramModelType]:
        models: Dict[str, NgramModelType] = {}

        for intent_name in self.intents.intents:
            intent_arpa_path = self.arpa_dir / f"{intent_name}.arpa"
            if not intent_arpa_path.exists():
                continue

            with open(intent_arpa_path, "r", encoding="utf-8") as intent_arpa_file:
                models[intent_name] = load_arpa(intent_arpa_file)

        return models


def load_arpa(arpa_file: TextIO) -> NgramModelType:
    model: NgramModelType = {}
    order = 0
    reading_ngrams = False

    for line in arpa_file:
        line = line.strip()

        # Start of new section?
        if line.startswith("\\") and "-grams:" in line:
            order = int(line.strip("\\-grams:"))
            reading_ngrams = True
            continue
        elif line.startswith("\\end\\"):
            break
        elif not line or line.startswith("ngram") or not reading_ngrams:
            continue

        parts = line.split()
        if len(parts) < order + 1:
            continue  # malformed line

        log_prob = float(parts[0])
        ngram = tuple(parts[1 : 1 + order])
        backoff = float(parts[1 + order]) if len(parts) > 1 + order else None

        model[ngram] = {"log_prob": log_prob}
        if backoff is not None:
            model[ngram]["backoff"] = backoff

    return model


# def get_log_prob(
#     tokens: Iterable[str],
#     model: NgramModelType,
#     order: int = 5,
#     unk_log_prob: float = -10.0,
# ) -> float:
#     total_log_prob = 0.0
#     context: List[str] = []

#     for word in tokens:
#         found = False

#         # Try highest to lowest n-gram order
#         for n in reversed(range(1, order + 1)):
#             prefix = tuple(context[-(n - 1) :]) if n > 1 else ()
#             ngram = prefix + (word,)

#             if ngram in model:
#                 total_log_prob += model[ngram]["log_prob"]
#                 found = True
#                 break
#             elif (prefix in model) and ("backoff" in model[prefix]):
#                 total_log_prob += model[prefix]["backoff"]

#         if not found:
#             total_log_prob += unk_log_prob

#         context.append(word)

#     return total_log_prob


def get_log_prob_cached(
    tokens: Iterable[str],
    model: NgramModelType,
    order: int = 5,
    unk_log_prob: float = -15.0,
    min_log_prob: Optional[float] = None,
    cache: Optional[Dict[Tuple[str, ...], float]] = None,
) -> float:
    if cache is None:
        cache = {}

    total_log_prob = 0.0
    context: List[str] = []

    for word in tokens:
        if word == "<s>":
            # Skip BOS since its not a normal token
            context.append(word)
            continue

        context_key = tuple(context + [word])

        # Check external prefix cache
        if context_key in cache:
            total_log_prob = cache[context_key]
            if (min_log_prob is not None) and (total_log_prob < min_log_prob):
                # Stop early
                return -math.inf

            context.append(word)
            continue

        found = False

        # Try highest to lowest n-gram order
        for n in reversed(range(1, order + 1)):
            prefix = tuple(context[-(n - 1) :]) if n > 1 else ()
            ngram = prefix + (word,)

            if ngram in model:
                total_log_prob += model[ngram]["log_prob"]
                found = True
                break
            elif (prefix in model) and ("backoff" in model[prefix]):
                total_log_prob += model[prefix]["backoff"]

        if not found:
            total_log_prob += unk_log_prob

        if (min_log_prob is not None) and (total_log_prob < min_log_prob):
            # Stop early
            return -math.inf

        context.append(word)

        # Store in external prefix cache
        cache[context_key] = total_log_prob

    return total_log_prob
