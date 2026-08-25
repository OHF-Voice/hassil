"""Methods for recognizing intents from text."""

import collections.abc
import itertools
import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    List,
    MutableSequence,
    Optional,
    Tuple,
)

from .expression import Expression, ListReference, Sentence, Sequence
from .intents import Intent, IntentData, Intents, SlotList, WildcardSlotList
from .models import MatchCapture, MatchEntity, UnmatchedEntity, UnmatchedTextEntity
from .string_matcher import MatchContext, MatchSettings, match_expression
from .util import (
    CJK_WHITESPACE,
    TrackedText,
    check_excluded_context,
    check_required_context,
    normalize_for_matching,
    remove_skip_words_tracked,
)

MISSING_ENTITY = "<missing>"

_LOGGER = logging.getLogger()


@dataclass
class RecognizeResult:
    """Result of recognition."""

    intent: Intent
    """Matched intent"""

    intent_data: IntentData
    """Matched intent data"""

    entities: Dict[str, MatchEntity] = field(default_factory=dict)
    """Matched entities mapped by name."""

    entities_list: List[MatchEntity] = field(default_factory=list)
    """Matched entities as a list (duplicates allowed)."""

    response: Optional[str] = None
    """Key for intent response."""

    context: Dict[str, Any] = field(default_factory=dict)
    """Context values acquired during matching."""

    unmatched_entities: Dict[str, UnmatchedEntity] = field(default_factory=dict)
    """Unmatched entities mapped by name."""

    unmatched_entities_list: List[UnmatchedEntity] = field(default_factory=list)
    """Unmatched entities as a list (duplicates allowed)."""

    text_chunks_matched: int = 0
    """Number of literal text chunks that were successfully matched."""

    intent_sentence: Optional[Sentence] = None
    """Sentence template that was matched."""

    intent_metadata: Optional[Dict[str, Any]] = None
    """Metadata from the intent sentence that was matched."""

    captures: Dict[str, MatchCapture] = field(default_factory=dict)
    """Captures for response mapped by name."""

    captures_list: List[MatchCapture] = field(default_factory=list)
    """Captures for response as a list (duplicates allowed)."""

    original_text: str = ""
    """Original text from match."""


def recognize(
    text: str,
    intents: Intents,
    slot_lists: Optional[Dict[str, SlotList]] = None,
    expansion_rules: Optional[Dict[str, Sentence]] = None,
    skip_words: Optional[List[str]] = None,
    intent_context: Optional[Dict[str, Any]] = None,
    default_response: Optional[str] = "default",
    allow_unmatched_entities: bool = False,
    language: Optional[str] = None,
) -> Optional[RecognizeResult]:
    """Return the first match of input text/words against a collection of intents.

    text: Text to recognize
    intents: Compiled intents
    slot_lists: Pre-defined text lists, ranges, or wildcards
    expansion_rules: Named template snippets
    skip_words: Strings to ignore in text
    intent_context: Slot values to use when not found in text
    default_response: Response key to use if not set in intent
    allow_unmatched_entities: True if entity values outside slot lists are allowed (slower)
    language: Optional language to use when converting digits to words

    Returns the first result.
    If allow_unmatched_entities is True, you should check for unmatched entities.
    """
    for result in recognize_all(
        text,
        intents,
        slot_lists=slot_lists,
        expansion_rules=expansion_rules,
        skip_words=skip_words,
        intent_context=intent_context,
        default_response=default_response,
        allow_unmatched_entities=allow_unmatched_entities,
        language=language,
    ):
        return result

    return None


def recognize_all(
    text: str,
    intents: Intents,
    slot_lists: Optional[Dict[str, SlotList]] = None,
    expansion_rules: Optional[Dict[str, Sentence]] = None,
    skip_words: Optional[Iterable[str]] = None,
    intent_context: Optional[Dict[str, Any]] = None,
    default_response: Optional[str] = "default",
    allow_unmatched_entities: bool = False,
    language: Optional[str] = None,
) -> Iterable[RecognizeResult]:
    """Return all matches for input text/words against a collection of intents.

    text: Text to recognize
    intents: Compiled intents
    slot_lists: Pre-defined text lists, ranges, or wildcards
    expansion_rules: Named template snippets
    skip_words: Strings to ignore in text
    intent_context: Slot values to use when not found in text
    default_response: Response key to use if not set in intent
    allow_unmatched_entities: True if entity values outside slot lists are allowed (slower)
    language: Optional language to use when converting digits to words

    Yields results as they're matched.
    If allow_unmatched_entities is True, you should check for unmatched entities.
    """
    # Offsets are tracked alongside every transformation so entity spans can be
    # reported against the text the caller passed in.
    tracked_with_skip_words = normalize_for_matching(text)
    text_with_skip_words = tracked_with_skip_words.text

    if skip_words is None:
        skip_words = intents.skip_words
    else:
        # Combine skip words
        skip_words = list(itertools.chain(skip_words, intents.skip_words))

    if skip_words:
        tracked_no_skip_words = tracked_with_skip_words.copy()
        remove_skip_words_tracked(
            tracked_no_skip_words, skip_words, intents.settings.ignore_whitespace
        )
    else:
        tracked_no_skip_words = tracked_with_skip_words

    text_no_skip_words = tracked_no_skip_words.text

    # True if removing skip words actually changed the text. When it did, a skip
    # word might really be part of a sentence template (e.g. "I want" in
    # "I want to watch TV"), so we also try matching with skip words left in.
    had_skip_words_removed = text_no_skip_words != text_with_skip_words

    text_keywords = text_no_skip_words.split()

    # Haystack for the required-fragment prefilter (see
    # Sentence.get_required_clauses). Both candidate texts are included because
    # removing skip words can join characters that were not adjacent before
    # (in ignore_whitespace mode), so neither text's fragments are a subset of
    # the other's.
    #
    # The matcher's break-words fallback needs no variant here: it only rewrites
    # "-"/"_" in the input to spaces, and a whitespace-free fragment inside the
    # rewritten text always lies within a "-"/"_"-free run of the original.
    prefilter_haystack = (
        f"{text_with_skip_words.casefold()}\n{text_no_skip_words.casefold()}"
    )

    if intents.settings.ignore_whitespace:
        # The matcher drops whitespace between CJK characters on both sides, so a
        # template fragment may only appear in the stripped form of the input.
        prefilter_haystack += "\n" + CJK_WHITESPACE.sub("", prefilter_haystack)

    if slot_lists is None:
        slot_lists = intents.slot_lists
    else:
        # Combine with intents
        slot_lists = {**intents.slot_lists, **slot_lists}

    if expansion_rules is None:
        expansion_rules = intents.expansion_rules
    else:
        # Combine rules
        expansion_rules = {**intents.expansion_rules, **expansion_rules}

    if intent_context is None:
        intent_context = {}

    # Filter intents based on context and keywords
    available_intents: MutableSequence[
        Tuple[Intent, IntentData, MatchSettings, Optional[List[Sentence]]]
    ] = []

    for intent in intents.intents.values():
        for intent_data in intent.data:
            if (
                intent_data.required_keywords
                and intent_data.required_keywords.isdisjoint(text_keywords)
            ):
                # No keyword overlap
                continue

            if intent_context:
                # Skip sentence templates that can't possibly be matched due to
                # requires/excludes context.
                #
                # Additional context can be added during matching, so we can
                # only be sure about keys that exist right now.
                if intent_data.requires_context and (
                    not check_required_context(
                        intent_data.requires_context,
                        intent_context,
                        allow_missing_keys=True,
                    )
                ):
                    continue

                if intent_data.excludes_context and (
                    not check_excluded_context(
                        intent_data.excludes_context, intent_context
                    )
                ):
                    continue

            match_settings = MatchSettings(
                slot_lists={
                    **slot_lists,
                    **intent_data.slot_lists,
                },
                expansion_rules={
                    **expansion_rules,
                    **intent_data.expansion_rules,
                },
                ignore_whitespace=intents.settings.ignore_whitespace,
                allow_unmatched_entities=allow_unmatched_entities,
                language=language or intents.language,
            )

            available_intents.append((intent, intent_data, match_settings, None))

    # Skip sentence templates whose required literal text is missing from the
    # input. Unlike a regex built from the template, this is a sound filter: a
    # template is only dropped when it provably cannot match, so no result is
    # lost. It therefore does not need a "nothing matched, try everything again"
    # fallback, and applies with allow_unmatched_entities too.
    filtered_intents: MutableSequence[
        Tuple[Intent, IntentData, MatchSettings, Optional[List[Sentence]]]
    ] = []

    for intent, intent_data, match_settings, _intent_sentences in available_intents:
        matching_intent_sentences = [
            intent_sentence
            for intent_sentence in intent_data.sentences
            if _has_required_fragments(
                intent_sentence.get_required_clauses(match_settings.expansion_rules),
                prefilter_haystack,
            )
        ]

        if matching_intent_sentences:
            filtered_intents.append(
                (intent, intent_data, match_settings, matching_intent_sentences)
            )

    available_intents = filtered_intents

    # Fall back to string matcher
    def make_matchable(tracked: TrackedText) -> TrackedText:
        """Apply the final match-text tweaks, keeping offsets aligned."""
        matchable = tracked.copy()
        if intents.settings.ignore_whitespace:
            matchable.sub(CJK_WHITESPACE)
        else:
            # Artifical word boundary
            matchable.append(" ")

        return matchable

    tracked_no_skip_words = make_matchable(tracked_no_skip_words)
    tracked_with_skip_words_matchable = make_matchable(tracked_with_skip_words)

    text_no_skip_words = tracked_no_skip_words.text

    tracked_at_start: Optional[TrackedText] = None
    tracked_at_end: Optional[TrackedText] = None

    def is_wildcard(e: Expression) -> bool:
        return isinstance(e, ListReference) and isinstance(
            slot_lists.get(e.list_name), WildcardSlotList
        )

    for intent, intent_data, match_settings, intent_sentences in available_intents:
        if not intent_sentences:
            intent_sentences = intent_data.sentences

        # Check each sentence template
        for intent_sentence in intent_sentences:
            tracked_match = tracked_no_skip_words
            if isinstance(intent_sentence.expression, Sequence):
                seq: Sequence = intent_sentence.expression
                if (len(seq.items) == 1) and is_wildcard(seq.items[0]):
                    # Entire sentence is a wild card
                    tracked_match = tracked_with_skip_words
                elif len(seq.items) > 1:
                    if is_wildcard(seq.items[0]):
                        # Starts with a wildcard
                        if tracked_at_end is None:
                            tracked_at_end = tracked_with_skip_words.copy()
                            remove_skip_words_tracked(
                                tracked_at_end,
                                skip_words,
                                intents.settings.ignore_whitespace,
                                start=False,
                            )
                            tracked_at_end = make_matchable(tracked_at_end)
                        tracked_match = tracked_at_end
                    elif is_wildcard(seq.items[-1]):
                        # Ends with a wildcard
                        if tracked_at_start is None:
                            tracked_at_start = tracked_with_skip_words.copy()
                            remove_skip_words_tracked(
                                tracked_at_start,
                                skip_words,
                                intents.settings.ignore_whitespace,
                                end=False,
                            )
                            tracked_at_start = make_matchable(tracked_at_start)
                        tracked_match = tracked_at_start

            # Text(s) to attempt matching against.
            tracked_matches = [tracked_match]
            if had_skip_words_removed and (
                tracked_match.text == tracked_no_skip_words.text
            ):
                # A skip word may actually be part of this (non-wildcard)
                # template, so try matching with skip words left in *first*.
                # The literal interpretation (what the user actually said) is
                # preferred over the one where words were dropped as filler.
                # Wildcard sentences already keep skip words where appropriate.
                tracked_matches.insert(0, tracked_with_skip_words_matchable)

            for candidate in tracked_matches:
                # Create initial context
                match_context = MatchContext(
                    text=candidate.text,
                    intent_context=intent_context,
                    intent_sentence=intent_sentence,
                    intent_data=intent_data,
                    original_text=text,
                )
                maybe_match_contexts = match_expression(
                    match_settings, match_context, intent_sentence.expression
                )
                # Entity spans come out in candidate-text coordinates; the offset
                # map puts them back into original-text coordinates.
                yield from _process_match_contexts(
                    maybe_match_contexts,
                    intent,
                    intent_data,
                    text_offsets=candidate.offsets,
                    default_response=default_response,
                    allow_unmatched_entities=allow_unmatched_entities,
                )


def _has_required_fragments(
    required_clauses: FrozenSet[FrozenSet[str]], haystack: str
) -> bool:
    """Return True if the haystack satisfies every clause (see get_required_clauses)."""
    for clause in required_clauses:
        for fragment in clause:
            if fragment in haystack:
                break
        else:
            # No fragment from this clause is present, so the template
            # cannot possibly match.
            return False

    return True


def _translate_text_span(
    text_span: Tuple[int, int], text_offsets: List[int], original_len: int
) -> Tuple[int, int]:
    """Map a span in the match text back to a span in the original text."""
    start, end = text_span
    num_offsets = len(text_offsets)

    if start < num_offsets:
        new_start = text_offsets[start]
    else:
        new_start = original_len

    # end is exclusive, so translate the last included character.
    if end > start:
        last = end - 1
        new_end = (text_offsets[last] + 1) if last < num_offsets else original_len
    else:
        new_end = new_start

    return (min(new_start, original_len), min(max(new_end, new_start), original_len))


def _finalize_entity_spans(
    entities: Iterable[MatchEntity],
    text_offsets: Optional[List[int]],
    original_len: int,
) -> None:
    """Trim wildcard text and map every span into original-text coordinates."""
    for entity in entities:
        if entity.is_wildcard:
            # Wildcards absorb the whitespace up to the next literal chunk. Trim
            # it, keeping text_span in step so it still delimits entity.text.
            raw_text = entity.text
            trimmed = raw_text.strip()
            if (entity.text_span is not None) and (raw_text != trimmed):
                lead = len(raw_text) - len(raw_text.lstrip())
                start = entity.text_span[0] + lead
                entity.text_span = (start, start + len(trimmed))

            entity.text = trimmed
            if isinstance(entity.value, str):
                entity.value = entity.value.strip()

        if (text_offsets is not None) and (entity.text_span is not None):
            entity.text_span = _translate_text_span(
                entity.text_span, text_offsets, original_len
            )


def _process_match_contexts(
    match_contexts: Iterable[MatchContext],
    intent: Intent,
    intent_data: IntentData,
    text_offsets: Optional[List[int]] = None,
    default_response: Optional[str] = None,
    allow_unmatched_entities: bool = False,
) -> Iterable[RecognizeResult]:
    for maybe_match_context in match_contexts:
        # Close any open wildcards or unmatched entities
        final_text = maybe_match_context.text.strip()
        if final_text:
            if unmatched_entity := maybe_match_context.get_open_entity():
                # Consume the rest of the text (unmatched entity)
                unmatched_entity.text += final_text
                unmatched_entity.is_open = False
                maybe_match_context.text = ""
            elif wildcard := maybe_match_context.get_open_wildcard():
                # Consume the rest of the text (wildcard)
                wildcard.text += final_text
                wildcard.value = wildcard.text
                wildcard.is_wildcard_open = False
                if wildcard.text_span is not None:
                    wildcard.text_span = (
                        wildcard.text_span[0],
                        wildcard.text_span[0] + len(wildcard.text),
                    )
                maybe_match_context.text = ""

        if not maybe_match_context.is_match:
            # Incomplete match with text still left at the end
            continue

        # Verify excluded context
        if intent_data.excludes_context and (
            not check_excluded_context(
                intent_data.excludes_context,
                maybe_match_context.intent_context,
            )
        ):
            continue

        # Verify required context
        slots_from_context: List[MatchEntity] = []
        if intent_data.requires_context and (
            not _copy_and_check_required_context(
                intent_data.requires_context,
                maybe_match_context,
                slots_from_context,
                allow_unmatched_entities=allow_unmatched_entities,
            )
        ):
            continue

        # Clean up wildcard entities and put spans in original-text coordinates
        _finalize_entity_spans(
            maybe_match_context.entities,
            text_offsets,
            len(maybe_match_context.original_text),
        )

        # Add fixed entities
        entity_names = set(entity.name for entity in maybe_match_context.entities)
        for slot_name, slot_value in intent_data.slots.items():
            if slot_name not in entity_names:
                maybe_match_context.entities.append(
                    MatchEntity(name=slot_name, value=slot_value, text="")
                )

        # Add context slots
        for slot_entity in slots_from_context:
            if slot_entity.name not in entity_names:
                maybe_match_context.entities.append(slot_entity)

        # Return each match
        response = default_response
        if intent_data.response is not None:
            response = intent_data.response

        intent_metadata: Optional[Dict[str, Any]] = None
        if maybe_match_context.intent_data is not None:
            intent_metadata = maybe_match_context.intent_data.metadata

        yield RecognizeResult(
            intent=intent,
            intent_data=intent_data,
            entities={entity.name: entity for entity in maybe_match_context.entities},
            entities_list=maybe_match_context.entities,
            response=response,
            context=maybe_match_context.intent_context,
            unmatched_entities={
                entity.name: entity for entity in maybe_match_context.unmatched_entities
            },
            unmatched_entities_list=maybe_match_context.unmatched_entities,
            text_chunks_matched=maybe_match_context.text_chunks_matched,
            intent_sentence=maybe_match_context.intent_sentence,
            intent_metadata=intent_metadata,
            captures={
                capture.name: capture for capture in maybe_match_context.captures
            },
            captures_list=maybe_match_context.captures,
            original_text=maybe_match_context.original_text,
        )


def is_match(
    text: str,
    sentence: Sentence,
    slot_lists: Optional[Dict[str, SlotList]] = None,
    expansion_rules: Optional[Dict[str, Sentence]] = None,
    skip_words: Optional[Iterable[str]] = None,
    entities: Optional[Dict[str, Any]] = None,
    intent_context: Optional[Dict[str, Any]] = None,
    ignore_whitespace: bool = False,
    allow_unmatched_entities: bool = False,
    language: Optional[str] = None,
) -> Optional[MatchContext]:
    """Return the first match of input text/words against a sentence expression."""
    tracked = normalize_for_matching(text)

    if skip_words:
        remove_skip_words_tracked(tracked, skip_words, ignore_whitespace)

    if ignore_whitespace:
        tracked.sub(CJK_WHITESPACE)
    else:
        # Artifical word boundary
        tracked.append(" ")

    text = tracked.text

    if slot_lists is None:
        slot_lists = {}

    if expansion_rules is None:
        expansion_rules = {}

    if intent_context is None:
        intent_context = {}

    settings = MatchSettings(
        slot_lists=slot_lists,
        expansion_rules=expansion_rules,
        ignore_whitespace=ignore_whitespace,
        allow_unmatched_entities=allow_unmatched_entities,
        language=language,
    )

    match_context = MatchContext(
        text=text,
        intent_context=intent_context,
        intent_sentence=sentence,
    )

    for maybe_match_context in match_expression(
        settings, match_context, sentence.expression
    ):
        if maybe_match_context.is_match:
            return maybe_match_context

    return None


def _copy_and_check_required_context(
    required_context: Dict[str, Any],
    maybe_match_context: MatchContext,
    slots_from_context: List[MatchEntity],
    allow_unmatched_entities: bool = False,
) -> bool:
    """Check required context and copy slots into new entities."""
    for (
        context_key,
        context_value,
    ) in required_context.items():
        copy_to_slot: Optional[str] = None
        if isinstance(context_value, collections.abc.Mapping):
            # Unpack dict
            # <context_key>:
            #   value: ...
            #   slot: true/false or "name"
            maybe_copy_to_slot = context_value.get("slot")
            if isinstance(maybe_copy_to_slot, str):
                # Slot name provided
                copy_to_slot = maybe_copy_to_slot
            elif maybe_copy_to_slot:
                # True
                copy_to_slot = context_key

            context_value = context_value.get("value")

        actual_value = maybe_match_context.intent_context.get(context_key)
        actual_text = ""
        actual_metadata: Optional[Dict[str, Any]] = None

        if isinstance(actual_value, collections.abc.Mapping):
            # Unpack dict
            actual_text = actual_value.get("text", "")
            actual_metadata = actual_value.get("metadata")
            actual_value = actual_value.get("value")

        if allow_unmatched_entities and (actual_value is None):
            # Look in unmatched entities
            for unmatched_context_entity in maybe_match_context.unmatched_entities:
                if (unmatched_context_entity.name == context_key) and isinstance(
                    unmatched_context_entity, UnmatchedTextEntity
                ):
                    actual_value = unmatched_context_entity.text
                    break

        if (actual_value == context_value) and (context_value is not None):
            # Exact match to context value, except when context value is required and not provided
            if copy_to_slot:
                slots_from_context.append(
                    MatchEntity(
                        name=copy_to_slot,
                        value=actual_value,
                        text=actual_text,
                        metadata=actual_metadata,
                    )
                )
            continue

        if (context_value is None) and (actual_value is not None):
            # Any value matches, as long as it's set
            if copy_to_slot:
                slots_from_context.append(
                    MatchEntity(
                        name=copy_to_slot,
                        value=actual_value,
                        text=actual_text,
                        metadata=actual_metadata,
                    )
                )
            continue

        if (
            isinstance(context_value, collections.abc.Collection)
            and not isinstance(context_value, str)
            and (actual_value in context_value)
        ):
            # Actual value was in context value list
            if copy_to_slot:
                slots_from_context.append(
                    MatchEntity(
                        name=copy_to_slot,
                        value=actual_value,
                        text=actual_text,
                        metadata=actual_metadata,
                    )
                )
            continue

        if allow_unmatched_entities:
            # Create missing entity as unmatched
            has_unmatched_entity = False
            for unmatched_context_entity in maybe_match_context.unmatched_entities:
                if unmatched_context_entity.name == context_key:
                    has_unmatched_entity = True
                    break

            if not has_unmatched_entity:
                maybe_match_context.unmatched_entities.append(
                    UnmatchedTextEntity(
                        name=context_key,
                        text=MISSING_ENTITY,
                        is_open=False,
                    )
                )
        else:
            # Did not match required context
            return False

    return True


def recognize_best(
    text: str,
    intents: Intents,
    slot_lists: Optional[Dict[str, SlotList]] = None,
    expansion_rules: Optional[Dict[str, Sentence]] = None,
    skip_words: Optional[Iterable[str]] = None,
    intent_context: Optional[Dict[str, Any]] = None,
    default_response: Optional[str] = "default",
    allow_unmatched_entities: bool = False,
    language: Optional[str] = None,
    best_metadata_key: Optional[str] = None,
    best_slot_name: Optional[str] = None,
) -> Optional[RecognizeResult]:
    """Find the best result with the following priorities:

    1. The result that has "best_metadata_key" in its metadata
    2. The result that has an entity for "best_slot_name" and longest text
    3. The result that matches the most literal text

    See "recognize_all" for other parameters.
    """
    metadata_found = False
    slot_found = False
    best_results: List[RecognizeResult] = []
    best_slot_quality: Optional[int] = None

    for result in recognize_all(
        text,
        intents,
        slot_lists=slot_lists,
        expansion_rules=expansion_rules,
        skip_words=skip_words,
        intent_context=intent_context,
        default_response=default_response,
        allow_unmatched_entities=allow_unmatched_entities,
        language=language,
    ):
        # Prioritize intents with a specific metadata key
        if best_metadata_key is not None:
            is_metadata = (
                result.intent_metadata is not None
                and result.intent_metadata.get(best_metadata_key)
            )

            if metadata_found and not is_metadata:
                continue

            if (not metadata_found) and is_metadata:
                metadata_found = True

                # Clear builtin results
                slot_found = False
                best_results = []
                best_slot_quality = None

        # Prioritize results with a specific slot
        if best_slot_name:
            entity = result.entities.get(best_slot_name)
            is_slot = (entity is not None) and not entity.is_wildcard

            if slot_found and not is_slot:
                continue

            if (not slot_found) and is_slot:
                slot_found = True

                # Clear non-slot results
                best_results = []

            if is_slot and (entity is not None) and isinstance(entity.value, str):
                # Prioritize results with a better slot value
                slot_quality = len(entity.text)
                if (best_slot_quality is None) or (slot_quality > best_slot_quality):
                    best_slot_quality = slot_quality

                    # Clear worse slot results
                    best_results = []
                elif slot_quality < best_slot_quality:
                    continue

        # Accumulate results. We will resolve the ambiguity below.
        best_results.append(result)

    if best_results:
        # Prioritize matches with fewer wildcards and more literal text matched.
        return sorted(best_results, key=_get_result_score)[0]

    return None


def _get_result_score(result: RecognizeResult) -> Tuple[int, int, int, str]:
    """Get sort score for a result.

    Sorted lowest first:

    1. Fewer wildcards.
    2. More literal text matched (negated).
    3. Less text captured by wildcards. This breaks ties in favor of the parse
       that binds a word to a concrete list slot instead of letting an adjacent
       wildcard swallow it (e.g. "play track Yesterday" -> media_class="track",
       search_query="Yesterday" rather than search_query="track Yesterday").
    4. Intent name, as a final tiebreaker for deterministic ordering. Without
       it, fully-tied results are decided by intent iteration order, which can
       vary across platforms (e.g. aarch64 vs x86_64).

    Note: criterion 3 is 0 for every candidate when none use wildcards, so
    parses made up entirely of list/range slots keep their original ordering
    apart from the intent-name tiebreaker.
    """
    num_wildcards = sum(1 for e in result.entities_list if e.is_wildcard)
    wildcard_text_len = sum(
        len(e.text or "") for e in result.entities_list if e.is_wildcard
    )
    return (
        num_wildcards,
        -result.text_chunks_matched,
        wildcard_text_len,
        result.intent.name,
    )
