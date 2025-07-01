from collections import defaultdict
import itertools
import time
from pathlib import Path

from yaml import safe_load

from hassil import Intents, TextSlotList
from hassil.expression import Group
from hassil.fuzzy import FuzzyNgramMatcher, SlotCombinationInfo


def test_match() -> None:
    with open(
        "/home/hansenm/opt/intent-sentences/intents.yaml", "r", encoding="utf-8"
    ) as f:
        intents_info = safe_load(f)

    intent_slot_names = set()
    slot_combinations = defaultdict(lambda: defaultdict(list))
    for intent_name, intent_info in intents_info.items():
        intent_slot_names.update(intent_info.get("slots", {}).keys())

        for combo_info in intent_info.get("slot_combinations", {}).values():
            combo_key = tuple(sorted(combo_info["slots"]))
            name_domains = combo_info.get("name_domains")
            if name_domains:
                name_domains = set(itertools.chain.from_iterable(name_domains.values()))

            slot_combinations[intent_name][combo_key].append(
                SlotCombinationInfo(name_domains)
            )

    intents = Intents.from_files(
        Path("/home/hansenm/opt/intent-sentences/sentences/en").glob("*.yaml")
    )
    intents.slot_lists["name"] = TextSlotList.from_tuples(
        [
            ("living room lamp", "living room lamp", {"domain": "light"}),
            ("tv", "tv", {"domain": "media_player"}),
            ("garage door", "garage door", {"domain": "cover"}),
        ]
    )
    intents.slot_lists["area"] = TextSlotList.from_strings(
        ["kitchen", "living room", "office"]
    )

    intent_slot_list_names = {}
    for intent_info in intents.intents.values():
        for intent_data in intent_info.data:
            if intent_data.expansion_rules:
                expansion_rules = {
                    **intents.expansion_rules,
                    **intent_data.expansion_rules,
                }
            else:
                expansion_rules = intents.expansion_rules

            for sentence in intent_data.sentences:
                if not isinstance(sentence.expression, Group):
                    continue

                # inferred_domain = intent_data.slots.get("domain")
                # if inferred_domain:
                #     print(sentence.text)

                for list_ref in sentence.expression.list_references(expansion_rules):
                    if list_ref.slot_name in intent_slot_names:
                        intent_slot_list_names[list_ref.list_name] = list_ref.slot_name

    for rule_body in intents.expansion_rules.values():
        if not isinstance(rule_body.expression, Group):
            continue

        for list_ref in rule_body.expression.list_references(intents.expansion_rules):
            if list_ref.slot_name in intent_slot_names:
                intent_slot_list_names[list_ref.list_name] = list_ref.slot_name

    # print(intent_slot_list_names)

    matcher = FuzzyNgramMatcher(
        intents,
        "/home/hansenm/opt/intent-sentences/fuzzy/en",
        intent_slot_list_names,
        slot_combinations,
        domain_keywords={
            "light": ["light", "lights"],
            "fan": ["fan", "fans"],
            "cover": ["window", "windows"],
        },
    )

    start_time = time.monotonic()
    matcher.match("open all of the windows in the  kitchen area")
    end_time = time.monotonic()
    print(end_time - start_time)
