import io
import json
import itertools
import sqlite3
from collections import defaultdict
from pathlib import Path

import pytest
from yaml import safe_load

from hassil import Intents, TextSlotList
from hassil.expression import Group
from hassil.fuzzy import FuzzyNgramMatcher, SlotCombinationInfo, FuzzySlotValue
from hassil.intents import WildcardSlotList
from hassil.ngram import Sqlite3NgramModel, NgramModel

LISTS_YAML = """
lists:
  name:
    values:
      - in: "Front Door"
        out: "Front Door"
        context:
          domain: "lock"
        metadata:
          domain: "lock"
      - in: "Kitchen Door"
        out: "Kitchen Door"
        context:
          domain: "lock"
        metadata:
          domain: "lock"
      - in: "Poorly Installed Door"
        out: "Poorly Installed Door"
        context:
          domain: "lock"
        metadata:
          domain: "lock"
      - in: "Openable Lock"
        out: "Openable Lock"
        context:
          domain: "lock"
        metadata:
          domain: "lock"
      - in: "Humidifier"
        out: "Humidifier"
        context:
          domain: "humidifier"
        metadata:
          domain: "humidifier"
      - in: "Dehumidifier"
        out: "Dehumidifier"
        context:
          domain: "humidifier"
        metadata:
          domain: "humidifier"
      - in: "Hygrostat"
        out: "Hygrostat"
        context:
          domain: "humidifier"
        metadata:
          domain: "humidifier"
      - in: "Demo Water Heater"
        out: "Demo Water Heater"
        context:
          domain: "water_heater"
        metadata:
          domain: "water_heater"
      - in: "Demo Water Heater Celsius"
        out: "Demo Water Heater Celsius"
        context:
          domain: "water_heater"
        metadata:
          domain: "water_heater"
      - in: "Pergola Roof"
        out: "Pergola Roof"
        context:
          domain: "cover"
          cover_supports_position: false
        metadata:
          domain: "cover"
      - in: "Heat pump"
        out: "Heat pump"
        context:
          domain: "climate"
        metadata:
          domain: "climate"
      - in: "Hvac"
        out: "Hvac"
        context:
          domain: "climate"
        metadata:
          domain: "climate"
      - in: "Ecobee"
        out: "Ecobee"
        context:
          domain: "climate"
        metadata:
          domain: "climate"
      - in: "Overhead light"
        out: "Overhead light"
        context:
          domain: "light"
          light_supports_color: true
          light_supports_brightness: true
        metadata:
          domain: "light"
      - in: "Bedroom Light"
        out: "Bedroom Light"
        context:
          domain: "light"
          light_supports_color: true
          light_supports_brightness: true
        metadata:
          domain: "light"
      - in: "Kitchen Lights"
        out: "Kitchen Lights"
        context:
          domain: "light"
          light_supports_color: true
          light_supports_brightness: true
        metadata:
          domain: "light"
      - in: "Office Light"
        out: "Office Light"
        context:
          domain: "light"
          light_supports_color: true
          light_supports_brightness: true
        metadata:
          domain: "light"
      - in: "Living Room Lights"
        out: "Living Room Lights"
        context:
          domain: "light"
          light_supports_color: true
          light_supports_brightness: true
        metadata:
          domain: "light"
      - in: "Entrance Color + White Lights"
        out: "Entrance Color + White Lights"
        context:
          domain: "light"
          light_supports_color: true
          light_supports_brightness: true
        metadata:
          domain: "light"
      - in: "Outside Temperature"
        out: "Outside Temperature"
        context:
          domain: "sensor"
        metadata:
          domain: "sensor"
      - in: "Outside Humidity"
        out: "Outside Humidity"
        context:
          domain: "sensor"
        metadata:
          domain: "sensor"
      - in: "Kitchen Window"
        out: "Kitchen Window"
        context:
          domain: "cover"
          cover_supports_position: false
        metadata:
          domain: "cover"
      - in: "Hall Window"
        out: "Hall Window"
        context:
          domain: "cover"
          cover_supports_position: true
        metadata:
          domain: "cover"
      - in: "Living Room Window"
        out: "Living Room Window"
        context:
          domain: "cover"
          cover_supports_position: true
        metadata:
          domain: "cover"
      - in: "Garage Door"
        out: "Garage Door"
        context:
          domain: "cover"
          cover_supports_position: false
        metadata:
          domain: "cover"
      - in: "Living Room Fan"
        out: "Living Room Fan"
        context:
          domain: "fan"
          fan_supports_speed: true
        metadata:
          domain: "fan"
      - in: "Ceiling Fan"
        out: "Ceiling Fan"
        context:
          domain: "fan"
          fan_supports_speed: true
        metadata:
          domain: "fan"
      - in: "Percentage Full Fan"
        out: "Percentage Full Fan"
        context:
          domain: "fan"
          fan_supports_speed: true
        metadata:
          domain: "fan"
      - in: "Percentage Limited Fan"
        out: "Percentage Limited Fan"
        context:
          domain: "fan"
          fan_supports_speed: true
        metadata:
          domain: "fan"
      - in: "Preset Only Limited Fan"
        out: "Preset Only Limited Fan"
        context:
          domain: "fan"
          fan_supports_speed: false
        metadata:
          domain: "fan"
      - in: "Decorative Lights"
        out: "Decorative Lights"
        context:
          domain: "switch"
        metadata:
          domain: "switch"
      - in: "A.C."
        out: "A.C."
        context:
          domain: "switch"
        metadata:
          domain: "switch"
      - in: "Search"
        out: "Search"
        context:
          domain: "media_player"
          media_player_supports_pause: false
          media_player_supports_volume_set: false
          media_player_supports_next_track: false
        metadata:
          domain: "media_player"
      - in: "Leaving the house"
        out: "Leaving the house"
        context:
          domain: "script"
        metadata:
          domain: "script"
      - in: "Shopping List"
        out: "Shopping List"
        context:
          domain: "todo"
        metadata:
          domain: "todo"
      - in: "Downstairs Chromecast"
        out: "Downstairs Chromecast"
        context:
          domain: "media_player"
          media_player_supports_pause: true
          media_player_supports_volume_set: false
          media_player_supports_next_track: false
        metadata:
          domain: "media_player"
      - in: "Demo Weather South"
        out: "Demo Weather South"
        context:
          domain: "weather"
        metadata:
          domain: "weather"
      - in: "Liste des courses"
        out: "Liste des courses"
        context:
          domain: "todo"
        metadata:
          domain: "todo"
      - in: "Todo List 1"
        out: "Todo List 1"
        context:
          domain: "todo"
        metadata:
          domain: "todo"
      - in: "party time"
        out: "party time"
        context:
          domain: "scene"
        metadata:
          domain: "scene"
      - in: "TV"
        out: "Family Room Google TV"
        context:
          domain: "media_player"
          media_player_supports_pause: true
          media_player_supports_volume_set: false
          media_player_supports_next_track: false
        metadata:
          domain: "media_player"
      - in: "Media Player"
        out: "Media Player"
        context:
          domain: "media_player"
          media_player_supports_pause: true
          media_player_supports_volume_set: true
          media_player_supports_next_track: false
        metadata:
          domain: "media_player"
  area:
    values:
      - "Living Room"
      - "Kitchen"
      - "cuisine"
      - "Bedroom"
      - "Garage"
      - "Entrance"
      - "Office"
      - "kontoret"
      - "Basement"
      - "Test Area"
  floor:
    values:
      - "First floor"
"""


@pytest.fixture(name="matcher", scope="session")
def matcher_fixture() -> FuzzyNgramMatcher:
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

            # TODO
            if ("name" in combo_key) and ("Timer" in intent_name):
                # Ignore wildcard names
                continue

            name_domains = combo_info.get("name_domains")
            if name_domains:
                name_domains = set(itertools.chain.from_iterable(name_domains.values()))

            slot_combinations[intent_name][combo_key].append(
                SlotCombinationInfo(name_domains)
            )

    intents = Intents.from_files(
        Path("/home/hansenm/opt/intent-sentences/sentences/en").glob("*.yaml")
    )

    with io.StringIO(LISTS_YAML) as f:
        lists_dict = safe_load(f)["lists"]

    intents.slot_lists["name"] = TextSlotList.from_tuples(
        (name_info["in"], name_info["out"], name_info["context"])
        for name_info in lists_dict["name"]["values"]
    )
    intents.slot_lists["area"] = TextSlotList.from_strings(lists_dict["area"]["values"])
    intents.slot_lists["floor"] = TextSlotList.from_strings(
        lists_dict["floor"]["values"]
    )

    # intents.slot_lists["name"] = TextSlotList.from_tuples(
    #     [
    #         ("living room lamp", "living room lamp", {"domain": "light"}),
    #         ("tv", "tv", {"domain": "media_player"}),
    #         ("garage door", "garage door", {"domain": "cover"}),
    #     ]
    # )
    # intents.slot_lists["area"] = TextSlotList.from_strings(
    #     ["kitchen", "living room", "office"]
    # )

    intent_slot_list_names = defaultdict(set)
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

                for list_ref in sentence.expression.list_references(expansion_rules):
                    if list_ref.slot_name not in intent_slot_names:
                        continue

                    slot_list = intent_data.slot_lists.get(list_ref.list_name)
                    if slot_list is None:
                        slot_list = intents.slot_lists.get(list_ref.list_name)

                    if (slot_list is None) or (slot_list is WildcardSlotList):
                        continue

                    intent_slot_list_names[list_ref.list_name].add(list_ref.slot_name)

    for rule_body in intents.expansion_rules.values():
        if not isinstance(rule_body.expression, Group):
            continue

        for list_ref in rule_body.expression.list_references(intents.expansion_rules):
            if list_ref.slot_name in intent_slot_names:
                intent_slot_list_names[list_ref.list_name].add(list_ref.slot_name)

    intent_models = {}
    for db_path in Path(
        "/home/hansenm/opt/intents-package/home_assistant_intents/fuzzy/en/ngram"
    ).glob("*.db"):
        intent_name = db_path.stem
        db_config_path = db_path.with_suffix(".json")
        with open(db_config_path, "r", encoding="utf-8") as db_config_file:
            db_config_dict = json.load(db_config_file)
        intent_models[intent_name] = Sqlite3NgramModel(
            order=db_config_dict["order"],
            words={
                word: str(word_id)
                for word, word_id in db_config_dict["words"].items()
            },
            database_path=db_path,
        )
        # ngram_path = db_path.with_suffix(".ngram.json")
        # with open(ngram_path, "r", encoding="utf-8") as db_config_file:
        #     ngram_dict = json.load(db_config_file)
        #     intent_models[intent_name] = NgramModel(
        #         order=ngram_dict["order"],
        #         probs={
        #             tuple(combo_key.split()): combo_probs
        #             for combo_key, combo_probs in ngram_dict["probs"].items()
        #         },
        #     )

    matcher = FuzzyNgramMatcher(
        intents,
        intent_models,
        intent_slot_list_names,
        slot_combinations,
        domain_keywords={
            "light": ["light", "lights"],
            "fan": ["fan", "fans"],
            "cover": [
                "window",
                "windows",
                "curtain",
                "curtains",
                "door",
                "doors",
                "garage door",
            ],
        },
    )

    return matcher


def test_domain_only(matcher: FuzzyNgramMatcher) -> None:
    result = matcher.match("turn on the lights in this room")
    assert result is not None
    assert result.intent_name == "HassTurnOn"
    assert result.slots.keys() == {"domain"}
    assert result.slots["domain"] == FuzzySlotValue(value="light", text="lights")


def test_name_only(matcher: FuzzyNgramMatcher) -> None:
    result = matcher.match("turn off that tv right now")
    assert result is not None
    assert result.intent_name == "HassTurnOff"
    assert result.slots.keys() == {"name"}
    assert result.slots["name"] == FuzzySlotValue(
        value="Family Room Google TV", text="TV"
    )
    assert result.name_domain == "media_player"


def test_name_area(matcher: FuzzyNgramMatcher) -> None:
    result = matcher.match("kitchen A.C. on")
    assert result is not None
    assert result.intent_name == "HassTurnOn"
    assert result.slots.keys() == {"name", "area"}
    assert result.slots["name"] == FuzzySlotValue(value="A.C.", text="A.C.")
    assert result.slots["area"] == FuzzySlotValue(value="Kitchen", text="Kitchen")
    assert result.name_domain == "switch"


def test_domain_area_device_class(matcher: FuzzyNgramMatcher) -> None:
    result = matcher.match("open up all of the windows in the kitchen area please")
    assert result is not None
    assert result.intent_name == "HassTurnOn"
    assert result.slots.keys() == {"domain", "area", "device_class"}
    assert result.slots["domain"] == FuzzySlotValue(value="cover", text="windows")
    assert result.slots["device_class"] == FuzzySlotValue(
        value="window", text="windows"
    )
    assert result.slots["area"] == FuzzySlotValue(value="Kitchen", text="Kitchen")


def test_brightness(matcher: FuzzyNgramMatcher) -> None:
    result = matcher.match("overhead light 50% brightness")
    assert result is not None
    assert result.intent_name == "HassLightSet"
    assert result.slots.keys() == {"name", "brightness"}
    assert result.slots["name"] == FuzzySlotValue(
        value="Overhead light", text="Overhead light"
    )
    assert result.slots["brightness"] == FuzzySlotValue(value=50, text="50%")
    assert result.name_domain == "light"


def test_temperature(matcher: FuzzyNgramMatcher) -> None:
    result = matcher.match("how is the temperature")
    assert result is not None
    assert result.intent_name == "HassClimateGetTemperature"
    assert not result.slots


def test_temperature_name(matcher: FuzzyNgramMatcher) -> None:
    result = matcher.match("ecobee temp")
    assert result is not None
    assert result.intent_name == "HassClimateGetTemperature"
    assert result.slots.keys() == {"name"}
    assert result.slots["name"] == FuzzySlotValue(value="Ecobee", text="Ecobee")
    assert result.name_domain == "climate"


# def test_start_timer(matcher: FuzzyNgramMatcher) -> None:
#     result = matcher.match("5 minute timer start")
#     assert result is not None
#     assert result.intent_name == "HassStartTimer"
#     assert result.slots.keys() == {"minutes"}
#     assert result.slots["minutes"] == FuzzySlotValue(value=5, text="5")


# def test_cancel_timer(matcher: FuzzyNgramMatcher) -> None:
#     result = matcher.match("cancel 5 minutes")
#     assert result is not None
#     assert result.intent_name == "HassCancelTimer"
#     assert result.slots.keys() == {"start_minutes"}
#     assert result.slots["start_minutes"] == FuzzySlotValue(value=5, text="5")


# def test_cancel_all_timer(matcher: FuzzyNgramMatcher) -> None:
#     result = matcher.match("stop all of these timers")
#     assert result is not None
#     assert result.intent_name == "HassCancelAllTimers"
#     assert not result.slots


# def test_increase_timer(matcher: FuzzyNgramMatcher) -> None:
#     result = matcher.match("add 5 minutes")
#     assert result is not None
#     assert result.intent_name == "HassIncreaseTimer"
#     assert result.slots.keys() == {"minutes"}
#     assert result.slots["minutes"] == FuzzySlotValue(value=5, text="5")


def test_get_time(matcher: FuzzyNgramMatcher) -> None:
    result = matcher.match("time now")
    assert result is not None
    assert result.intent_name == "HassGetCurrentTime"
    assert not result.slots


def test_get_date(matcher: FuzzyNgramMatcher) -> None:
    result = matcher.match("date today")
    assert result is not None
    assert result.intent_name == "HassGetCurrentDate"
    assert not result.slots


def test_weather(matcher: FuzzyNgramMatcher) -> None:
    result = matcher.match("weather")
    assert result is not None
    assert result.intent_name == "HassGetWeather"
    assert not result.slots


def test_weather_name(matcher: FuzzyNgramMatcher) -> None:
    result = matcher.match("demo weather south")
    assert result is not None
    assert result.intent_name == "HassGetWeather"
    assert result.slots.keys() == {"name"}
    assert result.slots["name"] == FuzzySlotValue(
        value="Demo Weather South", text="Demo Weather South"
    )
    assert result.name_domain == "weather"


def test_set_temperature(matcher: FuzzyNgramMatcher) -> None:
    result = matcher.match("make it 72 degrees")
    assert result is not None
    assert result.intent_name == "HassClimateSetTemperature"
    assert result.slots.keys() == {"temperature"}
    assert result.slots["temperature"] == FuzzySlotValue(value=72, text="72")


def test_set_color(matcher: FuzzyNgramMatcher) -> None:
    result = matcher.match("red bedroom")
    assert result is not None
    assert result.intent_name == "HassLightSet"
    assert result.slots.keys() == {"area", "color"}
    assert result.slots["area"] == FuzzySlotValue(value="Bedroom", text="Bedroom")
    assert result.slots["color"] == FuzzySlotValue(value="red", text="red")


def test_set_volume(matcher: FuzzyNgramMatcher) -> None:
    result = matcher.match("TV 50")
    assert result is not None
    assert result.intent_name == "HassSetVolume"
    assert result.slots.keys() == {"name", "volume_level"}
    assert result.slots["name"] == FuzzySlotValue(
        value="Family Room Google TV", text="TV"
    )
    assert result.slots["volume_level"] == FuzzySlotValue(value=50, text="50")
    assert result.name_domain == "media_player"


def test_set_position(matcher: FuzzyNgramMatcher) -> None:
    result = matcher.match("hall window 50")
    assert result is not None
    assert result.intent_name == "HassSetPosition"
    assert result.slots.keys() == {"name", "position"}
    assert result.slots["name"] == FuzzySlotValue(
        value="Hall Window", text="Hall Window"
    )
    assert result.slots["position"] == FuzzySlotValue(value=50, text="50")
    assert result.name_domain == "cover"


def test_degrees(matcher: FuzzyNgramMatcher) -> None:
    result = matcher.match("72°")
    assert result is not None
    assert result.intent_name == "HassClimateSetTemperature"
    assert result.slots.keys() == {"temperature"}
    assert result.slots["temperature"] == FuzzySlotValue(value=72, text="72°")


def test_nevermind(matcher: FuzzyNgramMatcher) -> None:
    result = matcher.match("nevermind, it's working")
    assert result is not None
    assert result.intent_name == "HassNevermind"
    assert not result.slots.keys()


def test_scene(matcher: FuzzyNgramMatcher) -> None:
    result = matcher.match("party time, excellent")
    assert result is not None
    assert result.intent_name == "HassTurnOn"
    assert result.slots.keys() == {"name"}
    assert result.slots["name"] == FuzzySlotValue(value="party time", text="party time")
    assert result.name_domain == "scene"
