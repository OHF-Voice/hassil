"""Tests for fuzzy matching."""

from pathlib import Path

import pytest
from home_assistant_intents import get_fuzzy_config, get_fuzzy_language, get_intents
from yaml import safe_load

from hassil import Intents, TextSlotList
from hassil.fuzzy import FuzzyNgramMatcher, FuzzySlotValue, SlotCombinationInfo
from hassil.ngram import Sqlite3NgramModel

_TESTS_DIR = Path(__file__).parent


@pytest.fixture(name="matcher", scope="session")
def matcher_fixture() -> FuzzyNgramMatcher:
    """Load fuzzy matcher for English."""
    intents_dict = get_intents("en")
    assert intents_dict
    intents = Intents.from_dict(intents_dict)

    with open(
        _TESTS_DIR / "test_fixtures.yaml", "r", encoding="utf-8"
    ) as fixtures_file:
        lists_dict = safe_load(fixtures_file)["lists"]

    intents.slot_lists["name"] = TextSlotList.from_tuples(
        (name_info["in"], name_info["out"], name_info["context"])
        for name_info in lists_dict["name"]["values"]
    )
    intents.slot_lists["area"] = TextSlotList.from_strings(lists_dict["area"]["values"])
    intents.slot_lists["floor"] = TextSlotList.from_strings(
        lists_dict["floor"]["values"]
    )

    fuzzy_config = get_fuzzy_config()
    lang_config = get_fuzzy_language("en")
    assert lang_config is not None

    matcher = FuzzyNgramMatcher(
        intents=intents,
        intent_models={
            intent_name: Sqlite3NgramModel(
                order=fuzzy_model.order,
                words={
                    word: str(word_id) for word, word_id in fuzzy_model.words.items()
                },
                database_path=fuzzy_model.database_path,
            )
            for intent_name, fuzzy_model in lang_config.ngram_models.items()
        },
        intent_slot_list_names=fuzzy_config.slot_list_names,
        slot_combinations={
            intent_name: {
                combo_key: [
                    SlotCombinationInfo(
                        name_domains=set(name_domains) if name_domains else None
                    )
                ]
                for combo_key, name_domains in intent_combos.items()
            }
            for intent_name, intent_combos in fuzzy_config.slot_combinations.items()
        },
        domain_keywords=lang_config.domain_keywords,
        stop_words=lang_config.stop_words,
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
    result = matcher.match("how about demo weather south")
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
    result = matcher.match("set 72°")
    assert result is not None
    assert result.intent_name == "HassClimateSetTemperature"
    assert result.slots.keys() == {"temperature"}
    assert result.slots["temperature"] == FuzzySlotValue(value=72, text="72°")


def test_nevermind(matcher: FuzzyNgramMatcher) -> None:
    result = matcher.match("oh nevermind")
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


def test_timer_status(matcher: FuzzyNgramMatcher) -> None:
    result = matcher.match("what about my timers")
    assert result is not None
    assert result.intent_name == "HassTimerStatus"
    assert not result.slots


def test_wrong_vocab(matcher: FuzzyNgramMatcher) -> None:
    assert not matcher.match("open office lights")
    assert not matcher.match("close A.C.")
    assert not matcher.match("garage door off")

    result = matcher.match("close front door")
    assert result is not None
    assert result.intent_name == "HassTurnOff"
    assert result.slots.keys() == {"device_class", "domain"}
    assert result.slots["device_class"] == FuzzySlotValue(value="door", text="door")
    assert result.slots["domain"] == FuzzySlotValue(value="cover", text="door")


def test_stop_words(matcher: FuzzyNgramMatcher) -> None:
    result = matcher.match(
        "hi there, so pls could you turn the A.C. off just now lol ok"
    )
    assert result is not None
    assert result.intent_name == "HassTurnOff"
    assert result.slots.keys() == {"name"}
    assert result.slots["name"] == FuzzySlotValue(value="A.C.", text="A.C.")
