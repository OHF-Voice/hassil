import io
from typing import Set, cast

import pytest

from hassil import Intents, recognize, recognize_all, recognize_best
from hassil.expression import TextChunk
from hassil.intents import TextSlotList
from hassil.models import MatchEntity, UnmatchedRangeEntity, UnmatchedTextEntity
from hassil.recognize import MISSING_ENTITY

TEST_YAML = """
language: "en"
intents:
  TurnOnTV:
    data:
      - sentences:
          - "turn on [the] TV in <area>"
          - "turn on <area> TV"
        slots:
          domain: "media_player"
          name: "roku"
  SetBrightness:
    data:
      - sentences:
          - "set [the] brightness in <area> to <brightness>"
        slots:
          domain: "light"
          name: "all"
      - sentences:
          - "set [the] brightness of <name> to <brightness>"
        requires_context:
          domain: "light"
        slots:
          domain: "light"
  GetTemperature:
    data:
      - sentences:
          - "<what_is> [the] temperature in <area>"
        slots:
          domain: "climate"
  CloseCover:
    data:
      - sentences:
          - "close <name>"
        requires_context:
          domain: "cover"
        slots:
          domain: "cover"
  Play:
    data:
      - sentences:
          - "play <name>"
        excludes_context:
          domain:
            - "cover"
            - "light"
  CloseCurtains:
    data:
      - sentences:
          - "close [the] curtains [in <area>]"
        slots:
          domain: "cover"
          device_class: "curtain"
        requires_context:
          area:
            slot: true
          not_copied: "not copied value"
          copied_to_different:
            value: null
            slot: "different_slot"
expansion_rules:
  area: "[the] {area}"
  name: "[the] {name}"
  brightness: "{brightness_pct}[%| percent]"
  what_is: "(what's | whats | what is)"
lists:
  brightness_pct:
    range:
      type: percentage
      from: 0
      to: 100
skip_words:
  - "please"
"""


@pytest.fixture
def intents():
    with io.StringIO(TEST_YAML) as test_file:
        return Intents.from_yaml(test_file)


@pytest.fixture
def slot_lists():
    return {
        "area": TextSlotList.from_tuples(
            [("kitchen", "area.kitchen"), ("living room", "area.living_room")]
        ),
        "name": TextSlotList.from_tuples(
            [
                ("hue", "light.hue", {"domain": "light"}),
                (
                    "garage door",
                    "cover.garage_door",
                    {"domain": "cover"},
                ),
                (
                    "blue curtains",
                    "cover.blue_curtains",
                    {
                        "domain": "cover",
                        "device_class": "curtain",
                        "area": "living_room",
                    },
                ),
                (
                    "roku",
                    "media_player.roku",
                    {"domain": "media_player"},
                ),
            ]
        ),
    }


# pylint: disable=redefined-outer-name
def test_turn_on(intents, slot_lists):
    result = recognize("turn on kitchen TV, please", intents, slot_lists=slot_lists)
    assert result is not None
    assert result.intent.name == "TurnOnTV"

    assert result.text_chunks_matched > 0

    area = result.entities["area"]
    assert area.name == "area"
    assert area.value == "area.kitchen"

    # From YAML
    assert result.entities["domain"].value == "media_player"
    assert result.entities["name"].value == "roku"


# pylint: disable=redefined-outer-name
def test_brightness_area(intents, slot_lists):
    result = recognize(
        "set the brightness in the living room to 75%", intents, slot_lists=slot_lists
    )
    assert result is not None
    assert result.intent.name == "SetBrightness"

    assert result.text_chunks_matched > 0

    assert result.entities["area"].value == "area.living_room"
    assert result.entities["brightness_pct"].value == 75

    # From YAML
    assert result.entities["domain"].value == "light"
    assert result.entities["name"].value == "all"


# pylint: disable=redefined-outer-name
def test_brightness_area_words(intents, slot_lists):
    result = recognize(
        "set brightness in the living room to forty-two percent",
        intents,
        slot_lists=slot_lists,
        language="en",
    )
    assert result is not None
    assert result.intent.name == "SetBrightness"

    assert result.entities["area"].value == "area.living_room"
    assert result.entities["brightness_pct"].value == 42

    # From YAML
    assert result.entities["domain"].value == "light"
    assert result.entities["name"].value == "all"


# pylint: disable=redefined-outer-name
def test_brightness_name(intents, slot_lists):
    result = recognize(
        "set brightness of the hue to 50%", intents, slot_lists=slot_lists
    )
    assert result is not None
    assert result.intent.name == "SetBrightness"

    assert result.entities["name"].value == "light.hue"
    assert result.entities["brightness_pct"].value == 50

    # From YAML
    assert result.entities["domain"].value == "light"


# pylint: disable=redefined-outer-name
def test_brightness_not_cover(intents, slot_lists):
    result = recognize(
        "set brightness of the garage door to 50%", intents, slot_lists=slot_lists
    )
    assert result is None


# pylint: disable=redefined-outer-name
def test_temperature(intents, slot_lists):
    result = recognize(
        "what is the temperature in the living room?", intents, slot_lists=slot_lists
    )
    assert result is not None
    assert result.intent.name == "GetTemperature"

    assert result.entities["area"].value == "area.living_room"

    # From YAML
    assert result.entities["domain"].value == "climate"


# pylint: disable=redefined-outer-name
def test_close_name(intents, slot_lists):
    result = recognize("close the garage door", intents, slot_lists=slot_lists)
    assert result is not None
    assert result.intent.name == "CloseCover"

    assert result.entities["name"].value == "cover.garage_door"

    # From YAML
    assert result.entities["domain"].value == "cover"


# pylint: disable=redefined-outer-name
def test_close_not_light(intents, slot_lists):
    result = recognize("close the hue", intents, slot_lists=slot_lists)
    assert result is None


# pylint: disable=redefined-outer-name
def test_play(intents, slot_lists):
    result = recognize("play roku", intents, slot_lists=slot_lists)
    assert result is not None
    assert result.intent.name == "Play"

    assert result.entities["name"].value == "media_player.roku"

    # From context
    assert result.context["domain"] == "media_player"


# pylint: disable=redefined-outer-name
def test_play_no_cover(intents, slot_lists):
    result = recognize("play the garage door", intents, slot_lists=slot_lists)
    assert result is None


# pylint: disable=redefined-outer-name
def test_requires_context_implicit(intents, slot_lists):
    intent_context = {
        "area": "living room",
        "not_copied": "not copied value",
        "copied_to_different": "copied value",
    }

    result = recognize(
        "close the curtains",
        intents,
        slot_lists=slot_lists,
        intent_context=intent_context,
    )
    assert result is not None
    assert result.intent.name == "CloseCurtains"

    # test_slot should not be copied over
    assert set(result.entities.keys()) == {
        "area",
        "domain",
        "device_class",
        "different_slot",
    }
    assert result.entities["area"].value == "living room"
    assert result.entities["domain"].value == "cover"
    assert result.entities["device_class"].value == "curtain"
    assert result.entities["different_slot"].value == "copied value"


# pylint: disable=redefined-outer-name
def test_requires_context_none_provided(intents, slot_lists):
    result = recognize("close the curtains", intents, slot_lists=slot_lists)
    assert result is None


def test_lists_no_template() -> None:
    """Ensure list values without template syntax are plain text."""
    yaml_text = """
    language: "en"
    intents: {}
    lists:
      test:
        values:
          - "test value"
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    test_list = cast(TextSlotList, intents.slot_lists["test"])
    text_in = test_list.values[0].text_in
    assert isinstance(text_in, TextChunk)
    assert text_in.text == "test value"


def test_list_text_normalized() -> None:
    """Ensure list text in values are normalized."""
    yaml_text = """
    language: "en"
    intents:
      TestIntent:
        data:
          - sentences:
              - "run {test_name}"
    lists:
      test_name:
        values:
          - "tEsT    1"
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    result = recognize("run test 1", intents)
    assert result is not None
    assert result.entities["test_name"].value == "tEsT    1"


def test_sentence_text_normalized() -> None:
    """Ensure sentence text is normalized."""
    yaml_text = """
    language: "en"
    intents:
      TestIntent:
        data:
          - sentences:
              - "what's the time"
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    result = recognize("what’s the time", intents)
    assert result is not None


def test_skip_prefix() -> None:
    yaml_text = """
    language: "en"
    intents:
      TestIntent:
        data:
          - sentences:
              - "run {test_name}"
    lists:
      test_name:
        values:
          - "test"
    skip_words:
      - "the"
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    result = recognize("run the test", intents)
    assert result is not None
    assert result.entities["test_name"].value == "test"


def test_skip_sorted() -> None:
    """Ensure skip words are processed longest first"""
    yaml_text = """
    language: "en"
    intents:
      TestIntent:
        data:
          - sentences:
              - "run {test_name}"
    lists:
      test_name:
        values:
          - "test"
    skip_words:
      - "could"
      - "could you"
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    result = recognize("could you run test", intents)
    assert result is not None
    assert result.entities["test_name"].value == "test"


def test_response_key() -> None:
    """Check response key in intent data"""
    yaml_text = """
    language: "en"
    intents:
      TestIntent:
        data:
          - sentences:
              - "this is a test"
            response: "test_response"
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    result = recognize("this is a test", intents)
    assert result is not None
    assert result.response == "test_response"


def test_entity_text() -> None:
    """Ensure original text is returned as well as substituted list value"""
    yaml_text = """
    language: "en"
    intents:
      TestIntent:
        data:
          - sentences:
              - "run test {name} [now]"
              - "{name} test"
    lists:
      name:
        values:
          - in: "alpha "
            out: "A"
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    for sentence in ("run test alpha, now", "run test alpha!", "alpha test"):
        result = recognize(sentence, intents)
        assert result is not None, sentence
        assert result.entities["name"].value == "A"
        assert result.entities["name"].text_clean == "alpha"


def test_number_text() -> None:
    """Ensure original text is returned as well as substituted number"""
    yaml_text = """
    language: "en"
    intents:
      TestIntent:
        data:
          - sentences:
              - "set {percentage}[%] [now]"
              - "{percentage}[%] set"
    lists:
      percentage:
        range:
          from: 0
          to: 100
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    for sentence in ("set 50% now", "set 50%", "50% set"):
        result = recognize(sentence, intents)
        assert result is not None, sentence
        assert result.entities["percentage"].value == 50
        assert result.entities["percentage"].text.strip() == "50"


def test_recognize_all() -> None:
    """Test recognize_all method for returning all matches."""
    yaml_text = """
    language: "en"
    intents:
      TestIntent1:
        data:
          - sentences:
              - "run test"
      TestIntent2:
        data:
          - sentences:
              - "run test"
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    results = list(recognize_all("run test", intents))
    assert len(results) == 2
    assert {result.intent.name for result in results} == {
        "TestIntent1",
        "TestIntent2",
    }


def test_ignore_whitespace() -> None:
    """Test option to ignore whitespace during matching."""
    yaml_text = """
    language: "en"
    settings:
      ignore_whitespace: true
    intents:
      TestIntent1:
        data:
          - sentences:
              - "run [the] test"
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    for sentence in ("runtest", "runthetest", "r u n t h e t e s t"):
        result = recognize(sentence, intents)
        assert result is not None, sentence


def test_skip_words_ignore_whitespace() -> None:
    """Test option to ignore whitespace with skip words during matching."""
    yaml_text = """
    language: "en"
    settings:
      ignore_whitespace: true
    intents:
      TestIntent1:
        data:
          - sentences:
              - "ad"
    skip_words:
      - "bc"
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    result = recognize("abcd", intents)
    assert result is not None


def test_local_expansion_rules() -> None:
    """Test local expansion rules, defined at the intent level"""
    yaml_text = """
    language: "en"
    intents:
      GetSmokeState:
        data:
          - expansion_rules:
              verb: "(are|is)"
              subject: "[all] [the] light[s]"
              state: "on"
              location: "[in <area>]"
            sentences:
              - "<verb> <subject> <state> <location>"
              - "<verb> <subject> <location> <state>"
    expansion_rules:
      area: "[the] {area}"
    lists:
      area:
        values:
          - kitchen
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    for sentence in (
        "are the lights on in the kitchen",
        "are the lights in the kitchen on",
    ):
        result = recognize(sentence, intents)
        assert result is not None, sentence
        assert result.intent.name == "GetSmokeState"


def test_local_slot_lists() -> None:
    """Test local slot lists, defined at the intent level"""
    yaml_text = """
    language: "en"
    intents:
      PlayTrackAtVolume:
        data:
          - sentences:
              - "play {track} at {volume}[%| percent] volume"
            lists:
              track:
                wildcard: true
    lists:
      volume:
        range:
          from: 1
          to: 100
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    for sentence in ("play paint it black at 90% volume",):
        result = recognize(sentence, intents)
        assert result is not None, sentence
        assert result.intent.name == "PlayTrackAtVolume"
        track = result.entities.get("track")
        volume = result.entities.get("volume")
        assert isinstance(track, MatchEntity)
        assert track.value == "paint it black"
        assert isinstance(volume, MatchEntity)
        assert volume.value == 90


def test_unmatched_entity() -> None:
    """Test allow_unmatched_entities option to provide better feedback."""
    yaml_text = """
    language: "en"
    intents:
      Test:
        data:
          - sentences:
              - "set [all] {domain} in {area} to {percent}[%] now"
              - "set {area} {domain} to {percent}"
    lists:
      area:
        values:
          - kitchen
          - bedroom
      domain:
        values:
          - lights
      percent:
        range:
          type: percentage
          from: 0
          to: 100
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    sentence = "set fans in living room to 101% now"

    # Should fail without unmatched entities enabled
    result = recognize(sentence, intents, allow_unmatched_entities=False)
    assert result is None, f"{sentence} should not match"

    # Should succeed now
    result = recognize(sentence, intents, allow_unmatched_entities=True)
    assert result is not None, f"{sentence} should match"
    assert set(result.unmatched_entities.keys()) == {"domain", "area", "percent"}
    domain = result.unmatched_entities["domain"]
    assert isinstance(domain, UnmatchedTextEntity)
    assert domain.text == "fans "

    area = result.unmatched_entities["area"]
    assert isinstance(area, UnmatchedTextEntity)
    assert area.text == "living room "

    percent = result.unmatched_entities["percent"]
    assert isinstance(percent, UnmatchedRangeEntity)
    assert percent.value == 101

    sentence = "set all lights in kitchen to blah blah blah now"
    result = recognize(sentence, intents, allow_unmatched_entities=True)
    assert result is not None, f"{sentence} should match"
    assert set(result.unmatched_entities.keys()) == {"percent"}

    percent = result.unmatched_entities["percent"]
    assert isinstance(percent, UnmatchedTextEntity)
    assert percent.text == "blah blah blah "

    # Test with unmatched entity at end of sentence
    sentence = "set kitchen lights to nothing"
    result = recognize(sentence, intents, allow_unmatched_entities=True)
    assert result is not None, f"{sentence} should match"
    assert set(result.unmatched_entities.keys()) == {"percent"}

    percent = result.unmatched_entities["percent"]
    assert isinstance(percent, UnmatchedTextEntity)
    assert percent.text == "nothing"


def test_unmatched_range_only() -> None:
    """Test allow_unmatched_entities option with an out-of-range value only."""
    yaml_text = """
    language: "en"
    intents:
      Test:
        data:
          - sentences:
              - "set {domain} to {percent}[%]"
    lists:
      domain:
        values:
          - lights
      percent:
        range:
          type: percentage
          from: 0
          to: 100
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    sentence = "set lights to 1001%"

    # Should fail without unmatched entities enabled
    result = recognize(sentence, intents, allow_unmatched_entities=False)
    assert result is None, f"{sentence} should not match"

    # Should succeed now
    result = recognize(sentence, intents, allow_unmatched_entities=True)
    assert result is not None, f"{sentence} should match"
    assert set(result.entities.keys()) == {"domain"}
    assert set(result.unmatched_entities.keys()) == {"percent"}

    domain = result.entities["domain"]
    assert domain.text == "lights"

    percent = result.unmatched_entities["percent"]
    assert isinstance(percent, UnmatchedRangeEntity)
    assert percent.value == 1001


def test_no_empty_unmatched_entity() -> None:
    """Test that unmatched entities are not empty."""
    yaml_text = """
    language: "en"
    intents:
      Test:
        data:
          - sentences:
              - "turn on {name}[ please]"
              - "illuminate all[ {area}] lights"
              - "activate {name} now"
    lists:
      name:
        values:
          - light
      area:
        values:
          - bedroom
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    sentence = "turn on "
    results = list(recognize_all(sentence, intents, allow_unmatched_entities=True))
    assert not results, f"{sentence} should not match"

    # With optional word at end
    sentence = "turn on please"
    results = list(recognize_all(sentence, intents, allow_unmatched_entities=True))
    assert not results, f"{sentence} should not match"

    sentence = "illuminate all lights"
    results = list(recognize_all(sentence, intents, allow_unmatched_entities=True))
    assert results, f"{sentence} should match"
    assert len(results) == 1, "Only 1 result expected"
    assert not results[0].unmatched_entities, "No unmatched entities expected"

    sentence = "illuminate all kitchen lights"
    results = list(recognize_all(sentence, intents, allow_unmatched_entities=True))
    assert results, f"{sentence} should match"
    assert len(results) == 1, "Only 1 result expected"
    result = results[0]
    assert set(result.unmatched_entities.keys()) == {"area"}
    area = result.unmatched_entities["area"]
    assert isinstance(area, UnmatchedTextEntity)
    assert area.text == "kitchen "

    # With required word at end
    sentence = "activate now"
    results = list(recognize_all(sentence, intents, allow_unmatched_entities=True))
    assert not results, f"{sentence} should not match"


def test_unmatched_entity_context() -> None:
    """Test that unmatched entities work with requires/excludes context."""
    yaml_text = """
    language: "en"
    intents:
      Test:
        data:
          - sentences:
              - "open {name}"
            requires_context:
              domain: cover
    lists:
      name:
        values:
          - garage door
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    sentence = "open garage door"

    # Should fail when unmatched entities aren't allowed
    result = recognize(sentence, intents, allow_unmatched_entities=False)
    assert result is None, f"{sentence} should not match"

    # Should succeed now with an unmatched domain entity
    result = recognize(sentence, intents, allow_unmatched_entities=True)
    assert result is not None, f"{sentence} should match"
    assert set(result.unmatched_entities.keys()) == {"domain"}
    domain = result.unmatched_entities["domain"]
    assert isinstance(domain, UnmatchedTextEntity)
    assert domain.text == MISSING_ENTITY

    # Now both entities are unmatched
    sentence = "open back door"
    result = recognize(sentence, intents, allow_unmatched_entities=True)
    assert result is not None, f"{sentence} should match"
    assert set(result.unmatched_entities.keys()) == {"domain", "name"}

    domain = result.unmatched_entities["domain"]
    assert isinstance(domain, UnmatchedTextEntity)
    assert domain.text == MISSING_ENTITY

    name = result.unmatched_entities["name"]
    assert isinstance(name, UnmatchedTextEntity)
    assert name.text == "back door"


def test_unmatched_slot_name() -> None:
    """Test that unmatched entities use slot name instead of list name."""
    yaml_text = """
    language: "en"
    intents:
      Test:
        data:
          - sentences:
              - "run {script_name:name}"
              - "execute script {script_number:number}"
    lists:
      script_name:
        values:
          - stealth mode
      script_number:
        range:
          from: 1
          to: 100
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    sentence = "run missing name"
    result = recognize(sentence, intents, allow_unmatched_entities=True)
    assert result is not None, f"{sentence} should match"
    assert set(result.unmatched_entities.keys()) == {"name"}

    sentence = "execute script wrong number"
    result = recognize(sentence, intents, allow_unmatched_entities=True)
    assert result is not None, f"{sentence} should match"
    assert set(result.unmatched_entities.keys()) == {"number"}

    # Outside range
    sentence = "execute script 0"
    result = recognize(sentence, intents, allow_unmatched_entities=True)
    assert result is not None, f"{sentence} should match"
    assert set(result.unmatched_entities.keys()) == {"number"}


def test_unmatched_entity_stops_at_optional() -> None:
    """Test that unmatched entities do not cross optional text chunks."""
    yaml_text = """
    language: "en"
    intents:
      Test:
        data:
          - sentences:
              - "set {area} [to] brightness <brightness>"
              - "set {name} [to] brightness <brightness>"
    lists:
      name:
        values:
          - lamp
      area:
        values:
          - kitchen
      brightness_pct:
          range:
            type: percentage
            from: 0
            to: 100
    expansion_rules:
        brightness: "{brightness_pct}[%| percent]"
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    sentence = "set unknown thing to brightness 100%"

    # Should fail without unmatched entities enabled
    result = recognize(sentence, intents, allow_unmatched_entities=False)
    assert result is None, f"{sentence} should not match"

    results = list(recognize_all(sentence, intents, allow_unmatched_entities=True))
    assert len(results) == 4

    area_names: Set[str] = set()
    entity_names: Set[str] = set()

    for result in results:
        assert len(result.unmatched_entities) == 1
        area_entity = result.unmatched_entities.get("area")
        if area_entity is not None:
            assert isinstance(area_entity, UnmatchedTextEntity)
            area_names.add(area_entity.text)
        else:
            name_entity = result.unmatched_entities.get("name")
            assert name_entity is not None
            assert isinstance(name_entity, UnmatchedTextEntity)
            entity_names.add(name_entity.text)

    assert area_names == {"unknown thing ", "unknown thing to "}
    assert entity_names == {"unknown thing ", "unknown thing to "}


def test_unmatched_entities_dont_share_text() -> None:
    """Test that text only goes into one unmatched entity."""
    yaml_text = """
    language: "en"
    intents:
      Test:
        data:
          - sentences:
              - "set [the] brightness [of] {name} [to] <brightness>"
    lists:
      name:
        values:
          - lamp
      brightness_pct:
          range:
            type: percentage
            from: 0
            to: 100
    expansion_rules:
        brightness: "{brightness_pct}[%| percent]"
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    sentence = "set brightness of unknown thing to 100%"

    # Should fail without unmatched entities enabled
    result = recognize(sentence, intents, allow_unmatched_entities=False)
    assert result is None, f"{sentence} should not match"

    results = list(recognize_all(sentence, intents, allow_unmatched_entities=True))
    assert len(results) == 2

    possible_names: Set[str] = set()
    for result in results:
        assert len(result.unmatched_entities) == 1
        assert "name" in result.unmatched_entities
        name_entity = result.unmatched_entities["name"]
        assert isinstance(name_entity, UnmatchedTextEntity)
        possible_names.add(name_entity.text)

    assert possible_names == {"unknown thing ", "of unknown thing "}


def test_unmatched_entities_cant_skip_words() -> None:
    yaml_text = """
    language: "en"
    intents:
      Test:
        data:
          - sentences:
              - "[turn] {name} [to] on"
    lists:
      name:
        values:
          - lamp
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    sentence = "turn on unknown thing"

    # Should fail without unmatched entities enabled
    result = recognize(sentence, intents, allow_unmatched_entities=False)
    assert result is None, f"{sentence} should not match"

    # Should also fail with unmatched entities enabled
    results = list(recognize_all(sentence, intents, allow_unmatched_entities=True))
    assert len(results) == 0


def test_unmatched_entities_text_chunks_matched() -> None:
    yaml_text = """
    language: "en"
    intents:
      Test:
        data:
          - sentences:
              - "[turn] on {name}"
              - "[turn] on {name} light"
    lists:
      name:
        values:
          - test
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    sentence = "turn on unknown light"

    results = list(recognize_all(sentence, intents, allow_unmatched_entities=True))
    assert len(results) == 2

    # '[turn] on {name} light' should have more literal text matched
    result_1, result_2 = results
    assert result_1.intent_sentence is not None
    if result_1.intent_sentence.text == "[turn] on {name}":
        assert result_1.text_chunks_matched < result_2.text_chunks_matched
    else:
        assert result_2.text_chunks_matched < result_1.text_chunks_matched


def test_wildcard() -> None:
    """Test wildcard slot lists/entities."""
    yaml_text = """
    language: "en"
    intents:
      Test:
        data:
          - sentences:
              - "play {album} by {artist}[ please] now"
              - "start {album} by {artist}"
              - "begin {album} by artist {artist}"
    lists:
      album:
        wildcard: true
      artist:
        wildcard: true
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    # Case should be kept
    sentence = "play The White Album by The Beatles please now"
    result = recognize(sentence, intents)
    assert result is not None, f"{sentence} should match"
    assert set(result.entities.keys()) == {"album", "artist"}
    assert result.entities["album"].value == "The White Album"
    assert result.entities["album"].is_wildcard
    assert result.entities["artist"].value == "The Beatles"
    assert result.entities["artist"].is_wildcard

    # Wildcards cannot be empty
    sentence = "play by please now"
    result = recognize(sentence, intents)
    assert result is None, f"{sentence} should not match"

    # Test without text at the end
    sentence = "start the white album by the beatles."
    result = recognize(sentence, intents)
    assert result is not None, f"{sentence} should match"
    assert set(result.entities.keys()) == {"album", "artist"}
    assert result.entities["album"].value == "the white album"
    assert result.entities["album"].is_wildcard
    assert result.entities["artist"].value == "the beatles"
    assert result.entities["artist"].is_wildcard

    # Test use of next word in wildcard
    sentence = "play day by day by taken by trees now"
    results = list(recognize_all(sentence, intents))
    assert results, f"{sentence} should match"
    assert len(results) == 3  # 3 "by" words

    # Verify each combination of album/artist is present
    album_artist = {
        (result.entities["album"].value, result.entities["artist"].value)
        for result in results
    }
    assert album_artist == {
        ("day", "day by taken by trees"),
        ("day by day", "taken by trees"),
        ("day by day by taken", "trees"),
    }
    for result in results:
        assert result.entities["album"].is_wildcard
        assert result.entities["artist"].is_wildcard

    # Test use of next word at end of word in wildcard
    sentence = "play Moby by Moby please now"
    result = recognize(sentence, intents)
    assert result is not None, f"{sentence} should match"
    assert set(result.entities.keys()) == {"album", "artist"}
    assert result.entities["album"].value == "Moby"
    assert result.entities["album"].is_wildcard
    assert result.entities["artist"].value == "Moby"
    assert result.entities["artist"].is_wildcard

    # Add "artist" word
    sentence = "begin day by day by artist taken by trees"
    result = recognize(sentence, intents)
    assert result is not None, f"{sentence} should match"
    assert set(result.entities.keys()) == {"album", "artist"}
    assert result.entities["album"].value == "day by day"
    assert result.entities["album"].is_wildcard
    assert result.entities["artist"].value == "taken by trees"
    assert result.entities["artist"].is_wildcard


def test_wildcard_degenerate() -> None:
    """Test degenerate case for wildcards."""
    yaml_text = """
    language: "en"
    intents:
      Test:
        data:
          - sentences:
              - "play {album} by {artist}"
    lists:
      album:
        wildcard: true
      artist:
        wildcard: true
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    sentence = "play by by by by by"
    results = list(recognize_all(sentence, intents))
    assert results, f"{sentence} should match"
    assert len(results) == 3  # 3 valid splits

    # Verify each combination
    album_artist = {
        (result.entities["album"].value, result.entities["artist"].value)
        for result in results
    }
    assert album_artist == {
        ("by", "by by by"),
        ("by by", "by by"),
        ("by by by", "by"),
    }


def test_optional_wildcard() -> None:
    """Test optional wildcard slot list."""
    yaml_text = """
    language: "en"
    intents:
      Test:
        data:
          - sentences:
              - "play {album}[by {artist}]"
    lists:
      album:
        wildcard: true
      artist:
        wildcard: true
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    # With all wildcards
    sentence = "play the white album by the beatles"
    result = recognize(sentence, intents)
    assert result is not None, f"{sentence} should match"
    assert set(result.entities.keys()) == {"album", "artist"}
    assert result.entities["album"].value == "the white album"
    assert result.entities["artist"].value == "the beatles"

    # Missing one wildcard
    sentence = "play the white album"
    result = recognize(sentence, intents)
    assert result is not None, f"{sentence} should match"
    assert set(result.entities.keys()) == {"album"}
    assert result.entities["album"].value == "the white album"


def test_wildcard_slot_name() -> None:
    """Test wildcard uses slot instead of list name."""
    yaml_text = """
    language: "en"
    intents:
      Test:
        data:
          - sentences:
              - "run {script_name:name}"
    lists:
      script_name:
        wildcard: true
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    sentence = "run script 1"
    result = recognize(sentence, intents)
    assert result is not None, f"{sentence} should match"
    assert set(result.entities.keys()) == {"name"}
    assert result.entities["name"].value == "script 1"


def test_wildcard_ordering() -> None:
    """Test wildcard ordering by number of literal text chunks."""
    yaml_text = """
    language: "en"
    intents:
      Test:
        data:
          - sentences:
              - "play {album} by {artist}"
              - "play {album} by {artist} in {room}"
    lists:
      album:
        wildcard: true
      artist:
        wildcard: true
      room:
        wildcard: true
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    sentence = "play the white album by the beatles in the living room"
    result = recognize(sentence, intents)
    assert result is not None, f"{sentence} should match"
    assert set(result.entities.keys()) == {"album", "artist", "room"}
    assert result.entities["album"].value == "the white album"
    assert result.entities["artist"].value == "the beatles"
    assert result.entities["room"].value == "the living room"

    # Check that the first sentence can still be used
    sentence = "play the white album by the beatles"
    result = recognize(sentence, intents)
    assert result is not None, f"{sentence} should match"
    assert set(result.entities.keys()) == {"album", "artist"}
    assert result.entities["album"].value == "the white album"
    assert result.entities["artist"].value == "the beatles"


def test_ordering_only_wildcards() -> None:
    """Test that re-ordering only affects wildcards."""
    yaml_text = """
    language: "en"
    intents:
      Test:
        data:
          - sentences:
              - "turn on {light} in {room}"
              - "turn on {light}"
    lists:
      light:
        values:
          - light
          - light in bedroom
      room:
        values:
          - bedroom
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    sentence = "turn on light in bedroom"
    result = recognize(sentence, intents)
    assert result is not None, f"{sentence} should match"
    assert set(result.entities.keys()) == {"light", "room"}
    assert result.entities["light"].value == "light"
    assert result.entities["room"].value == "bedroom"


def test_wildcard_punctuation() -> None:
    """Test that wildcards do not include punctuation."""
    yaml_text = """
    language: "en"
    intents:
      Test:
        data:
          - sentences:
              - "is {name} in {zone}"
    lists:
      name:
        wildcard: true
      zone:
        wildcard: true
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    sentence = "is Alice in New York!?"
    result = recognize(sentence, intents)
    assert result is not None, f"{sentence} should match"
    assert set(result.entities.keys()) == {"name", "zone"}
    assert result.entities["name"].value == "Alice"
    assert result.entities["zone"].value == "New York"


def test_wildcard_inside_word() -> None:
    """Test wildcard inside of a word."""
    yaml_text = """
    language: "en"
    intents:
      Test:
        data:
          - sentences:
              - "ab{test}cd"
    lists:
      test:
        wildcard: true
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    sentence = "ab123cd"
    result = recognize(sentence, intents)
    assert result is not None, f"{sentence} should match"
    assert set(result.entities.keys()) == {"test"}
    assert result.entities["test"].value == "123"
    assert result.entities["test"].is_wildcard


def test_wildcard_outside_word() -> None:
    """Test wildcard outside of a word."""
    yaml_text = """
    language: "en"
    intents:
      Test:
        data:
          - sentences:
              - "ab{test} cd"
    lists:
      test:
        wildcard: true
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    sentence = "ab123 cd"
    result = recognize(sentence, intents)
    assert result is not None, f"{sentence} should match"
    assert set(result.entities.keys()) == {"test"}
    assert result.entities["test"].value == "123"
    assert result.entities["test"].is_wildcard

    sentence = "ab123cd"
    result = recognize(sentence, intents)
    assert result is None, f"{sentence} should not match"


def test_wildcard_outside_word_ignore_whitespace() -> None:
    """Test wildcard outside of a word when ignoring whitespace."""
    yaml_text = """
    language: "en"
    settings:
      ignore_whitespace: true
    intents:
      Test:
        data:
          - sentences:
              - "ab{test} cd"
    lists:
      test:
        wildcard: true
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    for sentence in ("abc123 cd", "abc123cd"):
        result = recognize(sentence, intents)
        assert result is not None, f"{sentence} should match"


def test_entity_metadata() -> None:
    """Ensure metadata is returned for text slots"""
    yaml_text = """
    language: "en"
    intents:
      TestIntent:
        data:
          - sentences:
              - "run test {name} [now]"
              - "{name} test"
    lists:
      name:
        values:
          - in: "alpha "
            out: "A"
            metadata:
              is_alpha: true
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    for sentence in ("run test alpha, now", "run test alpha!", "alpha test"):
        result = recognize(sentence, intents)
        assert result is not None, sentence
        assert result.entities["name"].value == "A"
        assert result.entities["name"].text_clean == "alpha"
        assert result.entities["name"].metadata == {"is_alpha": True}


def test_sentence_metadata() -> None:
    """Test that metadata attached to sentences is passed through to the result."""
    yaml_text = """
    language: "en"
    intents:
      Test:
        data:
          - sentences:
              - "this is a test"
            metadata:
              string_key: "test value"
              int_key: 1234
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    sentence = "this is a test"
    result = recognize(sentence, intents)
    assert result is not None, f"{sentence} should match"
    assert result.intent_metadata is not None, "No metadata"
    assert result.intent_metadata == {"string_key": "test value", "int_key": 1234}


def test_digits_calc() -> None:
    """Test that metadata attached to sentences is passed through to the result."""
    yaml_text = """
    language: "en"
    intents:
      Calculate:
        data:
          - sentences:
              - "calc[ulate] {x} {operator} {y}"
    lists:
      operator:
        values:
          - in: "(+|plus)"
            out: "+"
      x:
        range:
          from: 0
          to: 100
          digits: true
          words: true
      y:
        range:
          from: 0
          to: 100
          digits: true
          words: true
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    sentence = "calc 1 + 2"
    result = recognize(sentence, intents)
    assert result is not None, f"{sentence} should match"
    assert result.entities.keys() == {"x", "operator", "y"}
    assert result.entities["x"].value == 1
    assert result.entities["operator"].value == "+"
    assert result.entities["y"].value == 2

    sentence = "calc 1 plus two"
    result = recognize(sentence, intents)
    assert result is not None, f"{sentence} should match"
    assert result.entities.keys() == {"x", "operator", "y"}
    assert result.entities["x"].value == 1
    assert result.entities["operator"].value == "+"
    assert result.entities["y"].value == 2


def test_range_params_calc() -> None:
    """Test that params attached to RangeSlotList affect the parsing."""
    yaml_text = """
    language: "en"
    intents:
      Calculate:
        data:
          - sentences:
              - "calc[ulate] {x} {operator} {y}"
    lists:
      operator:
        values:
          - in: "(+|plus)"
            out: "+"
      x:
        range:
          from: 0
          to: 100
          digits: false
          words: true
      y:
        range:
          from: 0
          to: 100
          digits: true
          words: false
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    # x can't have digits
    sentence = "calc 1 + 2"
    result = recognize(sentence, intents)
    assert result is None, f"{sentence} should not match"

    # y can't have words
    sentence = "calc one plus two"
    result = recognize(sentence, intents)
    assert result is None, f"{sentence} should not match"

    sentence = "calc one + 2"
    result = recognize(sentence, intents)
    assert result is not None, f"{sentence} should match"
    assert result.entities.keys() == {"x", "operator", "y"}
    assert result.entities["x"].value == 1
    assert result.entities["operator"].value == "+"
    assert result.entities["y"].value == 2


def test_range_rule_sets_calc() -> None:
    """Test that params attached to RangeSlotList affect the parsing."""
    # https://github.com/rhasspy/unicode-rbnf/blob/master/unicode_rbnf/engine.py#L13
    yaml_text = """
    language: "en"
    intents:
      Calculate:
        data:
          - sentences:
              - "calc[ulate] {x} {operator} {y}"
    lists:
      operator:
        values:
          - in: "(+|plus)"
            out: "+"
      x:
        range:
          from: 0
          to: 3000
          digits: true
          words: true
      y:
        range:
          from: 0
          to: 3000
          digits: true
          words: true
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    sentence = "calc one thousand nine hundred ninety-nine + 23"
    result = recognize(sentence, intents)
    assert result is not None, f"{sentence} should match"
    assert result.entities.keys() == {"x", "operator", "y"}
    assert result.entities["x"].value == 1999
    assert result.entities["operator"].value == "+"
    assert result.entities["y"].value == 23

    sentence = "calc 23 + nineteen ninety-nine"
    result = recognize(sentence, intents)
    assert result is None, f"{sentence} should not match"


# pylint: disable=redefined-outer-name
def test_context_dict(intents, slot_lists):
    yaml_text = """
    language: "en"
    intents:
      TestIntent:
        data:
          - sentences:
              - "test sentence"
            requires_context:
              slot1:
                value: value1
                slot: true
            excludes_context:
              slot2: value2
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    slot1 = {"value": "value1", "text": "Value One"}

    # Try includes context
    result = recognize(
        "test sentence",
        intents,
        slot_lists=slot_lists,
        intent_context={"slot1": slot1},
    )
    assert result is not None
    assert result.context.keys() == {"slot1"}
    assert result.context["slot1"] == slot1
    assert result.entities.keys() == {"slot1"}
    assert result.entities["slot1"].value == "value1"
    assert result.entities["slot1"].text == "Value One"

    # Try excludes context
    result = recognize(
        "test sentence",
        intents,
        slot_lists=slot_lists,
        intent_context={"slot1": slot1, "slot2": {"value": "value2"}},
    )
    assert result is None


# pylint: disable=redefined-outer-name
def test_range_multiplier(intents, slot_lists):
    yaml_text = """
    language: "en"
    intents:
      SetVolume:
        data:
          - sentences:
              - "set volume to {volume_level}"
    lists:
      volume_level:
        range:
          from: 0
          to: 100
          multiplier: 0.01
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    result = recognize("set volume to 50", intents)
    assert result is not None
    assert result.entities.keys() == {"volume_level"}
    assert result.entities["volume_level"].value == 0.5
    assert result.entities["volume_level"].text == "50"


def test_recognize_best():
    yaml_text = """
    language: "en"
    intents:
      TurnOn:
        data:
          - sentences:
              - "{anything} lamp"
            metadata:
              best_key: "best value"
          - sentences:
              - "turn on {area} lamp"
              - "turn on {name}"
    lists:
      area:
        values:
          - bedroom
      name:
        values:
          - bedroom lamp
      anything:
        wildcard: true
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    # Should match the sentence with the wildcard slot because it's listed first.
    result = recognize("turn on bedroom lamp", intents)  # not best
    assert result is not None
    assert result.entities.keys() == {"anything"}

    # Should match the sentence with the wildcard slot because of its metadata.
    result = recognize_best(
        "turn on bedroom lamp", intents, best_metadata_key="best_key"
    )
    assert result is not None
    assert result.entities.keys() == {"anything"}

    # Should match the sentence with the "area" slot because it has the most
    # literal text matched.
    result = recognize_best("turn on bedroom lamp", intents)
    assert result is not None
    assert result.entities.keys() == {"area"}

    # Should match the sentence with the "name" slot because it's a priority
    result = recognize_best("turn on bedroom lamp", intents, best_slot_name="name")
    assert result is not None
    assert result.entities.keys() == {"name"}
    assert result.entities["name"].value == "bedroom lamp"


def test_regex_branching():
    yaml_text = """
    language: "en"
    intents:
      TurnOn:
        data:
          - sentences:
              - "turn on ({area} {name}|{name})"
    lists:
      area:
        values:
          - bedroom
      name:
        values:
          - bedroom lamp
          - lamp
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    results = list(recognize_all("turn on bedroom lamp", intents))
    assert len(results) == 2


def test_commas_dont_change() -> None:
    """Ensure commas don't change the interpretation of a sentence."""
    yaml_text = """
    language: "en"
    intents:
      TurnOn:
        data:
          - sentences:
              - "turn on [the] {name}"
              - "turn on [the] {area} lights"
    lists:
      name:
        values:
          - lamp
      area:
        values:
          - living room
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    for sentence in (
        "turn on the living room lights",
        "turn, on the, living room lights",
        "turn on, the living room lights",
        "turn on the, living room lights",
        "turn on the living room, lights",
    ):
        result = recognize(sentence, intents)
        assert result is not None, sentence
        assert result.entities.keys() == {"area"}


def test_wildcard_then_other_stuff() -> None:
    """Test wildcard followed by expansion rule and list."""
    yaml_text = """
    language: "en"
    intents:
      SetTimer:
        data:
          - sentences:
              - "set timer {timer_name:name} <timer_duration>"
              - "set timer {timer_name:name} {timer_state:state} [now]"
      AddItem:
        data:
          - sentences:
              - "add {item} [to [my]] {todo_list}"
    lists:
      timer_name:
        wildcard: true
      item:
        wildcard: true
      minutes:
        range:
          from: 1
          to: 59
      timer_state:
        values:
          - "on"
          - "off"
      todo_list:
        values:
          - "shopping list"
    expansion_rules:
      timer_duration: "{minutes} minute[s]"
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    # Check ranges
    for sentence in ("set timer pizza 5 minutes", "set timer pizza five minutes"):
        result = recognize(sentence, intents)
        assert result is not None, f"{sentence} should match"
        assert set(result.entities.keys()) == {"name", "minutes"}
        assert result.entities["name"].text == "pizza"
        assert result.entities["name"].value == "pizza"
        assert result.entities["minutes"].text.strip() in {"5", "five"}
        assert result.entities["minutes"].value == 5

    # Check value list
    sentence = "set timer a big long name on now"
    result = recognize(sentence, intents)
    assert result is not None, f"{sentence} should match"
    assert set(result.entities.keys()) == {"name", "state"}
    assert result.entities["name"].text == "a big long name"
    assert result.entities["name"].value == "a big long name"
    assert result.entities["state"].text == "on"
    assert result.entities["state"].value == "on"

    sentence = "add apples to my shopping list"
    result = recognize(sentence, intents)
    assert result is not None, f"{sentence} should match"
    assert set(result.entities.keys()) == {"item", "todo_list"}
    assert result.entities["item"].text == "apples"
    assert result.entities["item"].value == "apples"
    assert result.entities["todo_list"].text == "shopping list"
    assert result.entities["todo_list"].value == "shopping list"


def test_range_list_with_one_number() -> None:
    """Test a range list with the same start/stop value."""
    yaml_text = """
    language: "en"
    intents:
      TestIntent:
        data:
          - sentences:
              - "test {value}"
    lists:
      value:
        range:
          from: 1
          to: 1
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    # Check ranges
    assert recognize("test 1", intents)
    assert not recognize("test 2", intents)


def test_range_list_with_halves() -> None:
    """Test a range list with fractions (1/2)."""
    yaml_text = """
    language: "en"
    intents:
      TestIntent:
        data:
          - sentences:
              - "test {value}"
    lists:
      value:
        range:
          from: 1
          to: 2
          fractions: halves
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    assert recognize("test 1", intents)
    result = recognize("test 1.5", intents)
    assert result is not None
    value = result.entities.get("value")
    assert value is not None
    assert value.value == 1.5
    assert value.text == "1.5"
    assert value.text_clean == "1.5"

    result = recognize("test one point five", intents, language="en")
    assert result is not None
    value = result.entities.get("value")
    assert value is not None
    assert value.value == 1.5
    assert value.text == "one point five"
    assert value.text_clean == "one point five"

    result = recognize("test 2.0", intents)
    assert result is not None
    value = result.entities.get("value")
    assert value is not None
    assert value.value == 2
    assert value.text == "2.0"
    assert value.text_clean == "2.0"

    result = recognize("test 2.5", intents)
    assert result is not None
    value = result.entities.get("value")
    assert value is not None
    assert value.value == 2.5
    assert value.text == "2.5"
    assert value.text_clean == "2.5"

    # Only halves
    assert not recognize("test 2.1", intents)

    # Comma separator
    result = recognize("test 2,5", intents)
    assert result is not None
    value = result.entities.get("value")
    assert value is not None
    assert value.value == 2.5
    assert value.text == "2,5"
    assert value.text_clean == "2,5"


def test_range_list_with_tenths() -> None:
    """Test a range list with fractions (1/10)."""
    yaml_text = """
    language: "en"
    intents:
      TestIntent:
        data:
          - sentences:
              - "test {value}"
    lists:
      value:
        range:
          from: 1
          to: 2
          fractions: tenths
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    assert recognize("test 1", intents)
    assert recognize("test 2", intents)

    for integer in range(1, 3):
        for tenth in range(1, 10):
            value_str = f"{integer}.{tenth}"
            result = recognize(f"test {value_str}", intents)
            assert result is not None
            value = result.entities.get("value")
            assert value is not None
            assert value.value == float(value_str)
            assert value.text == value_str
            assert value.text_clean == value_str

    result = recognize("test one point one", intents, language="en")
    assert result is not None
    value = result.entities.get("value")
    assert value is not None
    assert value.value == 1.1
    assert value.text == "one point one"

    result = recognize("test two point six", intents, language="en")
    assert result is not None
    value = result.entities.get("value")
    assert value is not None
    assert value.value == 2.6
    assert value.text == "two point six"

    # Only tenths
    assert not recognize("test 2.12", intents)


def test_range_lists_separated_by_punctuation() -> None:
    """Test range lists separated by punctuation."""
    yaml_text = """
    language: "en"
    intents:
      TestIntent:
        data:
          - sentences:
              - "test {value1}.{value2}"
    lists:
      value1:
        range:
          from: 0
          to: 1
      value2:
        range:
          from: 0
          to: 2
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    result = recognize("test 1.2", intents)
    assert result is not None

    value1 = result.entities.get("value1")
    assert value1 is not None
    assert value1.value == 1

    value2 = result.entities.get("value2")
    assert value2 is not None
    assert value2.value == 2


def test_range_lists_separated_by_punctuation_with_wildcard() -> None:
    """Test range lists separated by punctuation preceeded by a wildcard."""
    yaml_text = """
    language: "en"
    intents:
      TestIntent:
        data:
          - sentences:
              - "test {anything} {value1}.{value2}"
    lists:
      anything:
        wildcard: true
      value1:
        range:
          from: 0
          to: 1
      value2:
        range:
          from: 0
          to: 2
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    result = recognize("test for the big 1.2", intents)
    assert result is not None

    anything = result.entities.get("anything")
    assert anything is not None
    assert anything.value == "for the big"

    value1 = result.entities.get("value1")
    assert value1 is not None
    assert value1.value == 1

    value2 = result.entities.get("value2")
    assert value2 is not None
    assert value2.value == 2


def test_list_value_in_no_out() -> None:
    """Test list values with "in" but no "out"."""
    yaml_text = """
    language: "en"
    intents:
      TestIntent:
        data:
          - sentences:
              - "test {value1}"
          - sentences:
              - "also test {value2}"
            lists:
              value2:
                values:
                  - g[h]i
                  - j[k]l
    lists:
      value1:
        values:
          - a[b]c
          - in: d[e]f
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    for value in ("abc", "ac", "def", "df"):
        result = recognize(f"test {value}", intents)
        assert result is not None, value
        assert "value1" in result.entities
        assert result.entities["value1"].value == value

    for value in ("ghi", "gi", "jkl", "jl"):
        result = recognize(f"also test {value}", intents)
        assert result is not None, value
        assert "value2" in result.entities
        assert result.entities["value2"].value == value


def test_captures() -> None:
    """Test captures for response only."""
    yaml_text = """
    language: "en"
    intents:
      TestIntent:
        data:
          - sentences:
              - "open windows {preposition:@preposition} the {area}"
              - "close windows {@preposition} the {area}"
    lists:
      preposition:
        values:
          - "in"
          - "on"
      area:
        values:
          - kitchen
          - deck
    """

    with io.StringIO(yaml_text) as test_file:
        intents = Intents.from_yaml(test_file)

    result = recognize("open windows in the kitchen", intents)
    assert result is not None

    # area is an entity
    assert result.entities.keys() == {"area"}

    # preposition is a capture
    assert result.captures.keys() == {"preposition"}
    prep = result.captures["preposition"]
    assert prep.name == "preposition"
    assert prep.text == "in"

    assert len(result.captures_list) == 1
    assert result.captures_list[0] == prep

    # Test compact syntax
    result = recognize("close windows on the deck", intents)
    assert result is not None

    assert result.entities.keys() == {"area"}
    assert result.captures.keys() == {"preposition"}
    assert result.captures["preposition"].text == "on"
