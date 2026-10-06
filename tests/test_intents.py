from hassil import is_match, parse_sentence
from hassil.intents import TextSlotList


def test_no_match():
    sentence = parse_sentence("turn on the lights")
    assert is_match("turn on the lights", sentence)
    assert not is_match("turn off the lights", sentence)
    assert not is_match("don't turn on the lights", sentence)


def test_punctuation():
    sentence = parse_sentence("turn on the lights")
    assert is_match("turn on the lights.", sentence)
    assert is_match("turn on the lights!", sentence)


def test_dash():
    """Test that a dash appended by speech-to-text does not break matching."""
    sentence = parse_sentence("we are going to bed")
    assert is_match("we are going to bed—", sentence)
    assert is_match("we are going to bed –", sentence)


def test_whitespace():
    sentence = parse_sentence("turn on the lights")
    assert is_match("  turn      on the     lights", sentence)


def test_skip_punctuation():
    sentence = parse_sentence("turn on the lights")
    assert is_match("turn ! on ? the, lights.", sentence)


def test_skip_words():
    sentence = parse_sentence("turn on [the] lights")
    skip_words = {"please", "could", "you", "my"}
    assert is_match(
        "could you please turn on my lights?", sentence, skip_words=skip_words
    )
    assert is_match("turn on the lights, please", sentence, skip_words=skip_words)


def test_optional():
    sentence = parse_sentence("turn on [the] lights in [the] kitchen")
    assert is_match("turn on the lights in the kitchen", sentence)
    assert is_match("turn on lights in kitchen", sentence)


def test_optional_plural():
    sentence = parse_sentence("turn on the light[s]")
    assert is_match("turn on the light", sentence)
    assert is_match("turn on the lights", sentence)


def test_group_plural():
    sentence = parse_sentence("give me the penn(y|ies)")
    assert is_match("give me the penny", sentence)
    assert is_match("give me the pennies", sentence)


def test_list():
    sentence = parse_sentence("turn off {area}")
    areas = TextSlotList.from_strings(["kitchen", "living room"])
    assert is_match("turn off kitchen", sentence, slot_lists={"area": areas})
    assert is_match("turn off living room", sentence, slot_lists={"area": areas})


def test_list_prefix_suffix():
    sentence = parse_sentence("turn off abc-{area}-123")
    areas = TextSlotList.from_strings(["kitchen", "living room"])
    assert is_match("turn off abc-kitchen-123", sentence, slot_lists={"area": areas})
    assert is_match(
        "turn off abc-living room-123", sentence, slot_lists={"area": areas}
    )


def test_rule():
    sentence = parse_sentence("turn off <area>")
    assert is_match(
        "turn off kitchen",
        sentence,
        expansion_rules={"area": parse_sentence("[the] kitchen")},
    )


def test_rule_prefix_suffix():
    sentence = parse_sentence("turn off abc-<area>-123")
    assert is_match(
        "turn off abc-kitchen-123",
        sentence,
        expansion_rules={"area": parse_sentence("[the ]kitchen")},
    )


def test_alternative_whitespace():
    sentence = parse_sentence("(start|stopp)ed")
    assert is_match("started", sentence)
    assert is_match("stopped", sentence)


def test_alternative_whitespace_2():
    sentence = parse_sentence("set brightness to ( minimum | lowest)")
    assert is_match("set brightness to lowest", sentence)


def test_no_allow_template():
    sentence = parse_sentence("turn off {name}")
    names = TextSlotList.from_strings(["light[s]"])
    assert is_match("turn off lights", sentence, slot_lists={"name": names})

    names = TextSlotList.from_strings(["light[s]"], allow_template=False)
    assert not is_match("turn off lights", sentence, slot_lists={"name": names})
    assert is_match("turn off light[s]", sentence, slot_lists={"name": names})


def test_no_whitespace_fails():
    sentence = parse_sentence("this is a test")
    assert not is_match("thisisatest", sentence)


def test_permutations():
    sentence = parse_sentence("(in the kitchen;is there smoke)")
    assert is_match("in the kitchen is there smoke", sentence)
    assert is_match("is there smoke in the kitchen", sentence)

    sentence = parse_sentence("(a;b;c)")
    assert is_match("a b c", sentence)
    assert is_match("a c b", sentence)
    assert is_match("b a c", sentence)
    assert is_match("b c a", sentence)
    assert is_match("c a b", sentence)
    assert is_match("c b a", sentence)


def test_nl_optional_whitespace():
    sentence = parse_sentence(
        "[<doe>] (alle|in) <area>[ ]<lamp> aan [willen | kunnen] [<doe>]"
    )
    slot_lists = {
        "area": TextSlotList.from_strings(["Keuken", "Woonkamer"], allow_template=False)
    }
    expansion_rules = {
        "area": parse_sentence("[de|het] {area}"),
        "doe": parse_sentence("(zet|mag|mogen|doe|verander|maak|schakel)"),
        "lamp": parse_sentence("[de|het] (lamp[en]|licht[en]|verlichting)"),
    }

    for text in [
        "Mogen in de keuken de lampen aan?",
        "Mogen in de keukenlampen aan?",
    ]:
        assert is_match(
            text,
            sentence,
            slot_lists=slot_lists,
            expansion_rules=expansion_rules,
        )


def test_text_slot_list_turkish_dotless_i() -> None:
    """Turkish "I" lower cases to dotless "ı", which str.casefold does not do.

    The candidate index must stay at least as permissive as the ``re.IGNORECASE``
    matching that follows it, otherwise a value like "Işık" ("light") is keyed
    under "işık" and never offered for text that says "ışık".
    """
    names = TextSlotList.from_strings(["Işık", "Isıtıcı", "Lamba"])

    # Spoken/typed in lower case, as speech-to-text produces it.
    assert len(names.get_candidates("ışık kapat")) == 1
    assert len(names.get_candidates("ısıtıcı kapat")) == 1

    # Written the way the value is stored.
    assert len(names.get_candidates("Işık kapat")) == 1

    # A value with no "I" is unaffected.
    assert len(names.get_candidates("lamba kapat")) == 1

    # "İ" folds to two characters, so those values stay unindexed and are
    # always offered; they were already reachable and must remain so.
    dotted = TextSlotList.from_strings(["İstanbul"])
    assert len(dotted.get_candidates("istanbul")) == 1
