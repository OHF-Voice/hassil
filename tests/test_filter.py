"""Tests for filtering intents by Home Assistant info."""

from hassil import HomeInfo, filter_intents
from hassil.intents import Intents

YAML = """
language: "en"
intents:
  HassTurnOn:
    data:
      - sentences:
          - "turn on the lights in {area}"
        inferred_domain: "light"
      - sentences:
          - "open the curtains in {area}"
        inferred_domain: "cover"
      - sentences:
          - "turn on {name}"
        name_domains:
          - "light"
          - "switch"
          - "fan"
      - sentences:
          - "activate the alarm"
  HassClimateSetTemperature:
    data:
      - sentences:
          - "set the temperature to {temperature}"
        slots:
          domain: "climate"
lists:
  area:
    values:
      - "kitchen"
  name:
    wildcard: true
"""


def _load() -> Intents:
    return Intents.from_dict(__import__("yaml").safe_load(YAML))


def test_inferred_and_name_domains_parsed():
    intents = _load()
    data = intents.intents["HassTurnOn"].data
    assert data[0].inferred_domain == "light"
    assert data[0].domains == {"light"}
    assert data[2].name_domains == {"light", "switch", "fan"}
    assert data[2].domains == {"light", "switch", "fan"}
    # Block with no domain info
    assert data[3].domains == set()
    # Domain from slots
    assert intents.intents["HassClimateSetTemperature"].data[0].domains == {"climate"}


def test_filter_by_domain_keeps_matching_blocks():
    intents = filter_intents(_load(), domains={"light"})
    turn_on = intents.intents["HassTurnOn"].data
    # light (inferred), name_domains has light, and the domainless block remain;
    # the cover block is dropped.
    assert len(turn_on) == 3
    # climate intent is dropped entirely (no light)
    assert "HassClimateSetTemperature" not in intents.intents


def test_filter_drops_domainless_when_requested():
    intents = filter_intents(_load(), domains={"light"}, keep_domainless=False)
    turn_on = intents.intents["HassTurnOn"].data
    assert len(turn_on) == 2  # domainless "activate the alarm" dropped


def test_filter_by_intent_names():
    intents = filter_intents(_load(), intent_names={"HassClimateSetTemperature"})
    assert set(intents.intents) == {"HassClimateSetTemperature"}


def test_filter_with_home_info():
    home = HomeInfo(domains={"cover"}, intent_names={"HassTurnOn"})
    intents = filter_intents(_load(), home=home)
    assert set(intents.intents) == {"HassTurnOn"}
    turn_on = intents.intents["HassTurnOn"].data
    # cover (inferred) + domainless block kept; light blocks dropped.
    assert len(turn_on) == 2


def test_filter_does_not_mutate_original():
    original = _load()
    n_before = len(original.intents["HassTurnOn"].data)
    filter_intents(original, domains={"light"})
    assert len(original.intents["HassTurnOn"].data) == n_before


def test_no_filters_returns_equivalent():
    intents = filter_intents(_load())
    assert set(intents.intents) == {"HassTurnOn", "HassClimateSetTemperature"}
