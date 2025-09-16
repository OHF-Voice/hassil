from hassil.util import (
    is_template,
    merge_dict,
    normalize_text,
    normalize_whitespace,
    remove_escapes,
    remove_punctuation,
)


def test_merge_dict():
    base_dict = {"a": 1, "list": [1], "dict": {"a": 1}}
    merge_dict(base_dict, {"a": 2, "list": [2], "dict": {"b": 2}})
    assert base_dict == {"a": 2, "list": [1, 2], "dict": {"a": 1, "b": 2}}


def test_remove_escapes():
    assert remove_escapes("\\[test\\]") == "[test]"


def test_normalize_whitespace():
    assert normalize_whitespace("this    is a      test") == "this is a test"


def test_normalize_text():
    assert normalize_text("tHIS    is A      Test") == "tHIS is A Test"
    assert normalize_text("what’s") == "what's"


def test_is_template():
    assert not is_template("just some plain text")
    assert is_template("[optional] word")
    assert is_template("a {list}")
    assert is_template("a <rule>")
    assert is_template("(a group)")
    assert is_template("an | alternative")


def test_remove_punctuation():
    assert remove_punctuation("test") == "test"
    assert remove_punctuation("test.") == "test"
    assert remove_punctuation("A.C.") == "A.C."
    assert remove_punctuation("A.C") == "A.C"
    assert remove_punctuation("turn A.C. off") == "turn A.C. off"
    assert remove_punctuation("Main St. next") == "Main St next"
    assert remove_punctuation("Chambre d'Ariane") == "Chambre d'Ariane"
    assert remove_punctuation("Chambre d’Ariane") == "Chambre d’Ariane"
