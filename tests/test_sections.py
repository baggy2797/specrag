from specrag.sections import is_descendant, make_id, parent, split_id


def test_parent_numeric_and_appendix() -> None:
    assert parent("rfc6749#4.1.2") == "rfc6749#4.1"
    assert parent("rfc6749#4") is None
    assert parent("rfc7518#A.1") == "rfc7518#A"


def test_is_descendant_is_strict_and_component_wise() -> None:
    assert is_descendant("rfc6749#4.1.2.1", "rfc6749#4.1")
    assert not is_descendant("rfc6749#4.10", "rfc6749#4.1")
    assert not is_descendant("rfc6749#4.1", "rfc6749#4.1")
    assert not is_descendant("rfc6750#4.1.2", "rfc6749#4.1")


def test_roundtrip() -> None:
    assert split_id(make_id("oidc-core", "3.1.2.1")) == ("oidc-core", "3.1.2.1")
    assert split_id("oauth-2.1#A.1") == ("oauth-2.1", "A.1")
