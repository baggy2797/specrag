from specrag.tokens import count_tokens


def test_count_tokens_cl100k() -> None:
    assert count_tokens("") == 0
    assert count_tokens("hello world") == 2
