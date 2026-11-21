"""Section ID helpers. IDs have the form ``<doc>#<number>``."""


def make_id(doc_id: str, number: str) -> str:
    return f"{doc_id}#{number}"


def split_id(section_id: str) -> tuple[str, str]:
    """Split on the first ``#`` (doc IDs may contain hyphens and dots)."""
    doc_id, sep, number = section_id.partition("#")
    if not sep or not doc_id or not number:
        raise ValueError(f"invalid section id: {section_id!r}")
    return doc_id, number


def parent(section_id: str) -> str | None:
    doc_id, number = split_id(section_id)
    if "." not in number:
        return None
    return make_id(doc_id, number.rsplit(".", 1)[0])


def is_descendant(child: str, ancestor: str) -> bool:
    """True if ``child`` is strictly under ``ancestor`` (component-wise)."""
    c_doc, c_num = split_id(child)
    a_doc, a_num = split_id(ancestor)
    if c_doc != a_doc:
        return False
    c_parts = c_num.split(".")
    a_parts = a_num.split(".")
    return len(c_parts) > len(a_parts) and c_parts[: len(a_parts)] == a_parts
