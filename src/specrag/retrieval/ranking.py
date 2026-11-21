def sort_ranking(scores: dict[str, float]) -> list[tuple[str, float]]:
    """Order by descending score; ties break by ID ascending (deterministic)."""
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
