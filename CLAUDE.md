# specrag: working rules

- Design: `docs/superpowers/specs/2026-11-21-specrag-design.md`. The spec is the binding authority. Plans in `docs/superpowers/plans/`.
- Every change is sub-agent implemented, sub-agent reviewed, then reviewed as a whole and landed as **one squashed commit via a PR**, using `.github/pull_request_template.md` (Description, Risk level, Assessment, CI run, Verification). Merge only when CI is green.
- Never commit spec text (the corpus lives in `~/.cache/specrag/`). Tests never hit the network for the corpus.
- Before committing, run `uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest -q`.
