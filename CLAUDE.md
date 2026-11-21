# specrag: working rules

- Design: `docs/superpowers/specs/2026-11-21-specrag-design.md`. The spec is the binding authority. Plans in `docs/superpowers/plans/`.
- Every change is sub-agent implemented, sub-agent reviewed, then reviewed as a whole and landed as **one squashed commit via a PR**, using `.github/pull_request_template.md` (Description, Risk level, Assessment, CI run, Verification). Merge only when CI is green.
- Never commit spec text (the corpus lives in `~/.cache/specrag/`). Tests never hit the network for the corpus.
- Before committing, run `uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest -q`.
- CI (`.github/workflows/`): `ci.yml` runs lint, mypy, tests, a `corpus-tests` job (cached corpus, 0 corpus skips, coverage gate) and actionlint; `corpus.yml` refreshes the corpus cache weekly and opens a "Corpus re-pin needed" issue on failure; `codeql.yml` and dependabot run alongside; pre-commit runs locally. PRs touching `src/specrag/corpus/` or `src/specrag/models.py` must still paste local `uv run pytest -rs` output showing 0 skipped corpus tests.
- Dev setup: after `uv sync --group dev`, run `uv run pre-commit install` once (ruff check --fix, ruff format and hygiene hooks run on commit).
