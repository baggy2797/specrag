# specrag Plan 1: Foundation (scaffold, models, scoring rule, corpus) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A tested Python package that downloads the 23 pinned specs, parses each into a section tree, and implements the spec's hit rule and retrieval metrics, which everything else is scored with.

**Architecture:**
- `src/specrag/` holds pydantic models, a pure scoring module, and a `corpus/` subpackage (manifest, fetch, parse).
- The parser handles three HTML dialects behind one function: xml2rfc v3 RFC HTML, "htmlized" plain-text RFC HTML, and OpenID xml2rfc v2 HTML.
- Spec text lives only in `~/.cache/specrag/`, never in git.

**Tech Stack:** Python 3.12, uv, typer, pydantic v2, httpx, beautifulsoup4 + lxml, tiktoken, pyyaml, pytest, ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-11-21-specrag-design.md` (v7). Read §5, §6 "Section IDs and chunk content", §7.2 and §7.3 before starting.

**Series:** this is Plan 1 of 5.
1. Foundation (this plan).
2. Retrieval: chunkers, BM25, dense, hybrid, reranker, token-budget packing.
3. Golden set: drafting, pooling, clustering, split.
4. Generation: Ollama provider, cited prompt, validator, cache.
5. Harness: runner, statistics, release protocol, report, CI gate.

Plans 2–5 are written after this one lands, against its real interfaces.

## Global Constraints

- Python `>=3.12`. Dependencies managed with `uv`. Code passes `ruff check`, `ruff format --check`, and `mypy --strict src`.
- Token counting everywhere uses **tiktoken `cl100k_base`** through `specrag.tokens.count_tokens`. No other counter.
- Section ID format: `<doc>#<number>`, e.g. `rfc6749#4.1.2`; appendices `rfc7518#A.1`. Doc IDs: `rfc<N>`, `oidc-core`, `oidc-discovery`, `oidc-registration`, `oauth-2.1`.
- A section's **own text excludes its child subsections**.
- Spec text is **never committed**. The cache lives in `~/.cache/specrag/`, overridable by env var `SPECRAG_CACHE`.
- The 23 documents and source URLs are exactly those in spec §5. RFCs come from `https://www.rfc-editor.org/rfc/rfc<N>.html`. The draft comes from `https://datatracker.ietf.org/doc/html/draft-ietf-oauth-v2-1-<rev>` (pin the latest revision at first fetch). OpenID comes from `https://openid.net/specs/openid-connect-{core,discovery,registration}-1_0.html`.
- Hit threshold: `min(50, 0.3 × own_tokens(g))`. A "heading-only" section has `own_tokens < 50`.
- Tests must not use the network. Tests that need the real cached corpus are marked `@pytest.mark.corpus` and skipped when the cache is absent.

## Review Focus

1. **Heading-only gold sections** (`own_tokens < 50`, e.g. `rfc6749#4.1` before §4.1.1). A descendant chunk should hit; a parent chunk never should. Pinned in Task 3.
2. **Deep and appendix IDs** (`rfc6749#4.1.2.1`, `rfc7518#A.1`, `rfc7518#A`). `parent()` and `is_descendant()` must not treat `4.10` as a child of `4.1`. Pinned in Task 2.
3. **Network failure, HTTP 404 or hash drift during fetch.** Expect a non-zero exit, a message naming the document and the cause, and **no partial file left in the cache**. Pinned in Task 4.
4. **Page breaks in htmlized RFCs:** "[Page N]" footers, "RFC 6749 … October 2012" headers and form feeds must not leak into section text, and a paragraph split across a page must join. Pinned in Task 6.
5. **Duplicate or missing section numbers in a parsed document.** The parser must fail loudly, naming the document and section ID, rather than silently merging. Pinned in Task 5 (shared validation), used by Tasks 6–7.

---

### Task 1: Project scaffold and CI skeleton

**Files:**
- Create: `pyproject.toml`, `src/specrag/__init__.py`, `src/specrag/cli.py`, `src/specrag/tokens.py`, `tests/test_cli.py`, `tests/test_tokens.py`, `tests/conftest.py`, `.github/workflows/ci.yml`

**Interfaces:**
- Produces:
  - `specrag.__version__: str = "0.1.0"`
  - `specrag.cli.app: typer.Typer`
  - `specrag.tokens.count_tokens(text: str) -> int`
  - a pytest marker `corpus`, registered in `pyproject.toml` and auto-skipped in `conftest.py` when `corpus_cache_dir()` has no `manifest.lock.json`. `corpus_cache_dir` is defined in Task 4. Until then, the conftest skips the marker unconditionally.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_cli.py
from typer.testing import CliRunner
from specrag.cli import app
def test_version():
    r = CliRunner().invoke(app, ["--version"])
    assert r.exit_code == 0 and "0.1.0" in r.output

# tests/test_tokens.py
from specrag.tokens import count_tokens
def test_count_tokens_cl100k():
    assert count_tokens("") == 0
    assert count_tokens("hello world") == 2
```

- [ ] **Step 2:** Run `uv run pytest -q`. Expected: FAIL (module not found).
- [ ] **Step 3: Implement**
  - `pyproject.toml`:
    - hatchling build backend; `[project.scripts] specrag = "specrag.cli:app"`.
    - Dependencies: `typer>=0.12`, `pydantic>=2.6`, `httpx>=0.27`, `beautifulsoup4>=4.12`, `lxml>=5`, `tiktoken>=0.7`, `pyyaml>=6`.
    - Dev group: `pytest`, `pytest-cov`, `ruff`, `mypy`, `types-PyYAML`.
    - pytest `markers = ["corpus: needs fetched corpus"]`; ruff line-length 100; mypy strict.
  - `count_tokens` caches the encoder with `functools.cache`.
  - `--version` is a typer eager callback.
- [ ] **Step 4:** Run `uv run pytest -q && uv run ruff check . && uv run mypy src`. Expected: PASS, clean.
- [ ] **Step 5:** `ci.yml`: on `push` and `pull_request`, ubuntu-latest, `astral-sh/setup-uv` pinned by commit SHA, `uv sync`, then ruff, mypy and pytest. Set `permissions: contents: read`, `timeout-minutes: 10` and `concurrency`. These conventions match hardened-actions-template.
- [ ] **Step 6:** Commit with `feat: project scaffold, token counter, CI`, then push.

### Task 2: Section IDs and core models

**Files:**
- Create: `src/specrag/sections.py`, `src/specrag/models.py`, `tests/test_sections.py`, `tests/test_models.py`

**Interfaces:**
- Produces, in `sections.py`:
  - `make_id(doc_id: str, number: str) -> str`
  - `split_id(section_id: str) -> tuple[str, str]`
  - `parent(section_id: str) -> str | None`. Rules: `rfc6749#4.1.2` → `rfc6749#4.1`; `rfc7518#A.1` → `rfc7518#A`; top level → `None`.
  - `is_descendant(child: str, ancestor: str) -> bool`. Strict: false for equal IDs and for different docs.
- Produces, in `models.py` (pydantic, frozen):
  - `class DocStatus(StrEnum)`: `STANDARDS_TRACK`, `BCP`, `INFORMATIONAL`, `DRAFT`, `OPENID_FINAL`
  - `class Doc`: `doc_id: str`, `title: str`, `status: DocStatus`, `url: str`, `updates: list[str] = []`, `obsoletes: list[str] = []`
  - `class Section`: `id: str`, `doc_id: str`, `number: str`, `title: str`, `parent_id: str | None`, `depth: int`, `own_text: str`, `own_tokens: int`
  - `class SectionIndex`, built from `list[Section]`, with:
    - `get(id) -> Section`, raising `KeyError` naming the ID
    - `own_tokens(id) -> int`
    - `__contains__`
    - `ancestors(id) -> list[str]`
  - `class Chunk`: `id: str`, `doc_id: str`, `text: str`, `index_text: str`, `sections: list[str]`, `tokens: list[int]`, `n_tokens: int`. A validator requires `len(sections) == len(tokens) >= 1`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_sections.py
from specrag.sections import parent, is_descendant, make_id, split_id
def test_parent_numeric_and_appendix():
    assert parent("rfc6749#4.1.2") == "rfc6749#4.1"
    assert parent("rfc6749#4") is None
    assert parent("rfc7518#A.1") == "rfc7518#A"
def test_is_descendant_is_strict_and_component_wise():
    assert is_descendant("rfc6749#4.1.2.1", "rfc6749#4.1")
    assert not is_descendant("rfc6749#4.10", "rfc6749#4.1")   # Review Focus 2
    assert not is_descendant("rfc6749#4.1", "rfc6749#4.1")
    assert not is_descendant("rfc6750#4.1.2", "rfc6749#4.1")
def test_roundtrip():
    assert split_id(make_id("oidc-core", "3.1.2.1")) == ("oidc-core", "3.1.2.1")
```

```python
# tests/test_models.py
import pytest
from pydantic import ValidationError
from specrag.models import Chunk
def test_chunk_requires_parallel_sections_tokens():
    with pytest.raises(ValidationError):
        Chunk(id="c", doc_id="rfc6749", text="t", index_text="t", sections=["rfc6749#1"], tokens=[], n_tokens=1)
```

  Also add a `SectionIndex.ancestors` test: `["rfc6749#4.1", "rfc6749#4"]` for `rfc6749#4.1.2`.
- [ ] **Step 2:** Run `uv run pytest tests/test_sections.py tests/test_models.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement. `is_descendant` compares dot-separated components, never string prefixes.
- [ ] **Step 4:** Run the same command. Expected: PASS. Then run mypy: clean.
- [ ] **Step 5:** Commit with `feat: section IDs and core models`.

### Task 3: Hit rule and retrieval metrics (spec §7.2–7.3)

**Files:**
- Create: `src/specrag/scoring.py`, `tests/test_scoring.py`

**Interfaces:**
- Consumes: `Chunk`, `SectionIndex`, `is_descendant` (Task 2).
- Produces:
  - `hit_threshold(own_tokens: int) -> float`, which returns `min(50, 0.3 * own_tokens)`
  - `chunk_hits(chunk: Chunk, gold_id: str, idx: SectionIndex) -> bool`
  - `hit_set(chunk: Chunk, gold: set[str], idx: SectionIndex) -> set[str]`
  - `recall(ranked: list[Chunk], gold: set[str], idx: SectionIndex, ignored: set[str] = frozenset()) -> float | None`. Returns `None` when `gold - ignored` is empty (unanswerable questions are excluded).
  - `mrr(...) -> float | None`
  - `ndcg_at(ranked, gold, idx, k: int = 10, ignored=...) -> float | None`
  - `hit_rate_precision(...) -> float | None`
  - `condensed(ranked, judged: set[str], idx) -> list[Chunk]`, which drops chunks whose `sections` are all unjudged.

**Rules** (copy into docstrings):
- A chunk hits `g` if `g in chunk.sections` with that entry's tokens `>= hit_threshold(own_tokens(g))`.
- Otherwise, only if `own_tokens(g) < 50`, it hits when some entry `s` is a descendant of `g` with tokens `>= hit_threshold(own_tokens(s))`.
- A parent never hits a child.
- Each gold section is counted once across the ranking.
- `ignored` sections are removed from `gold`, and chunks whose only hits are ignored are removed from the ranking before nDCG, MRR and precision.
- nDCG uses binary relevance: rank *i* is relevant if its chunk hits a gold section not hit by an earlier rank. IDCG = `min(|gold|, k)` relevant ranks.

- [ ] **Step 1: Write failing tests.** Build a tiny `SectionIndex` fixture: `rfc6749#4` (own 10 tokens), `#4.1` (own 0), `#4.1.1` (own 200), `#4.1.2` (own 40), `#4.10` (own 120). Add a helper in the test module, `mk(sections: list[str], tokens: list[int]) -> Chunk`, which builds a chunk with dummy text, `doc_id="rfc6749"` and `n_tokens=sum(tokens)`.

```python
def test_direct_hit_needs_threshold(idx):
    assert chunk_hits(mk(["rfc6749#4.1.1"], [60]), "rfc6749#4.1.1", idx)      # 60 >= min(50, 60)
    assert not chunk_hits(mk(["rfc6749#4.1.1"], [49]), "rfc6749#4.1.1", idx)
def test_small_section_threshold_is_30pct(idx):
    assert chunk_hits(mk(["rfc6749#4.1.2"], [12]), "rfc6749#4.1.2", idx)      # 0.3*40 = 12
def test_heading_only_gold_credited_via_descendant(idx):                      # Review Focus 1
    assert chunk_hits(mk(["rfc6749#4.1.1"], [80]), "rfc6749#4.1", idx)
def test_parent_never_hits_child(idx):
    assert not chunk_hits(mk(["rfc6749#4.1"], [0]), "rfc6749#4.1.1", idx)
def test_sibling_prefix_is_not_descendant(idx):                               # Review Focus 2
    assert not chunk_hits(mk(["rfc6749#4.10"], [120]), "rfc6749#4.1", idx)  # 4.1 is heading-only, but 4.10 isn't under it
def test_gold_counted_once_and_unanswerable_excluded(idx):
    r = [mk(["rfc6749#4.1.1"], [100]), mk(["rfc6749#4.1.1"], [100])]
    assert recall(r, {"rfc6749#4.1.1", "rfc6749#4.1.2"}, idx) == 0.5
    assert recall(r, set(), idx) is None
def test_ndcg_binary_dedup(idx):
    r = [mk(["rfc6749#4.10"], [120]), mk(["rfc6749#4.1.1"], [100]), mk(["rfc6749#4.1.1"], [100])]
    assert ndcg_at(r, {"rfc6749#4.1.1"}, idx, k=10) == pytest.approx(1 / math.log2(3))
def test_ignored_removed_before_scoring(idx):
    r = [mk(["rfc6749#4.1.2"], [40]), mk(["rfc6749#4.1.1"], [100])]
    assert mrr(r, {"rfc6749#4.1.1", "rfc6749#4.1.2"}, idx, ignored={"rfc6749#4.1.2"}) == 1.0
```

  Add `test_non_heading_gold_gets_no_descendant_credit`. It uses a second fixture where `#4` has own 200 tokens and asserts `not chunk_hits(mk(["rfc6749#4.1.1"], [80]), "rfc6749#4", idx2)`. Also add a test that a chunk can hit two gold sections at once (a `fixed` chunk) and is counted for both in `recall`.
- [ ] **Step 2:** Run `uv run pytest tests/test_scoring.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement the signatures above. Pure functions, no I/O.
- [ ] **Step 4:** Run the same command. Expected: PASS. Coverage for `scoring.py` should be 100% (`--cov=specrag.scoring`).
- [ ] **Step 5:** Commit with `feat: hit rule and retrieval metrics per spec 7.2/7.3`.

### Task 4: Corpus manifest and fetch

**Files:**
- Create: `corpus/manifest.yaml`, `src/specrag/corpus/__init__.py`, `src/specrag/corpus/manifest.py`, `src/specrag/corpus/fetch.py`, `tests/corpus/test_fetch.py`
- Modify: `src/specrag/cli.py` (add the `fetch` command), `tests/conftest.py` (use `corpus_cache_dir`)

**Interfaces:**
- Consumes: `Doc`, `DocStatus` (Task 2).
- Produces:
  - `corpus_cache_dir() -> Path`: `$SPECRAG_CACHE` or `~/.cache/specrag`
  - `load_manifest(path: Path = Path("corpus/manifest.yaml")) -> list[ManifestEntry]`
  - `ManifestEntry(Doc)` adds `sha256: str | None`
  - `fetch_all(entries, *, client: httpx.Client, pin: bool = False, retries: int = 3) -> FetchReport`
  - `raw_path(doc_id: str) -> Path`
  - `class FetchError(Exception)` with fields `doc_id` and `reason`
  - CLI `specrag fetch [--pin]`: exit 0 on success; exit 3 on drift, with stderr `corpus unavailable: not caused by this change — <doc_id>: <reason>`

**Manifest:** the 23 entries from spec §5, each with `doc_id`, `title`, `status`, `url`, `updates` and `obsoletes` taken from each RFC's header, and `sha256: null` until pinned. `--pin` writes hashes back to the YAML and records the draft revision. The resulting manifest is committed. It holds hashes and URLs only.

- [ ] **Step 1: Write failing tests** using `httpx.MockTransport`:
  - `test_fetch_writes_atomically_and_verifies_sha`: a correct hash leaves the file cached.
  - `test_hash_drift_raises_and_leaves_no_file`: Review Focus 3. A wrong hash raises `FetchError(reason~"sha256 mismatch")`, and `raw_path` doesn't exist.
  - `test_404_raises_named_error`: the message contains the doc ID and `404`.
  - `test_retries_then_succeeds`: two 503 responses, then 200, succeeds. Retry backoff is injectable so the test doesn't sleep.
  - `test_manifest_has_23_unique_docs`: a real `corpus/manifest.yaml`, with IDs unique and `oauth-2.1` status `DRAFT`, `rfc9700` status `BCP`.
  - `test_cli_drift_exit_code_and_message`
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement. Write to a temp file in the same directory, then `os.replace`, only after the hash checks out. Write `manifest.lock.json` (doc ID → sha) to the cache after a full successful run.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Run `uv run specrag fetch --pin` for real. Expected: 23 files in the cache and hashes written. Commit the manifest with the message `feat: pinned corpus manifest and fetch`.

### Task 5: Dialect detection, shared validation and the xml2rfc v3 RFC parser

**Files:**
- Create: `src/specrag/corpus/parse.py`, `src/specrag/corpus/parse_v3.py`, `tests/corpus/fixtures/v3_min.html`, `tests/corpus/test_parse_v3.py`

**Interfaces:**
- Consumes: `Section`, `SectionIndex`, `make_id`, `parent` (Task 2), `count_tokens`, `raw_path` (Task 4).
- Produces:
  - `class Dialect(StrEnum)`: `RFC_V3`, `RFC_HTMLIZED`, `OPENID_V2`
  - `detect_dialect(html: str) -> Dialect`
  - `parse_doc(doc_id: str, html: str) -> list[Section]`, which dispatches by dialect and then calls `validate_sections`
  - `validate_sections(doc_id: str, sections: list[Section]) -> None`, which raises `ParseError` on a duplicate ID, an orphaned parent or an empty document (Review Focus 5)
  - `class ParseError(Exception)`
  - `parse_v3(doc_id: str, soup: BeautifulSoup) -> list[Section]`

- [ ] **Step 1: Recon (no code).** Open three cached files: `rfc9700` (expected v3), `rfc6749` (expected htmlized) and `oidc-core`.
  - Note the actual markers: v3 uses `<section id="section-N">` with `a.section-number`; htmlized uses `<pre>` with `span.h2…h6 a.selflink[id^=section-]`; OpenID uses `id="rfc.section.N"`.
  - Write them as a comment block at the top of `parse.py`.
  - If the markers differ from these expectations, note the real ones and continue with them.
- [ ] **Step 2: Write failing tests.**
  - The fixture `v3_min.html` is **synthetic, written by hand**, mimicking v3 structure: §1 with text, §2 heading-only, §2.1 and §2.2 with text, and Appendix A.1.
  - Tests:
    - `detect_dialect(fixture) == RFC_V3`.
    - Parsed IDs are `["rfcX#1", "rfcX#2", "rfcX#2.1", "rfcX#2.2", "rfcX#A", "rfcX#A.1"]`.
    - `#2`'s `own_text` excludes the §2.1 text and `own_tokens < 50`.
    - Titles are stripped of numbering.
    - `validate_sections` raises `ParseError` mentioning `rfcX#2.1` when it's duplicated.
- [ ] **Step 3:** Run the tests. Expected: FAIL.
- [ ] **Step 4:** Implement. `own_text` collects the section element's text minus nested `<section>` children, with whitespace normalised. Drop the RFC "Table of Contents", "Authors' Addresses" and front-matter "Status of This Memo" and "Copyright Notice" sections, since they're not citable content.
- [ ] **Step 5:** Run the tests. Expected: PASS.
- [ ] **Step 6:** Commit with `feat: dialect detection, section validation, v3 RFC parser`.

### Task 6: Htmlized plain-text RFC parser

**Files:**
- Create: `src/specrag/corpus/parse_htmlized.py`, `tests/corpus/fixtures/htmlized_min.html`, `tests/corpus/test_parse_htmlized.py`

**Interfaces:**
- Consumes: Task 5's `parse_doc` dispatch, `validate_sections`, `Section`.
- Produces: `parse_htmlized(doc_id: str, soup: BeautifulSoup) -> list[Section]`.

- [ ] **Step 1: Write failing tests.** The synthetic fixture contains, inside `<pre>`:
  - section headings marked like the recon found;
  - a paragraph split by a page break: `\f`, a footer line `Hardt   Standards Track   [Page 5]`, then a header line `RFC 6749   The OAuth 2.0 Authorization Framework   October 2012`.

  Tests:
  - `test_page_furniture_removed`: Review Focus 4. No section's `own_text` contains `[Page`, `\f` or the running header.
  - `test_paragraph_joined_across_page_break`: the two halves of the split sentence appear contiguous with a single space.
  - `test_ids_and_nesting`: same expectations as Task 5's ID test.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement. Get the plain text of `<pre>` and drop lines matching:
  - the footer: `^\S.*\[Page \d+\]\s*$`
  - the header: `^RFC \d+ .* \w+ \d{4}\s*$`

  Strip `\f` and collapse the blank lines left at page joins. Then split on the heading anchors in document order, and assign `own_text` to the span up to the next heading of any depth.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit with `feat: htmlized RFC parser with page-furniture removal`.

### Task 7: OpenID xml2rfc v2 parser

**Files:**
- Create: `src/specrag/corpus/parse_openid.py`, `tests/corpus/fixtures/openid_min.html`, `tests/corpus/test_parse_openid.py`

**Interfaces:**
- Produces: `parse_openid(doc_id: str, soup: BeautifulSoup) -> list[Section]`, dispatched by `parse_doc`.

- [ ] **Step 1: Write failing tests.** The synthetic fixture uses `<h2 id="rfc.section.3">`, `<h3 id="rfc.section.3.1">`, `<h4 id="rfc.section.3.1.2">` and paragraphs between them. The expected IDs are `oidc-core#3`, `#3.1`, `#3.1.2`. `own_text` excludes child text. A heading-level jump (h2 → h4) still sets `parent_id` from the **number**, not the tag level.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement. Walk the siblings after each heading until the next `rfc.section.*` heading. Take `parent_id` from `sections.parent()`.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit with `feat: OpenID spec parser`.

### Task 8: Parse the whole corpus, add snapshot tests and the `sections` CLI

**Files:**
- Create: `src/specrag/corpus/build.py`, `tests/corpus/test_snapshots.py`, `tests/corpus/snapshots.yaml`
- Modify: `src/specrag/cli.py` (add `specrag parse` and `specrag sections <doc_id>`)

**Interfaces:**
- Consumes: `load_manifest`, `raw_path`, `parse_doc`.
- Produces:
  - `build_sections(entries) -> SectionIndex`, which writes `corpus_cache_dir()/sections.jsonl`
  - `load_section_index() -> SectionIndex`, which reads that file (used by Plans 2–5)
  - CLI `specrag parse`, which prints a per-document count of sections and own tokens
  - CLI `specrag sections rfc6749`, which prints an ID/title/own-tokens tree

**Snapshot values** (`snapshots.yaml`, committed; facts about structure only, no spec text): per document, `n_sections`, `first_id`, `last_id`, and 3 `(id, title)` samples. Fill it in from the first real parse **after reading the output by hand against the spec's own table of contents**. For example, check that RFC 6749 has §4.1.2 "Authorization Response" and §10.12 "Cross-Site Request Forgery", and that RFC 9700 has §2.1.1.

- [ ] **Step 1:** Write `test_snapshots.py` (`@pytest.mark.corpus`), which parameterises over the documents in `snapshots.yaml` and asserts each field. Add `test_every_doc_parses_and_validates` (also `corpus`), which asserts at least 10 sections per document and no `ParseError`.
- [ ] **Step 2:** Run `uv run specrag parse` and inspect the output. For each of the 23 documents, compare the section counts and titles against the document's table of contents in a browser. Fix any parser bugs, and add the fixture case to the matching Task 5–7 test.
- [ ] **Step 3:** Fill in `snapshots.yaml`. Run `uv run pytest -m corpus -q`. Expected: PASS for all 23.
- [ ] **Step 4:** Run `uv run pytest -q` with no cache (`SPECRAG_CACHE=/tmp/empty`). Expected: corpus tests SKIPPED, everything else passes.
- [ ] **Step 5:** Commit with `feat: full-corpus parse, snapshots, sections CLI`, then push. CI should be green.

## Parallel execution map (for sub-agents)

```
Task 1 ──> Task 2 ──┬──> Task 3                                  (stream A: scoring)
                    └──> Task 4 ──> Task 5 ──┬──> Task 6 ──┐      (stream B: corpus)
                                             └──> Task 7 ──┴──> Task 8
```

- Tasks 1–2 run in sequence first.
- Then **Task 3 runs in parallel with Tasks 4–8**.
- Tasks 6 and 7 can run in parallel after Task 5.
- Task 4 Step 5 (the real fetch) and Task 8 Step 2 (checking against the table of contents) need network and judgement, so the controller runs them or checks them, not a sub-agent alone.
