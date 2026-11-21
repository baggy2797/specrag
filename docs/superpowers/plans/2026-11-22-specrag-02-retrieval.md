# specrag Retrieval Implementation Plan (Plan 2 of 5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the parsed section tree into three chunk sets. Index them with BM25, dense and hybrid retrieval plus an optional reranker. Return ranked chunks packed to exactly a 2,500-token budget, ready for the Plan 5 harness to score with `specrag.scoring`.

**Architecture:**
- Each document becomes one **token stream**: its content sections' own text, tokenized with the project tokenizer, in document order, with per-token section attribution.
- Every chunk is a span `[start, end)` of that stream. Chunking, budget truncation and overlap discounting are therefore all exact span arithmetic, and `sections[]`/`tokens[]` are always recomputed from attribution rather than estimated.
- Retrievers return a full ranking. One packer applies the budget for every chunker.

**Tech Stack:** Python 3.12, uv, numpy, sentence-transformers (`BAAI/bge-small-en-v1.5`, `BAAI/bge-reranker-base`) on CPU torch, sqlite3 (stdlib), tiktoken cl100k_base (existing `specrag.tokens`), pydantic v2, typer, pytest.

**Spec:** `docs/superpowers/specs/2026-11-21-specrag-design.md` (v7), §6 "Section IDs and chunk content", "Chunkers", "Retrieval"; §7.2; §7.5 CI bullets; §8.

## Decisions carried from Plan 1 (made here, correctable)

| Open item | Decision | Why |
|---|---|---|
| Reference sections (`Section.kind == "references"`, 67 sections, 40,118 tokens) | **Not chunked, not indexed.** | They are not citable answers and would pollute BM25 with RFC titles. The parser keeps them, so the decision can be reversed in one line. Plan 3 must not use them as gold. |
| Zero-token sections (53 heading-only content sections) | **Produce no chunk.** Their titles appear only in `section+meta` heading paths. | There is no text to retrieve. The hit rule already gives heading-only gold credit through descendants (§7.2). |
| Sections over 800 tokens (71; max 6,770) | **`section` splits them into contiguous, balanced pieces of ≤ 800 tokens**: `n = ceil(len/800)` pieces. **The target is recomputed after each cut** as `remaining / pieces_left`, and each cut must lie in `(prev, prev + 800]` *(re-review N1)*. Each cut goes at the nearest paragraph break (`\n\n`) within ±25% of the target, then at a sentence end (`. ` followed by an uppercase letter), then at a hard token cut through `safe_cut`. If the last piece would still exceed 800, `n` is increased by 1 and the split is redone. Simulated on the real corpus, 80 of 115 cuts land on paragraph breaks; most hard cuts fall in base64 JWS examples (rfc7515 Appendix A and B). No overlap, and every piece keeps the section ID (§6). | This is the spec rule. Balancing avoids tiny tail pieces that can never reach the 50-token hit threshold. The pre-Plan-2 parser PR (`fix/preserve-layout`) makes paragraph breaks real; before it, the corpus had none (review C1). |
| Figures / ASCII art / tables | **Kept with their layout.** After `fix/preserve-layout` they are verbatim, dedented line blocks, and tables are rows joined with ` \| `. The chunkers treat each one as a paragraph unit, so a figure is only hard-cut if it alone exceeds 800 tokens. | Figures are citable text, and the generator and the labeller must be able to read them. |
| list vs tuple on frozen models | **Keep `list`** for consistency with Plan 1. Frozen pydantic models already block reassignment, and no code mutates lists in place (the reviewer checks this). | Changing it now would ripple through Plan 1's tests for no behavioural gain. |

**Prerequisite:** PR `fix/preserve-layout` (the parser keeps paragraph breaks and preformatted layout) is merged (PR #12, bd4125c). The figures below are from that corpus: 1,299 content sections with 356,387 tokens. Each test that quotes one must read it from `snapshots.yaml` (the sum of `own_tokens_total` over content sections), never from a literal.

## Review fixes folded in (plan review 2026-11-22: 3 Critical, 9 Important, 10 Minor)

C1 (no paragraph breaks) is handled by the prerequisite PR and the balanced split above. C2 and C3 are fixed in the keying and the `RetrievalResult` sections. I1–I9 and M1–M10 are fixed in the tasks they name, each marked *(review Xn)*.

## Global Constraints

- The ID format is `<doc>#<number>`. A chunk's `sections[]`/`tokens[]` are parallel and non-empty (existing `Chunk` validator).
- `fixed`: **512 tokens with 64 overlap**, ignoring structure, within one document (it never spans documents).
- `section`: one section's own text per chunk. **A section over 800 tokens is split into pieces that keep the same section ID.**
- `section+meta`: a `section` chunk with `"<DOC LABEL> · <doc title> › <heading path>"` prepended to the indexed text. **The prefix counts toward the 2,500-token budget.** Example: `RFC 9700 · OAuth 2.0 Security BCP › 2 Best Practices › 2.1.1 Authorization Code Grant`.
- Dense: **one pinned model, `bge-small-en-v1.5`, always on CPU** (`device="cpu"`), locally and in CI. Ollama is **never** used for embeddings.
- Hybrid: **reciprocal rank fusion** of BM25 and dense (k = 60).
- Reranker: optional cross-encoder **`bge-reranker-base`, CPU**. It is not in the CI gate.
- Storage: **numpy arrays plus SQLite. No vector-database dependency.**
- **Retrieval depth is a token budget, not a fixed k:** ranked chunks fill **exactly 2,500 tokens**.
  - **The last chunk is truncated to fit**, not dropped, and its `sections[]`/`tokens[]` are recomputed. **At most one chunk per result is truncated, and packing stops after it** *(re-review N2)*, so no stream of tiny fragments is created.
  - Overlap text already included from an earlier `fixed` chunk does not count toward the budget.
  - Report the **tokens used**. It equals 2,500, except that `safe_cut` may pull it back by at most 3 tokens, and it is lower when the ranking is exhausted.
- Embeddings are cached by **(corpus key, chunker, chunker version, model, revision)**. Torch is installed from the **CPU wheel index** on Linux, and the Hugging Face model cache is cached in CI.
- The CI gate finishes in **under 10 minutes** with a warm cache on a 7 GB ubuntu runner.
- Every PR follows the repo's CLAUDE.md: a 5-section body, with local `uv run pytest -rs` output showing 0 corpus skips for corpus or model changes.

## Review Focus

1. **Splitting a multi-byte character at a chunk boundary.** A token window that cuts a UTF-8 character would put `�` into chunk text. Expected: no chunk text in the real corpus contains `�` (Task 2 test `test_no_replacement_chars_in_corpus_chunks`).
2. **A query with no matching terms, or an empty query.** Expected: `bm25.rank` returns `[]` for all-unknown terms. `specrag search ""` exits 2 with "query must not be empty". A query that matches nothing prints "no results" and exits 0 (Tasks 4 and 7b).
3. **A ranking that runs out before 2,500 tokens.** Expected: the packer returns everything with `tokens_used < 2500` and does not raise (Task 3 `test_pack_exhausted_ranking`).
4. **A stale index after re-pinning the corpus.** Expected: the index directory key includes the manifest hash, so a changed manifest triggers a rebuild, never silent reuse (Task 7a `test_index_key_changes_with_manifest_or_sections`).
5. **Tied scores.** Expected: every ranking is deterministic, with ties broken by chunk ID ascending (Task 4 `test_bm25_ties_break_by_chunk_id`, Task 6 `test_rrf_ties_break_by_chunk_id`).

---

## File structure

```
src/specrag/retrieval/
  __init__.py
  streams.py    # DocStream: per-doc token ids + attribution; slice -> (text, sections, tokens)
  chunkers.py   # fixed / section / section+meta → list[Chunk]
  budget.py     # pack(ranked, streams, budget) -> Packed
  bm25.py       # hand-built Okapi BM25
  dense.py      # Embedder protocol, SentenceTransformerEmbedder, DenseIndex, embedding cache
  hybrid.py     # rrf()
  rerank.py     # Reranker protocol, CrossEncoderReranker
  store.py      # SQLite chunk store + corpus_key/index_dir keying
  ranking.py    # sort_ranking (shared tie-break)
  models_pin.py # HF model ids + revisions
  pipeline.py   # RetrievalConfig, Retriever (rank -> optional rerank -> pack)
```

Model change: `Chunk` gains `span: tuple[int, int]` (body token offsets in the doc stream) and `prefix_tokens: int = 0`.

- **`n_tokens`** is the **size of the whole chunk**: `prefix_tokens + (span[1] - span[0])`. It is *not* the budget cost; the cost lives in `Packed.costs` *(review I2)*.
- **Separator tokens** (owner `None`) count in `n_tokens` but not in `tokens[]`, so the invariant is `sum(tokens) <= span length` *(review M2)*.
- **Chunk IDs:** `fixed:<doc_id>:<i:04d>` and `section:<section_id>~<piece>`. `section+meta` uses the prefix `meta:` with the same ID body.
- **Text vs index text** *(review I7)*:
  - `text` is always the **body only**: what is cited, scored and attributed.
  - `index_text` is what retrievers index and embed. It is `prefix + "\n" + text` for meta chunks and `text` otherwise.
  - The **generator (Plan 4) reads `index_text`**, so the prefix it is charged for in the budget is actually shown to it.

**Shared helpers:**
- `retrieval/ranking.py` *(review M5)*: `sort_ranking(scores: dict[str, float]) -> list[tuple[str, float]]` sorts by `(-score, id)`. Every ranker uses it.
- `safe_cut(stream, pos, direction) -> int` lives in `streams.py` *(review I3)*. It moves a cut position by up to 3 tokens until both sides decode without `�`. It is used by fixed windows, section hard cuts and budget truncation.

**Version constants:**
- `CHUNKER_VERSION = "1"` in `chunkers.py`. Bump it on any chunk-affecting change.
- `DENSE_REVISION` and `RERANK_REVISION`: HF commit SHAs as constants *(review M6)*.

**Keys** *(review C2)*:
- `corpus_key() = sha256(corpus/manifest.yaml bytes + sections.jsonl bytes)[:12]`. It uses the repo manifest (the lock file lives in the cache dir, `fetch.py:101`, so it can't be read from `corpus/`) plus the parsed content, so a parser change also invalidates.
- **Index dir:** `index/<chunker>-v<CHUNKER_VERSION>-<corpus_key>`.
- **Embedding file:** `emb-<sha256(corpus_key|chunker|CHUNKER_VERSION|model|revision)[:16]>.npy`. Its sidecar stores the chunk IDs **and** the `sha256` of each `index_text`. Any mismatch triggers a rebuild.

---

### Task 1: Document token streams and the span-aware Chunk

**Files:**
- Modify: `src/specrag/models.py` (`Chunk`)
- Create: `src/specrag/retrieval/__init__.py`, `src/specrag/retrieval/streams.py` (including `safe_cut`), `src/specrag/retrieval/ranking.py`
- Modify: `src/specrag/tokens.py`, adding `encode(text) -> list[int]` and `decode(ids) -> str` beside `count_tokens`
- Test: `tests/retrieval/test_streams.py`, plus `tests/test_models.py` updates

**Interfaces:**
- Consumes: `Section`, `SectionIndex`, `read_sections()` (Plan 1).
- Produces:
  - `DocStream` (frozen dataclass): `doc_id: str`, `ids: list[int]`, `owner: list[str | None]` (same length as `ids`; `None` for separator tokens), `section_spans: dict[str, tuple[int, int]]`.
  - `build_streams(sections: list[Section]) -> dict[str, DocStream]`: content sections with `own_tokens > 0` only, in input order. Each is followed by the separator `"\n\n"`, owned by `None`.
  - `DocStream.attribute(start: int, end: int) -> tuple[list[str], list[int]]`: the section IDs in first-appearance order, with their token counts. Sections with zero tokens are dropped.
  - `DocStream.text(start: int, end: int) -> str`.
  - `Chunk.span: tuple[int, int]` and `Chunk.prefix_tokens: int = 0`. The validator also requires `0 <= span[0] < span[1]` and `n_tokens == prefix_tokens + (span[1] - span[0])`.

- [ ] **Step 1: Write the failing tests**
  - `test_stream_excludes_references_and_empty`: references and zero-token sections are absent from `section_spans`.
  - `test_attribution_matches_own_tokens`: for a 3-section fixture, `attribute(*section_spans[s])` returns `([s], [own_tokens(s)])` for each s.
  - `test_attribute_across_boundary`: a span covering the last 3 tokens of A, a separator and the first 5 tokens of B returns `(["A","B"], [3,5])`.
  - `test_chunk_span_validator`: `n_tokens` must equal the prefix plus the span length; reversed spans raise.
  - `@pytest.mark.corpus test_corpus_streams_cover_all_content_tokens`: the sum of attributed tokens over the real corpus equals the sum of `own_tokens` over content sections with `own_tokens > 0`. Compute it from `read_sections()`, not from a literal.
  - `test_safe_cut_multibyte` *(review I3)*: on a synthetic stream of `"→é漢字🙂"` repeated, any `pos` passed to `safe_cut` returns a position where `decode(ids[:p])` and `decode(ids[p:])` both contain no `�`. Check this for every `pos` in range.
  - `test_sort_ranking_ties` *(review M5)*: equal scores are ordered by ID ascending.
- [ ] **Step 2:** Run `uv run pytest tests/retrieval/test_streams.py tests/test_models.py -v`. Expected: FAIL (module missing, `span` unknown).
- [ ] **Step 3:** Implement. Encode each section separately with the same encoder as `count_tokens`, so attribution equals `own_tokens` exactly. Update the existing `Chunk` constructions in `tests/test_scoring.py` and `tests/test_models.py` with valid spans.
- [ ] **Step 4:** Run `uv run pytest -rs`. Expected: all pass, with 0 corpus skips locally.
- [ ] **Step 5:** Commit `feat(retrieval): per-doc token streams and span-aware Chunk`.

### Task 2: The three chunkers

**Files:**
- Create: `src/specrag/retrieval/chunkers.py`
- Test: `tests/retrieval/test_chunkers.py`

**Interfaces:**
- Consumes: `DocStream`, `build_streams`, `SectionIndex`, `Doc`/`ManifestEntry` (for the title and label).
- Produces:
  - `ChunkerName = Literal["fixed", "section", "section+meta"]`.
  - `chunk_fixed(stream, size=512, overlap=64) -> list[Chunk]`: windows of `size` tokens with step `size - overlap`. The last window ends at the end of the stream, and a window made only of separators is skipped.
  - `chunk_section(stream, idx, max_tokens=800) -> list[Chunk]`.
  - `chunk_section_meta(stream, idx, doc: Doc, max_tokens=800) -> list[Chunk]`: `index_text = prefix + "\n" + text`, `prefix_tokens = count_tokens(prefix + "\n")`.
  - `DOC_META: dict[str, tuple[str, str]]` *(review I5)*: maps every one of the 23 doc IDs to an explicit `(label, short_title)`. Examples: `"rfc9700": ("RFC 9700", "OAuth 2.0 Security BCP")`, `"oidc-core": ("OpenID Connect Core 1.0", "OpenID Connect Core")`, `"oauth-2.1": ("draft-ietf-oauth-v2-1-17", "OAuth 2.1 (draft)")`. A test asserts the keys equal the manifest's doc IDs exactly. The implementer writes the other 20 entries from the manifest titles and lists them in the report for review.
  - `heading_path(section_id, idx) -> str`: ancestors root-first, then self, each as `"<number> <title>"` joined by `" › "`.
  - `make_chunks(name: ChunkerName, streams, idx, docs: dict[str, Doc]) -> list[Chunk]`.
  - `index_text == text` for fixed and section chunks.

- [ ] **Step 1: Write the failing tests**
  - `test_fixed_window_and_overlap`: on a 1,200-token stream the spans are `(0,512),(448,960),(896,1200)`.
  - `test_fixed_never_spans_docs`: all spans of a chunk come from one doc stream.
  - `test_section_one_section_per_chunk`: every chunk has `len(sections)==1`.
  - `test_fixed_short_and_exact_multiple` *(review M7)*:
    - a 300-token stream gives one chunk `(0,300)`;
    - a 960-token stream gives `(0,512),(448,960)`, with no duplicate tail window.
  - `test_section_split_balanced`: an 1,900-token section with paragraph breaks every ~100 tokens gives 3 contiguous pieces. Each piece is ≤ 800 tokens and within ±25% of 633, and every cut lands on a `\n\n`. The pieces cover the section's span exactly, with no gaps or overlap. Every piece has `sections == [sid]`, and the IDs are `section:<sid>~0..2`.
  - `test_section_split_sentence_fallback`: a 1,700-token single paragraph splits at sentence ends.
  - `test_section_split_hard_cut`: a 1,700-token unpunctuated blob still gives pieces of ≤ 800 tokens, cut through `safe_cut`.
  - `test_section_meta_prefix_example`:
    - for a fixture shaped like rfc9700#2.1.1 (with `DOC_META["rfc9700"]`), the prefix is exactly `"RFC 9700 · OAuth 2.0 Security BCP › 2 Best Practices › 2.1.1 Authorization Code Grant"`;
    - `prefix_tokens == count_tokens(prefix + "\n")`;
    - the ID starts with `meta:`;
    - `text` excludes the prefix;
    - `index_text` starts with it *(review M3, I7)*.
  - `test_doc_meta_covers_manifest` *(review I5)*.
  - `@pytest.mark.corpus test_no_replacement_chars_in_corpus_chunks` (Review Focus 1, the regression check; `test_safe_cut_multibyte` is the real guard).
  - `@pytest.mark.corpus test_section_chunks_cover_corpus`: the sum of `tokens` over `section` chunks equals the content total from `read_sections()`.
  - `@pytest.mark.corpus test_all_section_pieces_within_800` *(re-review N1)*: every `section` and `meta` piece has a body of ≤ 800 tokens across the whole corpus, including rfc7515#A.4.1.
  - `@pytest.mark.corpus test_no_references_chunked` *(review M4)*: no chunk's `sections[]` contains a section of kind `references`.
  - `@pytest.mark.corpus` piece-size report: print the min/median/max piece size and the share of cuts made at paragraph, sentence and hard positions, and record them in the PR.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement. Every cut goes through `safe_cut` (Task 1), and spans must stay strictly increasing.
- [ ] **Step 4:** Run `uv run pytest -rs`. Expected: all pass.
- [ ] **Step 5:** Commit `feat(retrieval): fixed, section and section+meta chunkers`.

### Task 3: The budget packer

**Files:**
- Create: `src/specrag/retrieval/budget.py`
- Test: `tests/retrieval/test_budget.py`

**Interfaces:**
- Consumes: `Chunk`, `DocStream`.
- Produces:
  - `BUDGET = 2500`.
  - `Packed` (frozen pydantic): `chunks: list[Chunk]`, `costs: list[int]` (parallel to `chunks`, with `sum(costs) == tokens_used`; review I2), `tokens_used: int`, `exhausted: bool` (true when the ranking ran out before the budget filled; review I8).
  - `pack(ranked: list[Chunk], streams: dict[str, DocStream], budget: int = BUDGET) -> Packed`.

The algorithm, which the tests do not determine on their own:

```
covered[doc] = set of body token positions already packed
for chunk in ranked:
    new = [p for p in range(*chunk.span) if p not in covered[doc]]
    if not new: continue                            # fully duplicated overlap: skip, free
    cost = chunk.prefix_tokens + len(new)
    if used + cost <= budget: take whole; used += cost; continue
    room = budget - used - chunk.prefix_tokens
    if room <= 0: continue                          # prefix doesn't fit: try lower-ranked chunks (review I1)
    cut = safe_cut(position after which exactly `room` new tokens are kept, direction=back)
    truncated = span (start, cut); sections/tokens = stream.attribute(start, cut);
                text/index_text recomputed (prefix kept); n_tokens = prefix + (cut - start)
    if truncated has no sections (e.g. it ends on separators): continue
    take truncated with cost = prefix + new tokens kept; used += that
    break                                           # at most one truncated chunk (re-review N2)
```

Stopping rule: the loop ends after the first truncated chunk, when `used == budget`, or when the ranking is exhausted. A chunk is skipped (and the loop continues) only when it is pure duplicated overlap, when its prefix doesn't fit, or when its truncation would contain no section tokens. So `used` is 2,500, minus at most 3 tokens of `safe_cut` pull-back, unless the ranking is exhausted.

Ranking depth is **n = 1,000** (Tasks 4–7), so exhaustion is rare. `exhausted` records it, and Plan 5 reports mean `tokens_used` per config (spec §6, review I8).

- [ ] **Step 1: Write the failing tests**
  - `test_pack_exact_budget`: with ranked chunks of sizes 1000, 1000 and 1000, `tokens_used == 2500` and the last chunk has a body of 500 tokens.
  - `test_pack_truncation_recomputes_sections`: a truncated chunk that originally spanned A and B, cut inside A, has `sections == ["A"]` and `tokens == [kept]`.
  - `test_pack_fixed_overlap_is_free`: two adjacent fixed chunks (512, overlap 64) cost 512 + 448 = 960.
  - `test_pack_meta_prefix_counts`: a meta chunk with prefix 20 and body 100 costs 120.
  - `test_pack_skips_unfittable_prefix` *(review I1)*: with 10 tokens left, a meta chunk with prefix 20 is skipped, and the next plain chunk is truncated to 10, so `tokens_used == 2500`.
  - `test_pack_at_most_one_truncated` *(re-review N2)*: after a truncation, no further chunk is added, even when smaller chunks follow in the ranking.
  - `test_pack_costs_sum` *(review I2)*: for a fixed-overlap ranking, `sum(costs) == tokens_used`, and a cost is less than `n_tokens` for an overlapped chunk.
  - `test_pack_truncated_on_separator` *(review M2)*: a cut that would leave only separator tokens skips that chunk.
  - `test_pack_exhausted_ranking` (Review Focus 3): ranked chunks totalling 700 tokens give `tokens_used == 700`, `exhausted is True` and no error.
  - `test_pack_empty`: `pack([]) == Packed(chunks=[], costs=[], tokens_used=0, exhausted=True)`.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement as above.
- [ ] **Step 4:** Run `uv run pytest -rs`. Expected: all pass.
- [ ] **Step 5:** Commit `feat(retrieval): 2,500-token budget packer with truncation and overlap discount`.

### Task 4: BM25

**Files:**
- Create: `src/specrag/retrieval/bm25.py`
- Test: `tests/retrieval/test_bm25.py`

**Interfaces:**
- Produces:
  - `tokenize(text: str) -> list[str]`: lowercase, then `re.findall(r"[a-z0-9]+(?:[_-][a-z0-9]+)*", ...)`. This keeps `code_verifier`, `client-id` and `rfc6749` as single terms. No stemming and no stopwords: RFC 2119 words such as MUST and SHOULD carry meaning.
  - `BM25Index.build(chunks: list[Chunk], k1: float = 1.2, b: float = 0.75) -> BM25Index`, indexing `chunk.index_text`.
  - `BM25Index.rank(query: str, n: int = 1000) -> list[tuple[str, float]]`: `(chunk_id, score)` with score > 0, ordered by `sort_ranking`.
  - IDF is the Lucene form `ln(1 + (N - df + 0.5)/(df + 0.5))`.

- [ ] **Step 1: Write the failing tests**
  - `test_tokenize_keeps_identifiers`: `tokenize("The code_verifier MUST use client-id") == ["the","code_verifier","must","use","client-id"]`.
  - `test_bm25_prefers_rare_term`: on a 3-doc fixture, the document containing the rare term ranks first.
  - `test_bm25_hand_computed_score`: a 2-chunk fixture score equals a hand-computed value to 1e-9.
  - `test_bm25_no_match_returns_empty` (Review Focus 2).
  - `test_bm25_ties_break_by_chunk_id` (Review Focus 5).
  - `@pytest.mark.corpus test_bm25_pkce_smoke`: on `section` chunks, "code_verifier code_challenge S256" ranks a chunk citing `rfc7636#4.2` or `rfc7636#4.1` in the top 5.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement with numpy, or dicts over postings. Keep it under about 100 lines.
- [ ] **Step 4:** Run `uv run pytest -rs`. Expected: all pass.
- [ ] **Step 5:** Commit `feat(retrieval): hand-built BM25`.

### Task 5a: ML dependencies, CPU torch and the CI model cache *(review I6, I9)*

**Files:**
- Modify: `pyproject.toml`
  - Dependencies: add `numpy>=1.26` and `sentence-transformers>=3`.
  - Torch index: `[[tool.uv.index]] name="pytorch-cpu" url="https://download.pytorch.org/whl/cpu" explicit=true`, with `[tool.uv.sources] torch = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]`.
  - Markers: `model: needs the dense model` and `reranker: needs the reranker model`.
  - mypy: `[[tool.mypy.overrides]] module = ["sentence_transformers.*", "torch.*", "transformers.*"] ignore_missing_imports = true`.
  - Run `uv lock`.
- Modify: `tests/conftest.py`
  - Skip `model` tests unless `SPECRAG_MODELS=1`.
  - Skip `reranker` tests unless `SPECRAG_RERANKER=1`.
  - Both are off by default.
- Modify: `.github/workflows/ci.yml`
  - In the `corpus-tests` job, cache `~/.cache/huggingface` with key `hf-${{ hashFiles('src/specrag/retrieval/models_pin.py') }}` *(re-review N3: a Python constant can't be read by Actions; hashing the pin file means a revision bump changes the key)*, and set `SPECRAG_MODELS=1`.
  - After the corpus-cache restore, add a step that runs `uv run specrag parse` *(re-review N6)*, so `sections.jsonl` always matches the parser on this commit, never a stale cached copy.
  - On a cache hit, set `HF_HUB_OFFLINE=1`.
  - The reranker is **never** downloaded in CI (spec §7.5).
- Create: `src/specrag/retrieval/models_pin.py` with `DENSE_MODEL`, `DENSE_REVISION`, `RERANK_MODEL` and `RERANK_REVISION`.
  - Look the SHAs up with `curl -s https://huggingface.co/api/models/<id> | jq -r .sha` and record them in the report.

- [ ] **Step 1: Test.** `@pytest.mark.model test_dense_model_loads_on_cpu`: loading `DENSE_MODEL` at `DENSE_REVISION` gives `device == "cpu"` and an embedding dimension of 384.
- [ ] **Step 2:** Run `uv sync` on the M3 and on CI. Record the installed torch wheel (it must be the `+cpu` wheel on Linux) and the HF cache size.
- [ ] **Step 3:** Push the branch. The PR's `corpus-tests` job must stay under the 10-minute timeout on a cold model cache. Record the cold and warm durations in the PR.
- [ ] **Step 4:** Commit `build: CPU torch, sentence-transformers, model pins and CI model cache`.

### Task 5b: Dense retrieval and the embedding cache

**Files:**
- Create: `src/specrag/retrieval/dense.py`
- Test: `tests/retrieval/test_dense.py`

**Interfaces:**
- Produces:
  - `class Embedder(Protocol)`: `name: str`, `revision: str`, `max_seq_tokens: int`, `def embed(self, texts: list[str], *, query: bool) -> np.ndarray` (L2-normalised float32, shape (n, d)), and `def truncated(self, texts: list[str]) -> int` (how many texts exceed the model's own token limit).
  - `SentenceTransformerEmbedder()`:
    - uses `DENSE_MODEL` and `DENSE_REVISION` with `device="cpu"`;
    - when `query=True`, prepends `"Represent this sentence for searching relevant passages: "`;
    - batch size is 32 and `max_seq_length` is 512.
  - `DenseIndex.build(chunks, embedder, cache_dir: Path | None, chunker: ChunkerName) -> DenseIndex`:
    - the cache key is as in "Keys" (C2);
    - the sidecar holds the chunk IDs plus a sha256 of each `index_text`, and any mismatch triggers a rebuild;
    - `DenseIndex.truncated_count: int` records how many chunks exceeded 512 model tokens *(review I4)*. It is printed by `specrag index` and recorded in the provenance (Task 7a).
  - `DenseIndex.rank(query, n=1000) -> list[tuple[str, float]]`: cosine similarity, ordered by `sort_ranking`.

- [ ] **Step 1: Write the failing tests** with a `FakeEmbedder` (deterministic hashed bag-of-words vectors):
  - `test_dense_rank_with_fake_embedder`
  - `test_embedding_cache_hit_skips_embed`: the second build makes zero embed calls.
  - `test_embedding_cache_invalidates_on_text_change`: changing one `index_text` triggers a rebuild *(review C2)*.
  - `test_embedding_cache_key_includes_chunker_version_model_revision`
  - `test_query_instruction_only_on_queries`
  - `test_truncated_count_reported` *(review I4)*: a fake embedder with `max_seq_tokens=10` counts the long texts.
  - `@pytest.mark.model test_real_model_normalised`: shape `(2, 384)` and norms 1 ± 1e-5.
  - Embed-all report *(re-review N4)*: a **local-only** script `scripts/embed_report.py`, not a test, so CI never embeds the whole corpus. It embeds all chunks for each chunker and prints the wall time and `truncated_count`, which go in the PR as the I4 bias disclosure. CI's model test embeds 2 sentences only. The Plan 5 dev-retrieval gate will add an embedding cache to CI keyed on `hashFiles('corpus/manifest.yaml', 'src/specrag/corpus/**', 'src/specrag/retrieval/**')`.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `SPECRAG_MODELS=1 uv run pytest -rs`. Expected: all pass, with 0 corpus and 0 model skips.
- [ ] **Step 5:** Commit `feat(retrieval): CPU dense retrieval with cached embeddings`.

### Task 6: Hybrid RRF and the reranker

**Files:**
- Create: `src/specrag/retrieval/hybrid.py`, `src/specrag/retrieval/rerank.py`
- Test: `tests/retrieval/test_hybrid.py`, `tests/retrieval/test_rerank.py`

**Interfaces:**
- Produces:
  - `rrf(rankings: list[list[tuple[str, float]]], k: int = 60, n: int = 1000) -> list[tuple[str, float]]`:
    - the score is `Σ 1/(k + rank)`, with rank starting at 1;
    - a missing ID contributes 0;
    - ordered by `sort_ranking`.
  - `RERANK_DEPTH = 50`.
  - `class Reranker(Protocol)`: `def score(self, query: str, texts: list[str]) -> list[float]`.
  - `CrossEncoderReranker()`: `RERANK_MODEL` at `RERANK_REVISION`, on CPU.
  - `rerank(query, ranked: list[tuple[Chunk, float]], reranker, depth=RERANK_DEPTH) -> list[tuple[Chunk, float]]`:
    - re-orders the top `depth` by reranker score, with ties broken by original rank;
    - the rest are appended unchanged with their original scores;
    - the reranker reads `index_text`.

- [ ] **Step 1: Write the failing tests**
  - `test_rrf_hand_computed`: A ranked 1st and 3rd, B ranked 2nd and 1st. The scores are `1/61+1/63` and `1/62+1/61`, so B wins.
  - `test_rrf_missing_contributes_zero`
  - `test_rrf_ties_break_by_chunk_id` (Review Focus 5)
  - `test_rerank_only_reorders_top_depth`: uses a fake reranker.
  - `@pytest.mark.reranker test_cross_encoder_prefers_relevant`: for the query "what is PKCE", a PKCE text scores higher than an unrelated JWS text. This runs locally only, never in CI.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `SPECRAG_MODELS=1 SPECRAG_RERANKER=1 uv run pytest -rs` locally. Expected: all pass, with 0 skips.
- [ ] **Step 5:** Commit `feat(retrieval): reciprocal rank fusion and optional cross-encoder rerank`.

### Task 7a: Chunk store and retrieval pipeline *(review C3)*

**Files:**
- Create: `src/specrag/retrieval/store.py`, `src/specrag/retrieval/pipeline.py`
- Test: `tests/retrieval/test_store.py`, `tests/retrieval/test_pipeline.py`

**Interfaces:**
- Produces:
  - `corpus_key() -> str` and `index_dir(chunker) -> Path`, as defined in "Keys" (C2, Review Focus 4).
  - `ChunkStore` over SQLite `chunks.sqlite`:
    - table: `chunks(id TEXT PRIMARY KEY, ord INTEGER, json TEXT)`;
    - methods: `save(chunks)`, `load() -> list[Chunk]` (in `ord` order), and `get_many(ids) -> list[Chunk]` (keeps the input order).
  - `RetrievalConfig` (frozen pydantic):
    - fields: `chunker: ChunkerName = "section"`, `retriever: Literal["bm25","dense","hybrid"] = "bm25"`, `rerank: bool = False`, `budget: int = 2500`, `depth: int = 1000`;
    - `def config_hash(self) -> str`: the sha256 of the canonical JSON, `[:12]`.
  - `RetrievalResult` (frozen pydantic):
    - `query: str`;
    - `config: RetrievalConfig`;
    - `ranking: list[tuple[str, float]]`: the full scored ranking to `depth`, after any rerank. Plan 3 pools from it, and Plan 5 uses it for nDCG@10 and recall@k;
    - `packed: Packed`;
    - `provenance: dict[str, str]`, with keys `corpus_key`, `chunker_version`, `config_hash`, `dense_model`, `dense_revision`, `rerank_revision` (when used) and `dense_truncated` (when dense is used). The values are strings.
  - `Retriever.open(cfg, embedder=None, reranker=None) -> Retriever`:
    - builds or loads the chunks and indexes for `cfg.chunker`;
    - dense and hybrid create a `SentenceTransformerEmbedder` lazily if none is given.
  - `Retriever.retrieve(query: str) -> RetrievalResult` *(re-review N5)*:
    - rank to `depth`;
    - map the IDs to `(Chunk, score)` pairs via the store;
    - rerank if set;
    - project back to `ranking: list[tuple[str, float]]`;
    - `pack` the Chunks in that order.
    - After a rerank, the top `RERANK_DEPTH` scores are reranker scores and the rest are the original scores. Provenance records `"rerank_depth": "50"` so Plan 5 never compares the two kinds of score.
  - `Retriever.chunks: dict[str, Chunk]`, so the harness can score the full ranking with `specrag.scoring`.

- [ ] **Step 1: Write the failing tests**
  - `test_store_roundtrip_preserves_order_and_span` *(review M1)*: `span` comes back as a tuple and the Chunk is equal to the original.
  - `test_index_key_changes_with_manifest_or_sections` (Review Focus 4, C2)
  - `test_index_key_changes_with_chunker_version`
  - `test_retrieve_result_shape`, using fake components:
    - `ranking` is longer than `packed.chunks`;
    - `packed.tokens_used == 2500`;
    - the `provenance` keys are present;
    - `config_hash` is stable across runs and changes when any field changes.
  - `@pytest.mark.corpus test_every_chunker_fills_budget_bm25`: for 5 fixed queries under BM25, each chunker gives `tokens_used == 2500` and `exhausted is False`.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `SPECRAG_MODELS=1 uv run pytest -rs`. Expected: all pass, with 0 skips.
- [ ] **Step 5:** Commit `feat(retrieval): SQLite chunk store and retrieval pipeline with full rankings and provenance`.

### Task 7b: `index` and `search` CLI

**Files:**
- Modify: `src/specrag/cli.py`, `README.md` (a "Search" section)
- Test: `tests/test_cli.py`

**Interfaces:**
- `specrag index --chunker {fixed,section,section+meta,all}` prints the chunk count, timings and (for dense) `truncated: N`.
- `specrag search QUERY [--chunker] [--retriever] [--rerank] [--budget] [--json]`:
  - prints one line per packed chunk, `<rank>. <chunk id>  <sections>  cost=<cost>`, then `tokens used: N` and the `config_hash`;
  - `--json` prints the `RetrievalResult`.
- Exit codes and messages:
  - empty query: exit 2 with "query must not be empty";
  - no results: print "no results" and exit 0 (Review Focus 2);
  - missing corpus: exit 3 with the existing "corpus unavailable" wording;
  - missing or undownloadable model: exit 3 with "model unavailable: <id>@<revision> — run with network once or set HF cache" *(review M10)*. Before the first download, print "downloading <id> (~130 MB) …".

- [ ] **Step 1: Write the failing tests**
  - `test_cli_search_empty_query_exits_2`
  - `test_cli_search_no_results`
  - `test_cli_search_model_unavailable_exits_3`: uses a fake embedder that raises.
  - `test_cli_search_json_roundtrips`
  - `@pytest.mark.corpus test_search_bm25_section_end_to_end`: `specrag search "PKCE code_verifier" --retriever bm25` exits 0 and prints `tokens used: 2500`.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement, and add the README section.
- [ ] **Step 4:** Run `SPECRAG_MODELS=1 uv run pytest -rs`. Expected: all pass, with 0 skips. Time `specrag index --chunker all` and one hybrid search, and record both in the PR.
- [ ] **Step 5:** Commit `feat(cli): index and search commands`.

## Notes for later plans *(review M8, M9)*

- **Plan 3:** assert that no gold section has kind `references`. 165 content sections have 0 < own_tokens < 50, so they hit only when a chunk carries at least 30% of them. Section chunks usually do; fixed windows may not.
- **Plan 5:** report mean `tokens_used`, `exhausted` counts and `dense_truncated` per config. For the failure taxonomy, optionally add per-retriever ranks (BM25 and dense) to hybrid results *(re-review N7, deferred)*. Note that the one truncated chunk per result can be small, so check its effect on `hit_rate_precision` in the first baseline.

## Out of scope (later plans)
- Golden set and the `ignored` sections (Plan 3).
- Generation and citations (Plan 4).
- The experiment runner, `eval/thresholds.yaml`, the dev retrieval gate in CI and statistics (Plan 5).
- Plan 2 adds no metric gate. CI only runs these tests.
