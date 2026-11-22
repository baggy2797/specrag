# specrag

[![CI](https://github.com/baggy2797/specrag/actions/workflows/ci.yml/badge.svg)](https://github.com/baggy2797/specrag/actions/workflows/ci.yml)
[![CodeQL](https://github.com/baggy2797/specrag/actions/workflows/codeql.yml/badge.svg)](https://github.com/baggy2797/specrag/actions/workflows/codeql.yml)
[![Corpus](https://github.com/baggy2797/specrag/actions/workflows/corpus.yml/badge.svg)](https://github.com/baggy2797/specrag/actions/workflows/corpus.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**specrag** is a cited question-answering system over the public OAuth 2.0, OpenID Connect and JWT standards (23 pinned documents). It is designed so that its real product is the **evaluation harness**. Answers are meant to cite section IDs such as `rfc9700#2.1.1`. Every retrieval result carries full provenance, and every headline number is meant to come from a pre-registered, statistically honest comparison. It's hand-built on small libraries, with no RAG framework and no vector database. After a one-off download it runs offline on a CPU laptop.

> **Status:** the corpus pipeline and the retrieval stack (Plans 1 and 2) are built, tested and merged. The golden set, answer generation and the eval harness (Plans 3 to 5) are **designed but not built**. There are **no evaluation results yet**, and this README doesn't claim any.

## Contents

- [Why this project](#why-this-project)
- [Status](#status)
- [Quickstart](#quickstart)
- [CLI reference](#cli-reference)
- [The corpus](#the-corpus)
- [How it works](#how-it-works)
- [Evaluation design (planned)](#evaluation-design-planned)
- [Engineering process](#engineering-process)
- [Known limitations](#known-limitations)
- [Repo layout](#repo-layout)
- [Development](#development)
- [License](#license)

## Why this project

Most RAG demos show that something *works*. specrag is built to show *how well* it works, measured in a way that holds up to a sceptical reviewer. The [design](docs/superpowers/specs/2026-11-21-specrag-design.md) (v7) commits to these rules up front:

- **Pre-registered comparisons.** There are three primary comparisons (P1 to P3), each with fixed metrics and views and Holm correction. They're fixed before the test split is ever evaluated. Everything else is labelled exploratory.
- **Clustered bootstrap statistics.** Questions that share gold sections are correlated. Confidence intervals therefore resample whole clusters, and comparisons use a paired cluster bootstrap on per-question differences.
- **A mandatory closed-book baseline.** RAG is compared with the same model answering without retrieval (P3). "Retrieval helps" is then a measured claim, not an assumption.
- **Release-only test evaluation.** Dev is for development. Test is evaluated only by a tagged release, and every test run is logged. Primary results come from the first release only.
- **Deterministic headline metrics.** LLM-judge scores are secondary. They're calibrated against blind human labels and shown as "uncalibrated" if calibration fails.

## Status

| Plan | Scope | Status |
|---|---|---|
| 1. Foundation | Scaffold, section IDs, data models, hit rule and retrieval metrics, pinned corpus fetch, three HTML parsers, snapshot tests, `sections` CLI | Done ([plan](docs/superpowers/plans/2026-11-21-specrag-01-foundation.md)) |
| 2. Retrieval | Token streams, three chunkers, 2,500-token budget packer, BM25, dense, RRF hybrid, cross-encoder reranker, SQLite chunk store, `index` and `search` CLI | Done ([plan](docs/superpowers/plans/2026-11-22-specrag-02-retrieval.md)) |
| 3. Golden set | 145 questions, pooling, pre-judging, clustered dev/test split | Planned |
| 4. Generation | Ollama provider, cited prompt, citation validator, caching | Planned |
| 5. Harness and report | Experiment runner, statistics, release protocol, report, CI eval gate | Planned |

**Works today:** `fetch`, `parse`, `sections`, `index` and `search`.

**Not built yet:** `eval`, `release`, `report`, answer generation, and any metric computed over real questions.

## Quickstart

You need [uv](https://docs.astral.sh/uv/) and Python 3.12; uv installs Python if it's missing. On Linux, torch is installed from the CPU wheel index.

```bash
git clone https://github.com/baggy2797/specrag.git
cd specrag
uv sync                               # ~1.1 GB virtualenv (torch + sentence-transformers)
uv run specrag fetch                  # download and verify the 23 pinned documents
uv run specrag parse                  # parse them into sections.jsonl
uv run specrag index --chunker all    # build the chunk stores (under a second each)
uv run specrag search "PKCE code_verifier" --retriever bm25
```

Dense and hybrid retrieval download a pinned model the first time they run:

| Download | Size | When |
|---|---|---|
| Virtualenv (`uv sync`) | ~1.1 GB | once |
| Corpus in `~/.cache/specrag` | ~23 MB, chunk stores and embeddings included | `specrag fetch` |
| Dense model `BAAI/bge-small-en-v1.5` | ~130 MB | first `--retriever dense` or `hybrid` |
| Reranker `BAAI/bge-reranker-base` | ~1.1 GB | first `--rerank` |

After these downloads, **everything runs offline on CPU**. You don't need a GPU or a paid API. Set `HF_HUB_OFFLINE=1` to make sure nothing contacts the Hugging Face Hub. `SPECRAG_CACHE` moves the corpus cache.

```bash
uv run specrag index --chunker all --dense     # also embed every chunk (~3 min on an M3, CPU)
uv run specrag search "how does DPoP bind tokens" --retriever hybrid
```

Add `--rerank` to any search to reorder the top 50 with the cross-encoder.

### Sample output

All output below is real, from this commit. It's trimmed only where marked.

`specrag sections rfc7636 | head` shows the section ID, title and own-text token count:

```text
  rfc7636#1  Introduction  (1110)
    rfc7636#1.1  Protocol Flow  (389)
  rfc7636#2  Notational Conventions  (265)
  rfc7636#3  Terminology  (159)
    rfc7636#3.1  Abbreviations  (39)
  rfc7636#4  Protocol  (0)
    rfc7636#4.1  Client Creates a Code Verifier  (236)
    rfc7636#4.2  Client Creates the Code Challenge  (204)
    rfc7636#4.3  Client Sends the Code Challenge with the Authorization Request  (74)
    rfc7636#4.4  Server Returns the Code  (128)
```

`specrag index --chunker all`, with the chunk stores already built:

```text
fixed: 808 chunks in 0.4s
section: 1365 chunks in 0.3s
section+meta: 1365 chunks in 0.3s
```

`specrag search "PKCE code_verifier" --retriever bm25` shows the rank, chunk ID, the sections the chunk contains, and its budget cost in tokens:

```text
1. section:rfc9700#4.5.3.1~0  rfc9700#4.5.3.1  cost=121
2. section:rfc9700#4.8.2~0  rfc9700#4.8.2  cost=240
3. section:rfc9700#2.1.1~0  rfc9700#2.1.1  cost=503
4. section:rfc9700#4.8.1~0  rfc9700#4.8.1  cost=334
5. section:oauth-2.1#7.5.1.1~0  oauth-2.1#7.5.1.1  cost=350
6. section:rfc7636#7.2~0  rfc7636#7.2  cost=344
7. section:rfc7636#6.2~0  rfc7636#6.2  cost=253
8. section:rfc9449#11.9~0  rfc9449#11.9  cost=333
9. section:rfc7636#1.1~0  rfc7636#1.1  cost=22
tokens used: 2500
config: 74d62153bce2
```

The last chunk is the truncated one. `rfc7636#1.1` has 389 tokens of its own text, but only 22 fit, so the budget is filled to 2,500 tokens (minus at most 3 when a cut is moved to a character boundary).

`specrag search "how does DPoP bind tokens" --retriever hybrid` (model-loading lines on stderr omitted):

```text
1. section:rfc9449#4~0  rfc9449#4  cost=220
2. section:rfc9449#5~2  rfc9449#5  cost=518
3. section:rfc9449#1~0  rfc9449#1  cost=395
4. section:rfc9449#10.1~0  rfc9449#10.1  cost=359
5. section:rfc9449#7~0  rfc9449#7  cost=347
6. section:rfc9449#3~0  rfc9449#3  cost=661
tokens used: 2500
config: b465225149a5
```

<details>
<summary><code>specrag index --chunker all --dense</code>, first build on an M3 MacBook (CPU)</summary>

`fixed` embeddings were already cached from an earlier run; the other two were built fresh. Progress bars and loader messages are omitted.

```text
fixed: 808 chunks in 5.4s, dense truncated: 666
section: 1365 chunks in 85.8s, dense truncated: 257
section+meta: 1365 chunks in 88.8s, dense truncated: 287
```

</details>

<details>
<summary><code>specrag parse</code>: sections and own-text tokens per document</summary>

```text
doc_id                 sections  own_tokens
rfc6749                     119       24724
rfc6750                      26        6155
rfc7636                      30        6250
rfc7009                      19        3320
rfc7662                      17        5917
rfc8414                      26        8915
rfc8252                      35        7698
rfc8628                      27        7259
rfc9126                      26        9636
rfc9449                      51       19336
rfc8705                      39        9452
rfc9068                      25        6106
rfc9700                      78       22364
rfc7515                      81       29613
rfc7516                      93       24203
rfc7517                      56       29636
rfc7518                     113       31104
rfc7519                      50       12026
rfc8725                      31        5431
oidc-core                   185       58940
oidc-discovery               33       13342
oidc-registration            34       13908
oauth-2.1                   172       41170
TOTAL                      1366      396505
```

</details>

## CLI reference

Run every command as `uv run specrag <command>`. `specrag --version` prints the version.

| Command | What it does | Options |
|---|---|---|
| `fetch` | Downloads every manifest document into the cache and verifies its SHA-256. Files already cached are skipped. | `--pin` writes the computed hashes into the manifest; use it only for a deliberate re-pin. `--manifest PATH` reads another manifest. |
| `parse` | Parses every cached document into `sections.jsonl` and prints a table per document. | `--manifest PATH` |
| `sections DOC_ID` | Prints one document's section tree (ID, title and own-text token count), indented by depth. | none |
| `index` | Builds the SQLite chunk store for a chunker, and the embedding index with `--dense`. Prints chunk counts and timings. | `--chunker fixed\|section\|section+meta\|all` (default `all`). `--dense` also prints `dense truncated: N`. |
| `search QUERY` | Ranks chunks and packs them into the token budget. Prints each chunk's rank, ID, sections and cost, then `tokens used` and the config hash. | `--chunker fixed\|section\|section+meta` (default `section`), `--retriever bm25\|dense\|hybrid` (default `bm25`), `--rerank`, `--budget N` (default 2500). `--json` prints the full `RetrievalResult`, including the whole ranking and provenance. |

**Exit codes**

| Code | Meaning | Examples |
|---|---|---|
| 0 | Success | |
| 2 | Bad input | `query must not be empty`, `unknown doc_id: nope`, an invalid option value |
| 3 | Corpus or model unavailable | a fetch failure or hash drift (`corpus unavailable: not caused by this change`); `parse`, `sections`, `index` or `search` before the corpus is fetched and parsed; `model unavailable: <model>@<revision>` when a model can't be loaded |

Exit code 3 is kept separate on purpose. CI retries a fetch that exits with 3 and reports it as an infrastructure failure, never as a test failure.

## The corpus

The corpus is 23 documents, each pinned by URL and SHA-256 in [`corpus/manifest.yaml`](corpus/manifest.yaml).

| doc_id | Title | Status |
|---|---|---|
| `rfc6749` | The OAuth 2.0 Authorization Framework | Standards Track |
| `rfc6750` | The OAuth 2.0 Authorization Framework: Bearer Token Usage | Standards Track |
| `rfc7636` | Proof Key for Code Exchange by OAuth Public Clients | Standards Track |
| `rfc7009` | OAuth 2.0 Token Revocation | Standards Track |
| `rfc7662` | OAuth 2.0 Token Introspection | Standards Track |
| `rfc8414` | OAuth 2.0 Authorization Server Metadata | Standards Track |
| `rfc8252` | OAuth 2.0 for Native Apps | BCP |
| `rfc8628` | OAuth 2.0 Device Authorization Grant | Standards Track |
| `rfc9126` | OAuth 2.0 Pushed Authorization Requests | Standards Track |
| `rfc9449` | OAuth 2.0 Demonstrating Proof of Possession (DPoP) | Standards Track |
| `rfc8705` | OAuth 2.0 Mutual-TLS Client Authentication and Certificate-Bound Access Tokens | Standards Track |
| `rfc9068` | JSON Web Token (JWT) Profile for OAuth 2.0 Access Tokens | Standards Track |
| `rfc9700` | Best Current Practice for OAuth 2.0 Security | BCP |
| `rfc7515` | JSON Web Signature (JWS) | Standards Track |
| `rfc7516` | JSON Web Encryption (JWE) | Standards Track |
| `rfc7517` | JSON Web Key (JWK) | Standards Track |
| `rfc7518` | JSON Web Algorithms (JWA) | Standards Track |
| `rfc7519` | JSON Web Token (JWT) | Standards Track |
| `rfc8725` | JSON Web Token Best Current Practices | BCP |
| `oidc-core` | OpenID Connect Core 1.0 | OpenID Final |
| `oidc-discovery` | OpenID Connect Discovery 1.0 | OpenID Final |
| `oidc-registration` | OpenID Connect Dynamic Client Registration 1.0 | OpenID Final |
| `oauth-2.1` | The OAuth 2.1 Authorization Framework (draft-ietf-oauth-v2-1-17) | Draft |

The manifest also records "Updates" and "Obsoletes" relations; for example, RFC 9700 updates RFC 6749, 6750 and 6819. The planned generator will use these to rank authority when sources disagree.

- **Pinned by hash.** `fetch` checks every file's SHA-256. If a hash has drifted, it fails with exit code 3 and a clear message. Re-pinning (`fetch --pin`) is a deliberate manifest PR, never automatic.
- **Never committed.** Spec text lives in `~/.cache/specrag/` (or `$SPECRAG_CACHE`), outside git. The repo commits only structural facts for the snapshot tests: section counts, first and last IDs, and sample titles.
- **Weekly drift check.** The `Corpus` workflow runs every Monday and on demand. It fetches into a fresh directory, so it always goes to upstream, and verifies every hash. It then refreshes the Actions cache. If anything fails, it opens or updates a "Corpus re-pin needed" issue.
- **Cloudflare beacon normalisation.** Some pages are served with a per-request Cloudflare `<script>` block (`window.__CF$cv$params={r:...,t:...}`), so the same document hashed differently on every fetch. Before hashing and storing, `fetch` removes the beacon in a single linear pass anchored on the `__CF$cv$params` marker: a block is cut only when a well-formed `<script ...>` tag, the marker and a `</script>` closer appear in that order with no other script boundary between them, and otherwise the data is left unchanged. Every other byte is untouched (PR #11, hardened in #25 for issue #24).

## How it works

```text
                    corpus/manifest.yaml (URL + sha256)
                                 │
  specrag fetch   ──►  ~/.cache/specrag/raw/<doc_id>.html   (normalised, hash-verified)
                                 │
  specrag parse   ──►  detect dialect ─► v3 | htmlized | OpenID parser
                                 │
                       sections.jsonl  (id, title, parent, depth, own_text, own_tokens)
                                 │
  specrag index   ──►  per-doc token stream ─► fixed | section | section+meta chunks
                                 │                 │
                        chunks.sqlite         emb-<key>.npy  (dense, optional)
                                 │
  specrag search  ──►  BM25 | dense | RRF hybrid  ─► optional cross-encoder rerank (top 50)
                                 │
                       budget packer (2,500 tokens, last chunk truncated, overlap free)
                                 │
                       RetrievalResult: full ranking + packed chunks + provenance
                                 │
               ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
                 planned: cited generation (Ollama) ─► eval harness ─► report
```

### Parsing three HTML dialects

The corpus uses HTML renderings, not plain text or XML. Every RFC has an rfc-editor `.html` rendering with section anchors, while RFCXML v3 exists only for newer RFCs. In practice there are three dialects, and the parser detects which one from the markup:

| Dialect | Documents | Section markers |
|---|---|---|
| xml2rfc v3 | RFC 8705, 8725, 9068, 9126, 9449, 9700 and the OAuth 2.1 draft | `<section id="section-N">` or `appendix-X`, with `a.section-number` |
| htmlized plain text | the 13 older RFCs | a `<pre>` body with `span.h2` to `h6` headings and `a.selflink[id^=section-]` |
| OpenID xml2rfc | the three OpenID specs | `rfc.section.N` anchors |

All three feed one shared validator. On a duplicate or missing section number it fails loudly, naming the document and the section, rather than silently merging. Reference sections are flagged with `kind: "references"`. Snapshot tests for each document pin its section count, first and last IDs and sample titles, which were checked by hand against each spec's table of contents. Paragraph breaks and preformatted layout (ASCII diagrams, tables and code) are kept in the text (PR #12).

### Section IDs and own text

- IDs take the form `<doc>#<number>`, for example `rfc6749#4.1.2`. Appendices look like `rfc7518#A.1`.
- A section's text is its **own text only**, without its child subsections. In the sample above, `rfc7636#4  Protocol  (0)` is a heading-only parent: all of its content lives in its subsections.
- A section's parent comes from its number, not from the HTML heading level, so a jump in heading level can't misplace it.

### Three chunkers

Each document becomes one cl100k token stream in which every token knows which section owns it. A chunk is a `[start, end)` span of that stream. Its `sections[]` and `tokens[]` (how many tokens come from each section) are therefore computed exactly, not estimated.

| Chunker | Rule | Chunks |
|---|---|---|
| `fixed` | 512-token windows with 64 tokens of overlap, ignoring structure (the baseline) | 808 |
| `section` | One section's own text per chunk. A section over 800 tokens is split into balanced pieces (at a paragraph, else a sentence, else a hard cut) that keep the same section ID. | 1,365 |
| `section+meta` | `section` chunks with a document label and heading path prepended to the indexed text, e.g. `RFC 9700 · OAuth 2.0 Security BCP › 2 Best Practices › 2.1.1 Authorization Code Grant` | 1,365 |

The `section+meta` prefix is ordinary metadata prepending. It is **not** Anthropic's Contextual Retrieval, which uses LLM-written context; that's a stretch goal. Only content sections with their own text are chunked: the 67 reference sections (40,118 tokens) and the 53 empty (zero-token) sections are parsed but never chunked or indexed, so 1,246 of the 1,366 sections reach the chunkers, and splitting long sections brings `section` to 1,365 chunks (the near-match with 1,366 is a coincidence). The `parse` total of 396,505 tokens includes references; content alone is 356,387. Chunk boundaries can move by up to 3 tokens (`safe_cut`) so that no cut splits a multi-byte character.

### The 2,500-token budget

Retrieval depth is a **token budget, not a fixed k**. That way, chunkers with very different chunk sizes are compared on the same amount of context.

- Ranked chunks are packed until 2,500 tokens are used (minus at most 3 when a cut is moved to a character boundary), unless the ranking runs out first.
- **The last chunk is truncated to fit**, not dropped, and its `sections[]` and `tokens[]` are recomputed for the text that's kept. At most one chunk is truncated, and packing stops there.
- **Overlap is free.** Tokens already packed from the same document, such as the 64-token overlap between `fixed` windows, don't count again. So `fixed` isn't penalised for repeating text.
- **The `section+meta` prefix counts** toward the budget. A chunk whose prefix no longer fits is skipped.

### Retrievers

| Retriever | Implementation |
|---|---|
| BM25 | Hand-built Okapi BM25 (k1 = 1.2, b = 0.75, Lucene IDF). Text is lowercased, identifiers like `code_verifier` and `client-id` stay whole, and there's no stemming or stopword list. Ties break deterministically (PR #23). |
| Dense | `BAAI/bge-small-en-v1.5` (384 dimensions) through sentence-transformers, **always on CPU**, L2-normalised, with a limit of 512 model tokens. |
| Hybrid | Reciprocal rank fusion of BM25 and dense: score = Σ 1/(60 + rank). |
| Rerank (optional) | The `BAAI/bge-reranker-base` cross-encoder on CPU, reordering the top 50 of any ranking. |

Both models are pinned to exact Hugging Face revisions in [`models_pin.py`](src/specrag/retrieval/models_pin.py). They're loaded from the pinned snapshot path, and loading fails closed if a local directory would shadow the model ID (CWE-426, PR #22). Storage is numpy plus SQLite; there's no vector database.

### Caching and staleness keys

- **Corpus key:** `sha256(manifest.yaml + sections.jsonl)[:12]`. Re-pinning or re-parsing changes it.
- **Index directory:** `index/<chunker>-v<CHUNKER_VERSION>-<corpus key>/chunks.sqlite`, written atomically. A changed corpus or chunker version gives a new directory and a rebuild, never silent reuse.
- **Embeddings:** `emb-<sha256(corpus key | chunker | chunker version | model | revision)[:16]>.npy`. A sidecar file stores the chunk IDs and the SHA-256 of every indexed text, and any mismatch triggers re-embedding.

### Provenance

Every `RetrievalResult` (see `search --json`) carries:

- the query and the full config;
- the **full ranking**, not just the packed chunks;
- the packed chunks and their costs;
- a provenance block with the corpus key, chunker version, config hash, the dense model and revision (or `n/a`), and the dense truncation count. With `--rerank` it also records the reranker revision and depth.

The harness will build its per-run provenance (spec §7.6) on top of this.

## Evaluation design (planned)

None of this is built yet; the [design doc](docs/superpowers/specs/2026-11-21-specrag-design.md) is the full specification.

- **Golden set (§7.1):** 145 questions (single-section, multi-section, conflict, unanswerable, adversarial, messy), gold labels completed by TREC-style pooling and human confirmation, split dev/test **by cluster** so no gold section appears in both splits.
- **Hit rule (§7.2), already implemented** in [`scoring.py`](src/specrag/scoring.py): a chunk hits gold section `g` only if at least min(50 tokens, 30% of `g`'s own text) comes from `g`; descendants count only for heading-only parents.
- **Metrics (§7.3):** deterministic retrieval and answer metrics are the headline; judge scores are secondary and shown only if calibration against blind human labels passes (κ lower bound ≥ 0.4).
- **Pre-registered comparisons (§7.4), Holm-corrected:** P1 `section` vs `fixed`, P2 hybrid vs the better single retriever, P3 best RAG vs closed-book `llama3.1:8b`. Paired cluster-bootstrap CIs; nulls reported as "inconclusive", never "no difference".
- **Release protocol (§7.5):** test is evaluated only by `specrag release`, every run is logged, primary results come from the first release, and CI gates retrieval on dev only.

<details>
<summary>Golden-set mix and comparison metrics</summary>

| Type | n |
|---|---|
| single-section fact | 50 |
| multi-section / cross-spec | 30 |
| conflict / supersession | 20 |
| unanswerable | 20 |
| adversarial (prompt injection) | 10 |
| messy style | 15 |

| | Comparison | Metric |
|---|---|---|
| P1 | `section` vs `fixed` chunking (BM25) | recall within the 2,500-token budget |
| P2 | hybrid vs the better of BM25 and dense, chosen on dev | recall within the budget |
| P3 | best RAG config vs closed-book | `must_include` coverage, checked against human fact labels on a 25-question subset |

</details>

## Engineering process

- **Sub-agent implementation and review.** A fresh sub-agent implements each plan task, and a fresh reviewer sub-agent checks it for spec compliance and code quality, with fix rounds until it's clean. The main session then reviews each change against the whole design before it lands. The design and plans went through review rounds too; the spec is at v7.
- **One squashed commit per change**, landed through a PR from its own branch. Every PR uses the [5-section template](.github/pull_request_template.md): Description, Risk level, Assessment, CI run and Verification. A PR merges only on green CI.
- **CI gates**, in [`.github/workflows/`](.github/workflows/):

| Gate | Where |
|---|---|
| `ruff check`, `ruff format --check`, `mypy` (strict), `pytest` | `CI` / `test` |
| Tests on the cached corpus with **0 corpus skips allowed** and coverage **≥ 90%**; dense-model tests (`SPECRAG_MODELS=1`); `parse` re-run on every commit | `CI` / `corpus-tests` |
| `actionlint` on the workflows | `CI` / `actionlint` |
| CodeQL for Python, on PRs, on `main` and weekly | `CodeQL` |
| Weekly fetch and hash check, opening a re-pin issue on failure | `Corpus` |
| Dependabot for GitHub Actions and uv, weekly and grouped; actions are pinned by commit SHA | [`dependabot.yml`](.github/dependabot.yml) |
| pre-commit: trailing whitespace, end of file, YAML, merge conflicts, ruff | [`.pre-commit-config.yaml`](.pre-commit-config.yaml) |

**Notable fixes from review**, from the merged history:

- **#9**, the Plan 1 final review: a `references` section kind, a STOP guard, explicit parser dispatch and pinned snapshots.
- **#11**: Cloudflare's per-request beacon made the pinned hashes unstable. It's now stripped before hashing, and `oauth-2.1` was re-pinned.
- **#12**: `own_text` was losing paragraph breaks and preformatted layout.
- **#22**: models now load from pinned snapshot paths, so a local directory can't shadow a pinned model ID.
- **#23**: deterministic BM25 ordering, end-to-end scoring integration tests, and hardening of the cache and the shadow-ID check.
- **#25** (fixes issue #24): Cloudflare beacon removal is anchored on the marker and done in a single forward pass.

## Known limitations

- **Dense truncates most `fixed` chunks.** `fixed` windows are 512 cl100k tokens, but bge-small's limit is 512 WordPiece tokens, and WordPiece splits spec text more finely. In the run above, 666 of the 808 `fixed` chunks (82%) were truncated, along with 257 of 1,365 `section` chunks and 287 of 1,365 `section+meta` chunks. `index --dense` reports the count and provenance records it. It's a real handicap for dense on `fixed`, and any P2 result has to be read with it in mind.
- **Small CPU-only models.** `bge-small-en-v1.5` was chosen to run on a laptop and a free CI runner, not for peak quality.
- **English only**, and only the 23 pinned documents. There's no live web search.
- **The planned features don't exist yet.** There's no generation, no golden set, no metric over real questions and no results table.
- **A known bias in the planned golden set.** A model drafts the questions while reading the specs, so their wording echoes the source text, which may favour BM25. Results will be broken down by `style` for this reason.
- **Out of scope for v1** (spec §3): a web UI, deployment or an API server, agents or multi-turn chat, and fine-tuning. Stretch goals come only after v1 ships: a second generator, query rewriting, LLM-written contextual retrieval, Claude provider rows, an HTML report, a LlamaIndex baseline and a sampling-variance study.
- **Not comparable to BEIR or MTEB.** The corpus is different, so scores are only for comparisons within this project.

## Repo layout

```text
specrag/
├── corpus/manifest.yaml          # 23 pinned documents (URL, sha256, status, relations)
├── docs/superpowers/
│   ├── specs/                    # the design (v7), the binding authority
│   └── plans/                    # Plan 1 (foundation) and Plan 2 (retrieval)
├── scripts/embed_report.py       # local only: embed every chunker, report truncation and time
├── src/specrag/
│   ├── cli.py                    # typer CLI: fetch, parse, sections, index, search
│   ├── models.py                 # Doc, Section, Chunk, SectionIndex (pydantic)
│   ├── sections.py               # section ID helpers
│   ├── scoring.py                # hit rule, recall, MRR, nDCG, hit-rate precision
│   ├── tokens.py                 # cl100k token counter
│   ├── corpus/                   # manifest, fetch, dialect detection, three parsers, build
│   └── retrieval/                # streams, chunkers, budget, bm25, dense, hybrid, rerank,
│                                 #   store, pipeline, models_pin
├── tests/                        # unit, snapshot, corpus, model and integration tests
├── .github/                      # CI, CodeQL and Corpus workflows; dependabot; PR template
├── CLAUDE.md                     # working rules for the agents that build this repo
├── LICENSE
└── pyproject.toml
```

## Development

```bash
uv sync --group dev
uv run pre-commit install         # once: ruff and hygiene hooks run on commit

uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest -q
```

Tests never fetch the corpus over the network. Marked tests are skipped unless what they need is present:

| Marker | Runs when |
|---|---|
| `corpus` | the corpus has been fetched (`specrag fetch`) |
| `model` | `SPECRAG_MODELS=1` (dense model) |
| `reranker` | `SPECRAG_RERANKER=1` (reranker model) |

```bash
SPECRAG_MODELS=1 uv run pytest -rs                      # corpus and dense tests, with skip reasons
SPECRAG_MODELS=1 uv run python scripts/embed_report.py  # local truncation and timing report
```

Add `SPECRAG_RERANKER=1` to run the reranker tests too; they download the ~1.1 GB reranker.

**Contributing rules**, from [`CLAUDE.md`](CLAUDE.md):

- The [design spec](docs/superpowers/specs/2026-11-21-specrag-design.md) is the binding authority.
- Make one squashed commit per change, through a PR that uses the template, and merge only on green CI.
- Never commit spec text.
- A PR that touches `src/specrag/corpus/` or `src/specrag/models.py` must paste local `uv run pytest -rs` output showing 0 skipped corpus tests.

## License

The code is [MIT](LICENSE)-licensed. The specifications are fetched from rfc-editor.org, datatracker.ietf.org and openid.net when you run `fetch`, and are never committed to this repository. They remain under their publishers' terms.
