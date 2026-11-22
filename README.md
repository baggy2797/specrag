# specrag

Cited question answering over the public OAuth 2.0 / OpenID Connect / JWT standards, built around a rigorous evaluation harness.

**Status:** design approved, build starting. See the [design doc](docs/superpowers/specs/2026-11-21-specrag-design.md).

## Search

Build the chunk stores once, then query:

```bash
uv run specrag index --chunker all
uv run specrag search "PKCE code_verifier" --retriever hybrid
```

`specrag index --chunker all --dense` also builds the dense embedding index and reports `dense truncated: N`.

Each result packs ranked chunks into a 2,500-token budget (`--budget` to change it); `--json` prints the full retrieval result. The first dense or rerank run downloads the pinned models (~130 MB dense, ~1.1 GB reranker).
