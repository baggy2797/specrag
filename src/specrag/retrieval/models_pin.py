"""Pinned Hugging Face models. The CI model cache key hashes this file."""

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
DENSE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
RERANK_MODEL = "BAAI/bge-reranker-base"
RERANK_REVISION = "2cfc18c9415c912f9d8155881c133215df768a70"

# Only what sentence-transformers needs on CPU: configs, tokenizer files, modules, safetensors.
# Excludes pytorch_model.bin, onnx/ and openvino/ weights.
ALLOW_PATTERNS = ["*.json", "*.txt", "*.model", "*.safetensors"]

# Files each pinned model needs to load on CPU (checked against the real cached snapshots; optional
# files such as README.md, vocab.txt, special_tokens_map.json are not listed). Each inner tuple is
# an any-of group. A cached snapshot missing any group is treated as absent (network fallback),
# because snapshot_download(local_files_only=True) returns a partial snapshots/<sha> as-is when no
# cached file listing exists.
REQUIRED_FILES: dict[str, tuple[tuple[str, ...], ...]] = {
    DENSE_MODEL: (
        ("modules.json",),
        ("config.json",),
        ("model.safetensors",),
        ("tokenizer.json",),
        ("tokenizer_config.json",),
        ("sentence_bert_config.json",),
        ("1_Pooling/config.json",),
    ),
    RERANK_MODEL: (
        ("config.json",),
        ("model.safetensors",),
        ("tokenizer.json", "sentencepiece.bpe.model"),
        ("tokenizer_config.json",),
    ),
}


class ShadowedModelError(ValueError):
    """A local path would shadow the pinned model id (CWE-426)."""


def _is_complete(model_id: str, snapshot: str) -> bool:
    from pathlib import Path

    groups = REQUIRED_FILES.get(model_id)
    if groups is None:  # no manifest: cannot vouch for the cache
        return False
    root = Path(snapshot)
    return all(any((root / f).is_file() for f in group) for group in groups)


def resolve_pinned(model_id: str, revision: str) -> str:
    """Absolute snapshot path of the pinned revision (CWE-426).

    Passing a bare model id to SentenceTransformer/CrossEncoder loads a local directory of that
    name if one exists, bypassing the pin, so fail closed and load from the snapshot path.

    Cache first: snapshot_download(local_files_only=True) never contacts the Hub, so a warm cache
    loads without network round-trips and without HF_HUB_OFFLINE=1. This is pin-safe: the cache
    stores each snapshot under snapshots/<commit sha>, and with a full sha as `revision`
    huggingface_hub returns exactly that directory (no ref or branch lookup), so it can only be
    the pinned commit; a non-sha revision is rejected with ValueError. huggingface_hub returns a
    partial snapshot as-is when no cached file listing exists, so the local path is trusted only
    if every file in REQUIRED_FILES is present. Otherwise (LocalEntryNotFoundError, or an
    incomplete or unknown-model cache) fall back to the network call with the same pinned
    revision and allow_patterns.
    """
    import os
    import re

    # Absolute and explicit-relative IDs can never be Hub IDs; a bare "org/name" is also a local
    # path if it exists relative to the cwd (that is how SentenceTransformer would resolve it).
    if os.path.isabs(model_id) or model_id.startswith((".", "~")) or os.path.exists(model_id):
        raise ShadowedModelError(f"local path shadows pinned model id: {model_id}")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"revision must be a full 40-hex commit sha: {revision!r}")
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        local = str(
            snapshot_download(
                model_id, revision=revision, allow_patterns=ALLOW_PATTERNS, local_files_only=True
            )
        )
        if _is_complete(model_id, local):
            return local
    except LocalEntryNotFoundError:
        pass
    return str(snapshot_download(model_id, revision=revision, allow_patterns=ALLOW_PATTERNS))
