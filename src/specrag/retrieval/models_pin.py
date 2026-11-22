"""Pinned Hugging Face models. The CI model cache key hashes this file."""

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
DENSE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
RERANK_MODEL = "BAAI/bge-reranker-base"
RERANK_REVISION = "2cfc18c9415c912f9d8155881c133215df768a70"

# Only what sentence-transformers needs on CPU: configs, tokenizer files, modules, safetensors.
# Excludes pytorch_model.bin, onnx/ and openvino/ weights.
ALLOW_PATTERNS = ["*.json", "*.txt", "*.model", "*.safetensors"]


def resolve_pinned(model_id: str, revision: str) -> str:
    """Absolute snapshot path of the pinned revision (CWE-426).

    Passing a bare model id to SentenceTransformer/CrossEncoder loads a local directory of that
    name if one exists, bypassing the pin, so fail closed and load from the snapshot path.
    Respects HF_HUB_OFFLINE=1 (snapshot_download then uses the cache).
    """
    import os

    if os.path.exists(model_id):
        raise ValueError(f"local path shadows pinned model id: {model_id}")
    from huggingface_hub import snapshot_download

    return str(snapshot_download(model_id, revision=revision, allow_patterns=ALLOW_PATTERNS))
