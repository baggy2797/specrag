import pytest

from specrag.retrieval.models_pin import DENSE_MODEL, DENSE_REVISION


@pytest.mark.model
def test_dense_model_loads_on_cpu() -> None:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(DENSE_MODEL, revision=DENSE_REVISION, device="cpu")
    assert model.device.type == "cpu"
    assert model.get_embedding_dimension() == 384
