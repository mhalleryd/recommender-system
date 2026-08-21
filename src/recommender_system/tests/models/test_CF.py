import pytest

from recommender_system.models.collaborative_filtering import (
    get_item_similarity_matrix,
    get_similar_items,
)


def test_get_item_similarity_matrix(sample_reviews):
    similarities, item_to_idx = get_item_similarity_matrix(sample_reviews)

    assert similarities.shape == (3, 3)
    assert set(item_to_idx) == {"A", "B", "C"}

def test_similarity_matrix_is_symmetric(sample_reviews):
    similarities, _ = get_item_similarity_matrix(sample_reviews)

    assert (similarities != similarities.T).nnz == 0

def test_items_are_similar_to_themselves(sample_reviews):
    similarities, item_to_idx = get_item_similarity_matrix(sample_reviews)

    for idx in item_to_idx.values():
        assert similarities[idx, idx] == pytest.approx(1.0)

def test_get_similar_items_returns_expected_columns(sample_reviews):
    similarities, item_to_idx = get_item_similarity_matrix(sample_reviews)

    result = get_similar_items(
        "A",
        similarities,
        item_to_idx,
        k=2,
    )

    assert list(result.columns) == ["item_id", "scores"]
    assert len(result) == 2

def test_get_similar_items_excludes_input_item(sample_reviews):
    similarities, item_to_idx = get_item_similarity_matrix(sample_reviews)

    result = get_similar_items(
        "A",
        similarities,
        item_to_idx,
        k=2,
    )

    assert "A" not in result["item_id"].values