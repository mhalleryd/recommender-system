import pytest


def test_known_user_recommendations_are_ordered(recommender):
    assert recommender.recommend("user-1", k=3) == [
        "item-b",
        "item-c",
        "item-d",
    ]


def test_unknown_user_uses_popularity_fallback(recommender):
    assert recommender.recommend("unknown-user", k=2) == [
        "item-d",
        "item-c",
    ]


def test_seen_items_are_never_recommended(recommender):
    recommendations = recommender.recommend("user-1", k=10)

    assert "item-a" not in recommendations


@pytest.mark.parametrize("k", [0, -1])
def test_recommend_rejects_non_positive_k(recommender, k):
    with pytest.raises(ValueError, match="k must be greater than 0"):
        recommender.recommend("user-1", k=k)


def test_k_larger_than_available_items_is_limited(recommender):
    recommendations = recommender.recommend("user-1", k=100)

    # There are four items, but item-a has already been seen.
    assert len(recommendations) == 3
    assert recommendations == ["item-b", "item-c", "item-d"]