import numpy as np
import pytest
from scipy.sparse import csr_matrix

from recommender_system.models.matrixfactorization import (
    MatrixFactorization,
    recommend_mf,
)


@pytest.fixture
def ratings():
    # 1 means the user recommended the item; 0 means no observed interaction.
    return csr_matrix(
        [
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
        ]
    )


@pytest.fixture
def trained_model(ratings):
    model = MatrixFactorization(n_factors=2, random_state=42)
    model.fit(ratings, n_epochs=2)
    return model


def test_matrix_factorization_fit_initializes_factors(ratings):
    model = MatrixFactorization(n_factors=2, random_state=42)

    result = model.fit(ratings, n_epochs=2)

    assert result is model
    assert model.U.shape == (3, 2)
    assert model.V.shape == (3, 2)


def test_predict_for_user_returns_scores_for_all_items(trained_model):
    scores = trained_model.predict_for_user(0)

    assert scores.shape == (3,)
    assert np.isfinite(scores).all()


def test_matrix_factorization_is_reproducible(ratings):
    model_a = MatrixFactorization(n_factors=2, random_state=42)
    model_b = MatrixFactorization(n_factors=2, random_state=42)

    model_a.fit(ratings, n_epochs=2)
    model_b.fit(ratings, n_epochs=2)

    np.testing.assert_allclose(model_a.U, model_b.U)
    np.testing.assert_allclose(model_a.V, model_b.V)


def test_recommend_mf_returns_unseen_items(trained_model, sample_reviews):

    result = recommend_mf(
        model=trained_model,
        test=sample_reviews[["user_id"]],
        train=sample_reviews,
        user_to_idx={"u1": 0, "u2": 1, "u3": 2},
        idx_to_item={0: "A", 1: "B", 2: "C"},
        n=1,
    )

    assert list(result.columns) == ["user_id", "item_id", "score"]
    assert len(result) == 3
    assert result["score"].notna().all()

    seen_by_user = sample_reviews.groupby("user_id")["item_id"].apply(set)

    for user_id, item_id in zip(result["user_id"], result["item_id"]):
        assert item_id not in seen_by_user[user_id]