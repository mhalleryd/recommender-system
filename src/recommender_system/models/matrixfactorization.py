import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


class MatrixFactorization:
    def __init__(self, n_factors=10, random_state=42):
        self.n_factors = n_factors
        self.random_state = random_state

    def fit(
        self,
        X: csr_matrix,
        n_epochs: int = 10,
        learning_rate: float = 0.01,
        regularization: float = 0.01,
    ):
        n_users, n_items = X.shape

        rng = np.random.default_rng(self.random_state)

        self.U = rng.normal(
            0, 0.1, size=(n_users, self.n_factors)
        )
        self.V = rng.normal(
            0, 0.1, size=(n_items, self.n_factors)
        )

        X_ = X.tocoo()

        for epoch in range(n_epochs):
            squared_error = 0.0

            for user, item, rating in zip(X_.row, X_.col, X_.data):
                prediction = self.U[user] @ self.V[item]
                error = rating - prediction

                squared_error += error**2

                user_factors = self.U[user].copy()

                self.U[user] += learning_rate * (
                    error * self.V[item]
                    - regularization * self.U[user]
                )

                self.V[item] += learning_rate * (
                    error * user_factors
                    - regularization * self.V[item]
                )

            print(
                f"Epoch {epoch + 1}/{n_epochs}, "
                f"RMSE: {np.sqrt(squared_error / X_.nnz):.4f}"
            )

        return self

    def predict_for_user(self, user_idx):
        return self.U[user_idx] @ self.V.T




def recommend_mf(
    model,
    test: pd.DataFrame,
    train: pd.DataFrame,
    user_to_idx: dict,
    idx_to_item: dict,
    n: int = 3,
) -> pd.DataFrame:

    # Convert training interactions to integer indices once
    seen_indices = (
        train.assign(
            user_idx=train["user_id"].map(user_to_idx),
            item_idx=train["item_id"].map(
                {v: k for k, v in idx_to_item.items()}
            ),
        )
        .groupby("user_idx")["item_idx"]
        .apply(np.array)
        .to_dict()
    )

    recommendations : list[dict[str, object]] = []

    for user_id in test["user_id"].unique():
        user_idx = user_to_idx[user_id]

        scores = model.predict_for_user(user_idx).copy()

        # Remove already seen items
        scores[seen_indices.get(user_idx, [])] = -np.inf

        # Efficient top-k
        top_indices = np.argpartition(scores, -n)[-n:]
        top_indices = top_indices[
            np.argsort(scores[top_indices])[::-1]
        ]

        recommendations.extend(
            {
                "user_id": user_id,
                "item_id": idx_to_item[idx],
                "score": scores[idx],
            }
            for idx in top_indices
        )

    return pd.DataFrame(recommendations)
