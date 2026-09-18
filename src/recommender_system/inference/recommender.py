import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import scipy.sparse as sp


class Recommender:
    """Recommender system based on matrix factorization.

    Attributes
    ----------
        U : np.ndarray
            User latent factors matrix.
        V : np.ndarray
            Item latent factors matrix.
        user_to_idx : dict[str, int]
            Mapping from user IDs to indices.
        item_to_idx : dict[str, int]
            Mapping from item IDs to indices.
        X : scipy.sparse.csr_matrix
            Interaction matrix.
        idx_to_item : dict[int, str]
            Mapping from item indices to item IDs.
        popular_items : list[str]
            List of popular items for cold start recommendations."""
    
    def __init__(
        self,
        U: np.ndarray,
        V: np.ndarray,
        user_to_idx: dict[str, int],
        item_to_idx: dict[str, int],
        X: sp.csr_matrix,
        popular_items: list[str],
    ):
        self.U = U
        self.V = V
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        self.idx_to_item = {
            idx: item for item, idx in item_to_idx.items()
        }

        self.X = X
        self.popular_items = popular_items

    def recommend(self, user_id: str, k: int = 10) -> list[str]:
        """Recommend the top-k items for a given user.

        Parameters
        ----------
        user_id : str
            The ID of the user to recommend items for.
        k : int
            The number of items to recommend.

        Returns
        -------
        recommended_items : list
            A list of item IDs recommended for the user."""

        if k <= 0:
            raise ValueError("k must be greater than 0")

        # Ensure k does not exceed the number of items
        k = min(k, self.V.shape[0])

        if user_id not in self.user_to_idx:
            return self.popular(k)

        user_idx = self.user_to_idx[user_id]
        scores = self.U[user_idx] @ self.V.T

        # Remove already seen items
        seen_indices = self.X[user_idx].indices
        scores[seen_indices] = -np.inf

        # Ensure k does not exceed the number of items not seen by the user
        available = self.V.shape[0] - len(seen_indices)
        k = min(k, available)

        if k == 0:
            return []

        # Efficient top-k
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        recommended_items = [
            self.idx_to_item[idx] for idx in top_indices
        ]

        return recommended_items

    def popular(self, k: int = 10) -> list[str]:
        """Return the top-k popular items."""
        if k <= 0:
            raise ValueError("k must be greater than 0")
        
        return self.popular_items[:k]

    @classmethod
    def load(cls, artifact_dir: str | Path) -> "Recommender":
        """Load a Recommender instance from a file.

        Parameters
        ----------
        artifact_path : str
            Path to the file containing the saved Recommender instance.

        Returns
        -------
        recommender : Recommender
            The loaded Recommender instance."""
        artifact_dir = Path(artifact_dir)
        
        model = np.load(artifact_dir / "model.npz")

        # Load and cast interaction matrix
        loaded_interactions = cast(
            Any,
            sp.load_npz(artifact_dir / "interactions.npz"),
        )
        X = cast(sp.csr_matrix, loaded_interactions.tocsr())

        # Load mappings and popular items
        with open(artifact_dir / "user_to_idx.json", "r") as f:
            user_to_idx = json.load(f)

        with open(artifact_dir / "item_to_idx.json", "r") as f:
            item_to_idx = json.load(f)

        with open(artifact_dir / "popular_items.json", "r") as f:
            popular_items = json.load(f)

        return cls(
            U=model["U"],
            V=model["V"],
            user_to_idx=user_to_idx,
            item_to_idx=item_to_idx,
            X=X,
            popular_items=popular_items,
        )