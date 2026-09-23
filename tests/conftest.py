import json

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix, save_npz

from recommender_system.inference.recommender import Recommender


@pytest.fixture
def sample_reviews():
    return pd.DataFrame({
        "user_id": ["u1", "u1", "u2", "u2", "u3"],
        "item_id": ["A", "B", "A", "C", "B"],
        "recommend": [1, 1, 1, 1, 1],
    })


@pytest.fixture
def artifact_dir(tmp_path):
    """Create a small recommender artifact for testing."""

    U = np.array([[1.0]])
    V = np.array([
        [1.0],
        [0.8],
        [0.5],
        [0.2],
    ])

    X = csr_matrix([[1, 0, 0, 0]])

    user_to_idx = {
        "user-1": 0,
    }

    item_to_idx = {
        "item-a": 0,
        "item-b": 1,
        "item-c": 2,
        "item-d": 3,
    }

    popular_items = [
        "item-d",
        "item-c",
        "item-b",
        "item-a",
    ]

    item_metadata = {
            "item-a": {
                "app_name": "Item A",
                "developer": "Developer A",
                "genres": ["Action"],
            },
            "item-b": {
                "app_name": "Item B",
                "developer": "Developer B",
                "genres": ["Adventure"],
            },
            "item-c": {
                "app_name": "Item C",
                "developer": "Developer C",
                "genres": ["RPG"],
            },
            "item-d": {
                "app_name": "Item D",
                "developer": "Developer D",
                "genres": ["Strategy"],
            },
        }

    # Save model
    np.savez(
        tmp_path / "model.npz",
        U=U,
        V=V,
    )

    # Save interactions
    save_npz(
        tmp_path / "interactions.npz",
        X,
    )

    # Save metadata
    artifacts = {
        "user_to_idx.json": user_to_idx,
        "item_to_idx.json": item_to_idx,
        "popular_items.json": popular_items,
        "item_metadata.json": item_metadata,
    }

    for filename, data in artifacts.items():
        with open(tmp_path / filename, "w") as f:
            json.dump(data, f)

    return tmp_path


@pytest.fixture
def recommender(artifact_dir):
    return Recommender.load(artifact_dir)