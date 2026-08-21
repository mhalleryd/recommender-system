import pandas as pd
import pytest


@pytest.fixture
def sample_reviews():
    return pd.DataFrame({
        "user_id": ["u1", "u1", "u2", "u2", "u3"],
        "item_id": ["A", "B", "A", "C", "B"],
        "recommend": [1, 1, 1, 1, 1],
    })