import pandas as pd

from recommender_system.data.transform import transform_reviews


def test_transform_reviews():
    df = pd.DataFrame([
        {
            "user_id": "user_1",
            "reviews": [
                {
                    "item_id": "123",
                    "recommend": True,
                    "review": "Great game!",
                    "funny": "",
                    "last_edited": "",
                    "helpful": "No ratings yet",
                },
                {
                    "item_id": "456",
                    "recommend": False,
                    "review": "Not very good.",
                    "funny": "",
                    "last_edited": "",
                    "helpful": "No ratings yet",
                },
            ],
        }
    ])

    result = transform_reviews(df)

    assert len(result) == 2
    assert list(result["user_id"]) == ["user_1", "user_1"]
    assert list(result["item_id"]) == ["123", "456"]
    assert list(result["recommend"]) == [True, False]