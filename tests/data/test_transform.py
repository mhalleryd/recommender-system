import pandas as pd

from recommender_system.data.transform import train_test_split, transform_reviews


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

        },
        {
            "user_id": "user_2",
            "reviews": [
                {
                    "item_id": pd.NA,
                    "recommend": False,
                    "review": "hmm, unsure",
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


def test_train_test_split():
    reviews = pd.DataFrame({
        "user_id": ["A", "A", "A", "B", "B"],
        "item_id": ["1", "2", "3", "4", "5"],
        "recommend": [True, True, True, True, True],
        "posted": [
            "Posted January 1, 2020.",
            "Posted January 2, 2020.",
            "Posted January 3, 2020.",
            "Posted January 1, 2020.",
            "Posted January 2, 2020.",
        ],
    })

    train, test = train_test_split(reviews, n_test=1)
    assert len(train) == 3
    assert len(test) == 2
    assert list(train["user_id"]) == ["A", "A", "B"]
    assert list(train["item_id"]) == ["1", "2", "4"]
    assert list(test["user_id"]) == ["A", "B"]
    assert list(test["item_id"]) == ["3", "5"]