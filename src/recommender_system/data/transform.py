import pandas as pd


def transform_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten user reviews into one row per user-game interaction."""

    reviews = df.explode('reviews', ignore_index=True)

    # Normalizes the reviews column into separate columns for each review attribute
    reviews = pd.concat(
        [
            reviews[["user_id"]],
            pd.json_normalize(reviews["reviews"])
        ],
        axis=1
    )

    reviews = reviews.drop(
    columns=["review", "funny", "last_edited", "helpful"]
    )
    reviews = reviews.astype({'item_id': 'str', 'recommend': 'bool'})

    return reviews.dropna()


def train_test_split(
    reviews: pd.DataFrame,
    n_test: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    reviews = reviews.copy()

    reviews["posted"] = (
        reviews["posted"]
        .str.replace("Posted ", "", regex=False)
        .str.replace(".", "", regex=False)
    )
    reviews["posted"] = pd.to_datetime(reviews["posted"], errors="coerce")

    positive = reviews[reviews["recommend"] == True].copy()

    # Only include users with enough history for both train and test
    positive = positive[
        positive.groupby("user_id")["item_id"].transform("size") > n_test
    ]

    positive = positive.sort_values(["user_id", "posted"])

    test = positive.groupby("user_id").tail(n_test)
    train = positive.drop(test.index)

    return train, test