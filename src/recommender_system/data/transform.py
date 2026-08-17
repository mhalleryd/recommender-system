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

