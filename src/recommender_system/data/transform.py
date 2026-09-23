from typing import Hashable, Any

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

def transform_item_metadata(df: pd.DataFrame) -> dict[Hashable, dict[Hashable, Any]]:
    """Transform item metadata into a dictionary for easy lookup."""

    df = df.rename({'id': 'item_id'}, axis=1)
    df = df.drop(columns=['discount_price', 
                          'reviews_url', 
                          'price', 
                          'early_access',
                          'metascore',
                          'title']
                    )
    df = df.dropna(subset=['app_name', 'item_id'])

    # Fill NaN values with None for JSON serialization
    df = df.astype(object).where(df.notna(), None)

    return df.drop_duplicates('item_id').set_index('item_id').to_dict(orient='index')

