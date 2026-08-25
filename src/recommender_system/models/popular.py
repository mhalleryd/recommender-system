import pandas as pd


def recommend_popular(
    reviews: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:
    """Recommend the n most popular unseen items for every user."""

    popular_items = (
        reviews.groupby("item_id")
        .size()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    user_items = reviews.groupby("user_id")["item_id"].agg(set)

    recommendations = [] #type: ignore

    for user_id, seen_items in user_items.items():
        unseen_items = [
            item for item in popular_items
            if item not in seen_items
        ][:n]

        recommendations.extend(
            {"user_id": user_id, "item_id": item}
            for item in unseen_items
        )

    return pd.DataFrame(recommendations)