import pandas as pd


def recall_at_k(
    recommendations: pd.DataFrame,
    test: pd.DataFrame,
    k: int = 10,
) -> float:
    """Calculate mean Recall@K across users."""

    recommendations_by_user = recommendations.groupby("user_id")["item_id"]

    recalls = []

    for user_id, test_user in test.groupby("user_id"):
        recommended_items = set(
            recommendations_by_user.get_group(user_id).head(k)
            if user_id in recommendations_by_user.groups
            else []
        )

        relevant_items = set(test_user["item_id"])

        hits = len(relevant_items & recommended_items)
        recalls.append(hits / len(relevant_items))

    return sum(recalls) / len(recalls)