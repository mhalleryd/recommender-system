import numpy as np
import pandas as pd

from recommender_system.data.load import load_data


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


def ndcg_at_k(
    recommendations: pd.DataFrame,
    test: pd.DataFrame,
    k: int = 10,
) -> float:
    """Calculate mean NDCG@K across users."""

    recommendations_by_user = recommendations.groupby("user_id")["item_id"]

    ndcgs = []

    for user_id, test_user in test.groupby("user_id"):
        if user_id not in recommendations_by_user.groups:
            ndcgs.append(0.0)
            continue

        recommended = (
            recommendations_by_user
            .get_group(user_id)
            .head(k)
            .tolist()
        )

        relevant = set(test_user["item_id"])

        dcg = sum(
            1 / np.log2(rank + 2)
            for rank, item_id in enumerate(recommended)
            if item_id in relevant
        )

        # Ideal ranking has all relevant items first
        ideal_hits = min(len(relevant), k)
        idcg = sum(
            1 / np.log2(rank + 2)
            for rank in range(ideal_hits)
        )

        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return sum(ndcgs) / len(ndcgs)


def get_reviews_per_game(reviews: pd.DataFrame) -> pd.DataFrame:
    """Get the number of positive and negative reviews for each game."""
    
    num_positive = reviews.groupby('item_id')['recommend'].sum()
    num_negative = reviews.groupby('item_id')['recommend'].count() - num_positive

    reviews_per_game = pd.DataFrame(reviews['item_id'].unique(), columns=['item_id'])
    reviews_per_game = reviews_per_game.join(pd.DataFrame([num_positive, num_negative]).T, on='item_id')
    reviews_per_game.columns = ['item_id', 'num_positive', 'num_negative']
    reviews_per_game['total'] = reviews_per_game['num_positive'] + reviews_per_game['num_negative']

    games = load_data("../data/steam_games.json.gz")
    games = games.rename({'id': 'item_id'}, axis=1)

    reviews_per_game = pd.merge(games[['app_name', 'item_id']], reviews_per_game, on='item_id')
    reviews_per_game = reviews_per_game.sort_values(by='num_positive', ascending=False, ignore_index=True)

    return reviews_per_game