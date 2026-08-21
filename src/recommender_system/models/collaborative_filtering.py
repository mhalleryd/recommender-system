import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from recommender_system.features.interactions import df2interact_mat


def get_item_similarity_matrix(reviews: pd.DataFrame) -> tuple[sparse.csr_matrix, dict]:
    """
    Compute the item-item similarity matrix from user-item interactions.

    Parameters:
    reviews (pd.DataFrame): DataFrame containing user-item interactions with columns 'user_id', 'item_id', and 'recommend'.

    Returns:
    similarities (sparse.csr_matrix): Sparse matrix of item-item similarities.
    item_to_idx (dict): Mapping from item IDs to their corresponding indices in the similarity matrix.
    """
    required_columns = {'user_id', 'item_id', 'recommend'}
    if not required_columns.issubset(reviews.columns):
        raise ValueError(f"Reviews must contain these columns: {required_columns}")

    interactions, item_to_idx = df2interact_mat(reviews, 'user_id', 'item_id', 'recommend')

    similarities = cosine_similarity(interactions.T, dense_output=False) #Transpose leads to item-item similarity matrix

    return similarities, item_to_idx #type: ignore


def get_similar_items(item_id: str, similarities: sparse.csr_matrix, item_to_idx: dict, k: int = 5) -> pd.DataFrame:
    """
    Get the top k similar items for a given item based on the similarity matrix.

    Parameters:
    item_id (str): The ID of the item for which to find similar items.
    similarities (sparse.csr_matrix): Sparse matrix of item-item similarities.
    item_to_idx (dict): Mapping from item IDs to their corresponding indices in the similarity matrix.
    k (int): Number of top similar items to return.

    Returns:
    pd.DataFrame: DataFrame containing the top k similar items and their similarity scores.
    """
    if item_id not in item_to_idx:
        raise ValueError(f"Item ID {item_id} not found in the item_to_idx mapping.")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer.")

    idx = item_to_idx[item_id]
    sim_scores = similarities[idx].toarray().flatten()
    # Get indices of the top k similar items, excluding the item itself
    top_k_indices = sim_scores.argsort()[::-1][1:k+1]

    idx_to_item = {index: item for item, index in item_to_idx.items()}
    items = [idx_to_item[index] for index in top_k_indices]

    return pd.DataFrame(
        {'item_id': items, 'scores': sim_scores[top_k_indices]}
    ).sort_values(by='scores', ascending=False).reset_index(drop=True)