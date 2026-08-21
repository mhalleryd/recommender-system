from recommender_system.features.interactions import df2interact_mat


def test_df2interact_mat_returns_correct_shape_and_mapping(sample_reviews):
    interactions, item_to_idx = df2interact_mat(sample_reviews, 'user_id', 'item_id', 'recommend')

    assert interactions.shape == (3, 3)  # 3 users and 3 items
    assert item_to_idx == {"A": 0, "B": 1, "C": 2}  # Check if the mapping is correct