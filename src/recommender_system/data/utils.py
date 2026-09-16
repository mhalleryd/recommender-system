import pandas as pd


def get_reviews_per_game(reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Get the number of positive and negative reviews per game."""

    num_positive = reviews.groupby('item_id')['recommend'].sum()
    num_negative = reviews.groupby('item_id')['recommend'].count() - num_positive

    reviews_per_game = pd.DataFrame(reviews['item_id'].unique(), columns=['item_id'])
    reviews_per_game = reviews_per_game.join(pd.DataFrame([num_positive, num_negative]).T, on='item_id')
    reviews_per_game.columns = ['item_id', 'num_positive', 'num_negative']
    reviews_per_game['total'] = reviews_per_game['num_positive'] + reviews_per_game['num_negative']
    return reviews_per_game