import pandas as pd
from scipy import sparse


def df2interact_mat(df: pd.DataFrame, user_col: str, item_col: str, interact_col: str) -> tuple[sparse.csr_matrix, dict]:
	"""
	Convert a pandas DataFrame to a sparse interaction matrix using scipy.sparse."""
	# explicit, stable mapping
	user_ids = pd.Index(df[user_col].drop_duplicates())
	item_ids = pd.Index(df[item_col].drop_duplicates())

	user_to_idx = {u: i for i, u in enumerate(user_ids)}
	item_to_idx = {item: j for j, item in enumerate(item_ids)}

	rows = df[user_col].map(user_to_idx).to_numpy()
	cols = df[item_col].map(item_to_idx).to_numpy()
	values = df[interact_col].astype(float).to_numpy()

	interactions = sparse.csr_matrix(
		(values, (rows, cols)),
		shape=(len(user_ids), len(item_ids))
	)

	return interactions, item_to_idx