import json
from pathlib import Path

from scipy.sparse import save_npz

from recommender_system.data.load import load_data
from recommender_system.data.transform import transform_reviews
from recommender_system.models.collaborative_filtering import df2interact_mat
from recommender_system.models.matrixfactorization import MatrixFactorization

DATA_PATH = Path("data/australian_user_reviews.json.gz")
ARTIFACT_DIR = Path("artifacts/mf")

N_FACTORS = 25
N_EPOCHS = 10
LEARNING_RATE = 0.01
REGULARIZATION = 0.0
RANDOM_STATE = 42


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # Load and transform
    df = load_data(DATA_PATH)
    reviews = transform_reviews(df)

    # Build interaction matrix
    X, item_to_idx, user_to_idx = df2interact_mat(
        reviews,
        "user_id",
        "item_id",
        "recommend",
    )

    # Train
    model = MatrixFactorization(
        n_factors=N_FACTORS,
        random_state=RANDOM_STATE,
    )

    model.fit(
        X,
        n_epochs=N_EPOCHS,
        learning_rate=LEARNING_RATE,
        regularization=REGULARIZATION,
    )

    # Get popular items - used for cold start recommendations
    popular_items = (
            reviews.groupby("item_id")
            .size()
            .sort_values(ascending=False)
            .index
            .tolist()
        )

    # Save artifacts
    model.save(ARTIFACT_DIR / "model.npz")

    save_npz(
        ARTIFACT_DIR / "interactions.npz",
        X,
    )

    with open(ARTIFACT_DIR / "item_to_idx.json", "w") as f:
        json.dump(item_to_idx, f)

    with open(ARTIFACT_DIR / "user_to_idx.json", "w") as f:
        json.dump(user_to_idx, f)

    with open(ARTIFACT_DIR / "popular_items.json", "w") as f:
        json.dump(popular_items, f)

    print(f"Artifacts saved to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()