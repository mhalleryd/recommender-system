from fastapi import FastAPI, Query

from recommender_system.inference.recommender import Recommender

app = FastAPI(
    title="Steam Recommender API",
    version="1.0.0",
)

recommender = Recommender.load("artifacts/mf")


@app.get("/recommendations/{user_id}")
def get_recommendations(
    user_id: str,
    k: int = Query(default=10, ge=1, le=100),
):
    item_ids = recommender.recommend(user_id, k)

    recommendations = []

    for item_id in item_ids:
        metadata = recommender.get_item_metadata(item_id) or {}

        recommendations.append({
            "item_id": item_id,
            "name": metadata.get("app_name"),
            "developer": metadata.get("developer"),
            "genres": metadata.get("genres"),
        })

    return {
        "user_id": user_id,
        "recommendations": recommendations,
    }


@app.get("/health")
def health():
    return {"status": "ok"}