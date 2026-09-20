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
    items = recommender.recommend(user_id, k)

    return {
        "user_id": user_id,
        "recommendations": items,
    }