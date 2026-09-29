# Steam Game Recommender

An end-to-end recommendation system built using the [Steam Review and Bundle Dataset](https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data).

The project explores recommendation methods for sparse implicit-feedback data and implements a deployable recommendation API.

## Models

Three approaches are implemented and evaluated:

- Popularity baseline
- Item-item collaborative filtering
- Matrix factorization

Models are evaluated using a temporal train/test split with **Recall@K** and **NDCG@K**. The popularity baseline performs best on the current dataset, which is highly sparse.

The matrix factorization model is used for personalized inference, with popular items used as a fallback for unknown users.

## API

The trained recommender is exposed through a **FastAPI** service:

```text
GET /recommendations/{user_id}?k=10
```

Recommendations include Steam game metadata and exclude games the user has already interacted with.

## Project Structure

```text
src/recommender_system/
├── api/          # FastAPI service
├── data/         # Data loading and preprocessing
├── features/     # Interaction matrix construction
├── inference/    # Recommendation inference
└── models/       # Recommendation models

scripts/          # Training scripts
tests/            # Unit and integration tests
notebooks/        # Exploration and experiments
```

## Tech Stack

Python · NumPy · SciPy · pandas · scikit-learn · FastAPI · pytest · Ruff · uv · Docker

## Running the API with Docker

```bash
docker build -t steam-recommender .
docker run --rm -p 8000:8000 steam-recommender
```

The interactive API documentation is then available at:

```text
http://localhost:8000/docs
```