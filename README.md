# Steam Recommendation System

An end-to-end recommendation system built using Steam user-game interaction data. The project explores collaborative filtering methods for recommending games, with a particular focus on reproducible ML workflows.

The project is also used to practice production-oriented software engineering for machine learning, including automated testing, type checking, linting, dependency management, and continuous integration.

## Project Status

Current work focuses on item-item collaborative filtering. Future work will include matrix factorization and comparison of different recommendation approaches.

## Repository structure
```text
.
├── src/
│   └── recommender_system/
│       ├── data/
│       │   ├── load.py
│       │   └── transform.py
│       ├── features/
│       │   └── interactions.py
│       └── models/
│           └── collaborative_filtering.py
│
├── tests/
│   ├── data/
│   ├── features/
│   └── models/
│
├── notebooks/
│   ├── EDA.ipynb
│   └── collaborative_filtering.ipynb
│
├── pyproject.toml
├── uv.lock
└── README.md
```

## Dataset

The project uses the [Steam User Reviews dataset](https://huggingface.co/datasets/recommender-system/steam-review-and-bundle-dataset).

The data contains user reviews of Steam games, including:

- User IDs
- Game IDs
- Review text
- Recommendation outcome
- Review metadata

The current pipeline extracts user-game interactions from the review data and constructs a sparse user-item interaction matrix.

## Current Approach

### 1. Data processing

Raw review data is loaded and transformed into a tabular representation where each row represents a user-game interaction.

```text
Raw Steam reviews
       ↓
Data loading
       ↓
Review transformation
       ↓
User-game interactions
       ↓
Sparse interaction matrix