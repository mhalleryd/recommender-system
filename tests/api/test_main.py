import importlib

import pytest
from fastapi.testclient import TestClient

import recommender_system.inference.recommender as recommender_module


@pytest.fixture
def client(monkeypatch, recommender):
    # Prevent main.py from loading the real production artifacts.
    monkeypatch.setattr(
        recommender_module.Recommender,
        "load",
        lambda _: recommender,
    )

    from recommender_system.api import main

    main = importlib.reload(main)
    return TestClient(main.app)


def test_known_user_recommendations(client):
    response = client.get("/recommendations/user-1?k=2")

    assert response.status_code == 200

    body = response.json()

    assert body["user_id"] == "user-1"
    assert len(body["recommendations"]) == 2

    assert set(body["recommendations"][0]) == {
        "item_id",
        "name",
        "developer",
        "genres",
    }

    assert [item["item_id"] for item in body["recommendations"]] == [
        "item-b",
        "item-c",
    ]


def test_unknown_user_uses_popularity_fallback(client):
    response = client.get("/recommendations/unknown-user?k=2")

    assert response.status_code == 200

    body = response.json()

    assert body["user_id"] == "unknown-user"
    assert [item["item_id"] for item in body["recommendations"]] == [
        "item-d",
        "item-c",
    ]


@pytest.mark.parametrize("k", [0, 101])
def test_invalid_k_returns_422(client, k):
    response = client.get(f"/recommendations/user-1?k={k}")

    assert response.status_code == 422


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}