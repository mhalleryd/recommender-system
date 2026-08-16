import gzip

from recommender_system.data.load import load_data


def test_load_data(tmp_path):
    file_path = tmp_path / "test_data.json.gz"

    rows = [
        "{'user_id': 'user_1', 'reviews': []}\n",
        "{'user_id': 'user_2', 'reviews': []}\n",
    ]

    with gzip.open(file_path, "wt", encoding="utf-8") as f:
        f.writelines(rows)

    result = load_data(file_path)

    assert len(result) == 2
    assert list(result["user_id"]) == ["user_1", "user_2"]
    assert "reviews" in result.columns