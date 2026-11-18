from types import SimpleNamespace

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_retrieval_endpoint_returns_ranked_documents(monkeypatch):
    fake_document = {
        "text": "Water damage claim investigation notes",
        "source": "claim_notes.txt",
        "doc_type": "claim_note",
        "metadata": {"claim_id": "CLM-1001"},
    }

    fake_result = SimpleNamespace(
        documents=[
            SimpleNamespace(
                document=fake_document,
                score=25.5,
                methods=["bm25"],
                bm25_score=20.5,
            )
        ],
        exact_count=0,
        bm25_count=1,
        filtered_count=0,
        latency_seconds=0.025,
    )

    def fake_retrieve(**kwargs):
        return fake_result

    monkeypatch.setattr(
        "backend.api.retrieval.retrieve",
        fake_retrieve,
    )

    response = client.post(
        "/api/v1/retrieve",
        json={"query": "water damage", "top_k": 5},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total_results"] == 1
    assert data["results"][0]["source"] == "claim_notes.txt"
    assert data["results"][0]["score"] == 25.5
    assert data["results"][0]["methods"] == ["bm25"]


def test_retrieval_endpoint_passes_filters_to_engine(monkeypatch):
    captured = {}

    fake_result = SimpleNamespace(
        documents=[],
        exact_count=0,
        bm25_count=0,
        filtered_count=0,
        latency_seconds=0.001,
    )

    def fake_retrieve(**kwargs):
        captured.update(kwargs)
        return fake_result

    monkeypatch.setattr(
        "backend.api.retrieval.retrieve",
        fake_retrieve,
    )

    response = client.post(
        "/api/v1/retrieve",
        json={
            "query": "closed flood claims",
            "top_k": 5,
            "filters": {
                "claim_status": "Closed",
                "loss_type": "Flood",
            },
        },
    )

    assert response.status_code == 200
    assert captured["sidebar_filters"] == {
        "claim_status": "Closed",
        "loss_type": "Flood",
    }


def test_retrieval_endpoint_rejects_blank_query():
    response = client.post(
        "/api/v1/retrieve",
        json={"query": "   "},
    )

    assert response.status_code == 422


def test_retrieval_endpoint_rejects_invalid_top_k():
    response = client.post(
        "/api/v1/retrieve",
        json={"query": "water damage", "top_k": 100},
    )

    assert response.status_code == 422
