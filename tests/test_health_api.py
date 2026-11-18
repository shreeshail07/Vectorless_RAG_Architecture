
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health_endpoint_returns_healthy_status():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["service"] == "Insurance Claims Intelligence API"
    assert data["version"] == "1.0.0"


def test_root_endpoint_returns_welcome_message():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["message"] == "Welcome to Insurance Claims Intelligence API"
    assert data["documentation"] == "/docs"


def test_unknown_endpoint_returns_404():
    response = client.get("/unknown-endpoint")

    assert response.status_code == 404
