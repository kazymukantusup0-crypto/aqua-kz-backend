from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["app_name"] == "AI Drought Monitoring Platform"


def test_risk_summary_endpoint() -> None:
    response = client.get("/api/risk-summary")
    assert response.status_code == 200
    summary = response.json()
    assert summary["total_observations"] >= 30
    assert summary["low_count"] + summary["medium_count"] + summary["high_count"] == summary["total_observations"]
    assert 0 <= summary["average_risk_score"] <= 1


def test_drone_targets_are_high_risk_and_sorted() -> None:
    response = client.get("/api/drone-targets")
    assert response.status_code == 200
    targets = response.json()
    assert targets
    assert all(target["risk_level"] == "high" for target in targets)
    assert [target["risk_score"] for target in targets] == sorted(
        (target["risk_score"] for target in targets), reverse=True
    )
