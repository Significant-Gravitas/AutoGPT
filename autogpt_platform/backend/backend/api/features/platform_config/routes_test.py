import fastapi
import fastapi.testclient
import pytest

from .routes import router, settings

app = fastapi.FastAPI()
app.include_router(router, prefix="/api")
client = fastapi.testclient.TestClient(app)


@pytest.mark.parametrize("enabled", [True, False])
def test_the_browser_reads_the_payment_setting(set_platform_payment, enabled):
    set_platform_payment(enabled)
    response = client.get("/api/platform-config")
    assert response.status_code == 200
    assert response.json()["payment_enabled"] is enabled


@pytest.mark.parametrize(
    ("setting", "expected"), [("", None), ("  ", None), ("1.2.0", "1.2.0")]
)
def test_the_app_reads_the_minimum_version(monkeypatch, setting, expected):
    monkeypatch.setattr(settings.config, "mobile_min_app_version", setting)
    response = client.get("/api/platform-config")
    assert response.status_code == 200
    assert response.json()["mobile_min_version"] == expected
