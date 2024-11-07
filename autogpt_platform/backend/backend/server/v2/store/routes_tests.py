import fastapi
import fastapi.testclient

import backend.server.routers.v2.store

app = fastapi.FastAPI()
app.include_router(backend.server.routers.v2.store.router)

client = fastapi.testclient.TestClient(app)


def test_get_agents_featured():
    response = client.get(
        "/agents?featured=true&top=false&categories=&page=1&page_size=20"
    )
    assert response.status_code == 200
    assert all(agent["featured"] for agent in response.json())


def test_get_agents_top():
    response = client.get(
        "/agents?featured=false&top=true&categories=&page=1&page_size=20"
    )
    assert response.status_code == 200
    assert response.json()  # Ensure there are agents returned


def test_get_agents_categories():
    response = client.get(
        "/agents?featured=false&top=false&categories=SEO&page=1&page_size=20"
    )
    assert response.status_code == 200
    assert all("SEO" in agent["description"] for agent in response.json())


def test_get_agents_pagination():
    response = client.get(
        "/agents?featured=false&top=false&categories=&page=2&page_size=1"
    )
    assert response.status_code == 200
    assert len(response.json()) == 1


def test_get_agents_no_agents():
    response = client.get(
        "/agents?featured=false&top=false&categories=NonExistentCategory&page=1&page_size=20"
    )
    assert response.status_code == 200
    assert response.json() == []


def test_get_agents_invalid_request():
    response = client.get(
        "/agents?featured=invalid&top=false&categories=&page=1&page_size=20"
    )
    assert response.status_code == 422  # Unprocessable Entity


def test_get_agent_details():
    response = client.get("/agents/AI%20Labs/Super%20SEO%20Optimizer")
    assert response.status_code == 200
    assert response.json()["agentName"] == "Super SEO Optimizer"


def test_get_creators():
    response = client.get("/creators?page=1&page_size=20")
    assert response.status_code == 200
    assert response.json()  # Ensure there are creators returned


def test_get_creator_details():
    response = client.get("/creator/AI%20Labs")
    assert response.status_code == 200
    assert response.json()["username"] == "AI Labs"


def test_get_creator_invalid():
    response = client.get("/creator/NonExistentUser")
    assert response.status_code == 404  # Not Found
