from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock

from backend.api.features.experts import raise_intro, raise_routes, raise_soul
from backend.api.features.experts.raise_intro import RaiseIntroRequest
from backend.api.features.experts.raise_soul import SoulAnswer, SoulDraftRequest
from backend.api.features.onboarding_dump.structured import StructuredReply

EXPERT_ID = "expert-deps"

app = fastapi.FastAPI()
app.include_router(raise_routes.router)
app.dependency_overrides[raise_routes.auth.get_user_id] = lambda: "user-1"
app.dependency_overrides[raise_routes.enforce_payment_paywall] = lambda: None
client = fastapi.testclient.TestClient(app)


@pytest.fixture(autouse=True)
def no_rate_limit(mocker: pytest_mock.MockerFixture) -> None:
    mocker.patch.object(raise_routes._intro_limiter, "check", new_callable=AsyncMock)
    mocker.patch.object(raise_routes._soul_limiter, "check", new_callable=AsyncMock)


def _expert(**overrides) -> SimpleNamespace:
    values = {
        "id": EXPERT_ID,
        "name": "Deps merger",
        "job_title": "Release engineer",
        "identity": "I'm Deps merger, an AI Expert created by you.",
        "boundaries": "",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _answers() -> SoulDraftRequest:
    return SoulDraftRequest(
        intro="I'll keep dependency updates reviewed and merged.",
        answers=[
            SoulAnswer(question="Which repo?", answer="autogpt"),
            SoulAnswer(question="How much freedom?", answer="Ask before merging"),
        ],
    )


def test_intro_answers_with_the_template_when_no_model_is_configured(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch.object(raise_intro, "get_openai_client", return_value=None)

    response = client.get(
        "/raise/intro", params={"name": "Deps merger", "job_title": "Release engineer"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["is_generated"] is False
    assert "Release engineer" in data["intro"]
    assert 2 <= len(data["questions"]) <= 3
    assert all(len(q["options"]) == 3 for q in data["questions"])


def test_intro_uses_the_model_answer(mocker: pytest_mock.MockerFixture) -> None:
    mocker.patch.object(raise_intro, "get_openai_client", return_value=Mock())
    mocker.patch.object(
        raise_intro,
        "request_json",
        new_callable=AsyncMock,
        return_value=StructuredReply(
            data={
                "intro": "I'll keep your dependency update PRs reviewed and merged.",
                "questions": [
                    {
                        "question": "Which repo should I watch?",
                        "options": ["The main app", "All repos", "Just one"],
                    },
                    {"question": "Dropped: one option", "options": ["Only"]},
                ],
            },
            finish_reason="stop",
            chars=200,
        ),
    )

    response = client.get("/raise/intro", params={"name": "Deps merger"})

    data = response.json()
    assert data["is_generated"] is True
    assert data["intro"].startswith("I'll keep your dependency")
    assert [q["question"] for q in data["questions"]] == ["Which repo should I watch?"]


@pytest.mark.parametrize(
    "data",
    [None, {}, {"intro": "Hi", "questions": []}, {"intro": " ", "questions": "x"}],
)
def test_parse_intro_rejects_unusable_answers(data: object) -> None:
    assert raise_intro.parse_intro(data) is None


@pytest.mark.asyncio
async def test_intro_falls_back_when_the_model_fails(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch.object(raise_intro, "get_openai_client", return_value=Mock())
    mocker.patch.object(
        raise_intro,
        "request_json",
        new_callable=AsyncMock,
        side_effect=TimeoutError(),
    )

    intro = await raise_intro.generate_intro(RaiseIntroRequest(name="Deps merger"))

    assert intro.is_generated is False
    assert intro.questions


def test_soul_draft_queues_the_write_for_the_owner(
    mocker: pytest_mock.MockerFixture,
) -> None:
    expert = _expert()
    get_expert = mocker.patch.object(
        raise_routes.experts_db, "get_expert", new_callable=AsyncMock
    )
    get_expert.return_value = expert
    write = mocker.patch.object(raise_soul, "write_soul", new_callable=AsyncMock)

    response = client.post(
        f"/{EXPERT_ID}/soul/draft", json=_answers().model_dump(mode="json")
    )

    assert response.status_code == 202
    assert response.json() == {"expected_identity": expert.identity}
    get_expert.assert_awaited_once_with("user-1", EXPERT_ID, include_workflows=False)
    write.assert_awaited_once()


def test_soul_draft_404s_for_an_unknown_expert(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch.object(
        raise_routes.experts_db,
        "get_expert",
        new_callable=AsyncMock,
        return_value=None,
    )

    response = client.post(
        f"/{EXPERT_ID}/soul/draft", json=_answers().model_dump(mode="json")
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_write_soul_records_the_answers_without_a_model(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch.object(raise_soul, "get_openai_client", return_value=None)
    update = mocker.patch.object(
        raise_soul.experts_db,
        "update_soul_fields_if_current",
        new_callable=AsyncMock,
        return_value=True,
    )
    expert = _expert()

    await raise_soul.write_soul("user-1", expert, _answers())

    kwargs = update.await_args.kwargs
    assert kwargs["expected_identity"] == expert.identity
    assert kwargs["identity"].startswith(
        "I'm Deps merger, the Release engineer on your team."
    )
    assert "Which repo? autogpt" in kwargs["identity"]
    assert "How much freedom? Ask before merging" in kwargs["identity"]


@pytest.mark.asyncio
async def test_write_soul_uses_the_model_draft(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch.object(raise_soul, "get_openai_client", return_value=Mock())
    mocker.patch.object(
        raise_soul,
        "request_json",
        new_callable=AsyncMock,
        return_value=StructuredReply(
            data={
                "identity": "I keep autogpt's dependency updates merged.",
                "boundaries": "I ask before merging a major version bump.",
            },
            finish_reason="stop",
            chars=100,
        ),
    )
    update = mocker.patch.object(
        raise_soul.experts_db,
        "update_soul_fields_if_current",
        new_callable=AsyncMock,
        return_value=True,
    )

    await raise_soul.write_soul("user-1", _expert(), _answers())

    kwargs = update.await_args.kwargs
    assert kwargs["identity"] == "I keep autogpt's dependency updates merged."
    assert kwargs["boundaries"] == "I ask before merging a major version bump."


def test_intro_rejects_a_blank_name() -> None:
    assert client.get("/raise/intro", params={"name": ""}).status_code == 422


def test_intro_is_rate_limited_per_user(mocker: pytest_mock.MockerFixture) -> None:
    mocker.patch.object(
        raise_routes._intro_limiter,
        "check",
        new_callable=AsyncMock,
        side_effect=fastapi.HTTPException(status_code=429, detail="slow down"),
    )
    generate = mocker.patch.object(
        raise_intro, "generate_intro", new_callable=AsyncMock
    )

    response = client.get("/raise/intro", params={"name": "Deps merger"})

    assert response.status_code == 429
    generate.assert_not_awaited()
