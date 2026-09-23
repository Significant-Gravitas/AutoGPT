import pytest
from fastapi import FastAPI

from backend.api.features.subscription_trial_routes import router


@pytest.mark.parametrize("path", ["/cancel", "/confirm"])
def test_trial_mutation_errors_are_documented(path):
    app = FastAPI()
    app.include_router(router)
    responses = app.openapi()["paths"][f"/credits/trial{path}"]["post"]["responses"]
    assert {"200", "409", "502"} <= responses.keys()
