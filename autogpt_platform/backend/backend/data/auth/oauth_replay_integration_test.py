import asyncio
from uuid import uuid4

import pytest
from prisma.enums import APIKeyPermission

from backend.data.auth import oauth
from backend.data.db import prisma


@pytest.mark.integration
@pytest.mark.asyncio(loop_scope="session")
async def test_oauth_code_and_refresh_token_have_one_concurrent_winner(server):
    user_id = f"oauth-race-user-{uuid4()}"
    app_id = str(uuid4())
    redirect_uri = "https://client.example/callback"
    scopes = [APIKeyPermission.READ_GRAPH]
    await prisma.user.create(data={"id": user_id, "email": f"{user_id}@example.com"})
    await prisma.oauthapplication.create(
        data={
            "id": app_id,
            "name": "OAuth replay race",
            "clientId": f"oauth-race-{uuid4()}",
            "clientSecret": "unused-test-hash",
            "clientSecretSalt": "unused-test-salt",
            "redirectUris": [redirect_uri],
            "scopes": scopes,
            "ownerId": user_id,
        }
    )

    code = await oauth.create_authorization_code(app_id, user_id, scopes, redirect_uri)
    code_results = await asyncio.gather(
        *(
            oauth.consume_authorization_code(code.code, app_id, redirect_uri)
            for _ in range(2)
        ),
        return_exceptions=True,
    )
    assert sum(not isinstance(result, Exception) for result in code_results) == 1
    assert (
        sum(isinstance(result, oauth.InvalidGrantError) for result in code_results) == 1
    )

    refresh = await oauth.create_refresh_token(app_id, user_id, scopes)
    refresh_results = await asyncio.gather(
        *(
            oauth.refresh_tokens(refresh.token.get_secret_value(), app_id)
            for _ in range(2)
        ),
        return_exceptions=True,
    )
    assert sum(not isinstance(result, Exception) for result in refresh_results) == 1
    assert (
        sum(isinstance(result, oauth.InvalidGrantError) for result in refresh_results)
        == 1
    )
    assert await prisma.oauthaccesstoken.count(where={"applicationId": app_id}) == 1
    assert await prisma.oauthrefreshtoken.count(where={"applicationId": app_id}) == 2
