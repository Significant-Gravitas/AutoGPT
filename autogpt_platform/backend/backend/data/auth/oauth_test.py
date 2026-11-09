"""What checking an OAuth access token costs before the database is asked."""

import pytest
import pytest_mock

from backend.data.auth.oauth import InvalidTokenError, validate_access_token


@pytest.mark.parametrize(
    "token", ["", "garbage", "agpt_rt_refresh-token", "agpt_an-api-key"]
)
async def test_a_value_not_in_the_access_token_format_is_never_looked_up(
    mocker: pytest_mock.MockFixture, token: str
) -> None:
    """Every access token carries the prefix, so nothing else can match one."""
    model = mocker.patch("backend.data.auth.oauth.PrismaOAuthAccessToken")

    with pytest.raises(InvalidTokenError):
        await validate_access_token(token)

    model.prisma.assert_not_called()
