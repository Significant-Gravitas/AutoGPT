from http import HTTPStatus
from typing import Any, Dict, List, Optional, Union, cast

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.check_weather_using_get_response_200 import (
    CheckWeatherUsingGETResponse200,
)
from ...types import UNSET, Response


def _get_kwargs(
    *,
    client: Client,
    location: str,
) -> Dict[str, Any]:
    url = "{}/api/weather".format(client.base_url)

    headers: Dict[str, str] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    params: Dict[str, Any] = {}
    params["location"] = location

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    return {
        "method": "get",
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.get_timeout(),
        "follow_redirects": client.follow_redirects,
        "params": params,
    }


def _parse_response(
    *, client: Client, response: httpx.Response
) -> Optional[CheckWeatherUsingGETResponse200]:
    if response.status_code == HTTPStatus.OK:
        response_200 = CheckWeatherUsingGETResponse200.from_dict(response.json())

        return response_200
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: Client, response: httpx.Response
) -> Response[CheckWeatherUsingGETResponse200]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Client,
    location: str,
) -> Response[CheckWeatherUsingGETResponse200]:
    """Get current weather information

    Args:
        location (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[CheckWeatherUsingGETResponse200]
    """

    kwargs = _get_kwargs(
        client=client,
        location=location,
    )

    response = httpx.request(
        verify=client.verify_ssl,
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Client,
    location: str,
) -> Optional[CheckWeatherUsingGETResponse200]:
    """Get current weather information

    Args:
        location (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        CheckWeatherUsingGETResponse200
    """

    return sync_detailed(
        client=client,
        location=location,
    ).parsed


async def asyncio_detailed(
    *,
    client: Client,
    location: str,
) -> Response[CheckWeatherUsingGETResponse200]:
    """Get current weather information

    Args:
        location (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[CheckWeatherUsingGETResponse200]
    """

    kwargs = _get_kwargs(
        client=client,
        location=location,
    )

    async with httpx.AsyncClient(verify=client.verify_ssl) as _client:
        response = await _client.request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Client,
    location: str,
) -> Optional[CheckWeatherUsingGETResponse200]:
    """Get current weather information

    Args:
        location (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        CheckWeatherUsingGETResponse200
    """

    return (
        await asyncio_detailed(
            client=client,
            location=location,
        )
    ).parsed
