import json
import os
import inspect
from httpx import AsyncClient
import pytest
import requests

from autogpt.tests.v1.features.test_helpers import get_test_variables, get_request_and_response_path, get_response_async, \
    compare_old_response_to_new_response, build_response, set_random_fields_to_static_values

@pytest.mark.anyio
async def test_lifecycle_agents(config, client: AsyncClient) -> None:
    mode, test_folder, static_fields_config = get_test_variables(config)

    ## Create the agent
    request, response_path = get_request_and_response_path("1_create_agent", test_folder)

    response = await get_response_async(client, request)

    assert response.status_code == 200

    new_response = build_response(response)

    compare_old_response_to_new_response(mode, new_response, response_path)

    ## Start the agent
    request, response_path = get_request_and_response_path("2_start_agent", test_folder)

    response = await get_response_async(client, request)

    assert response.status_code == 200
    new_response = build_response(response)

    thread_id = new_response["body"]["thread_id"]

    await set_random_fields_to_static_values(new_response, static_fields_config)


    compare_old_response_to_new_response(mode, new_response, response_path)

    ## Stop the agent as soon as it starts

    request, response_path = get_request_and_response_path("3_stop_agent", test_folder)
    request["body"]["thread_id"] = thread_id
    response = await get_response_async(client, request)

    assert response.status_code == 200
    new_response = build_response(response)
    compare_old_response_to_new_response(mode, new_response, response_path)
