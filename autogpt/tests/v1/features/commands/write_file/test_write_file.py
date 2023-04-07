import sys
from asyncio import sleep
from httpx import AsyncClient
import pytest
from autogpt.tests.v1.features.test_helpers import compare_old_response_to_new_response, build_response, get_response_async, \
    get_request_and_response_path, get_test_variables, set_random_fields_to_static_values


@pytest.mark.anyio
async def test_write_file(config, client: AsyncClient) -> None:
    mode, test_folder, static_fields_config = get_test_variables(config)

    ## Create the agent
    request, response_path = get_request_and_response_path("1_create_agent", test_folder)

    response = await get_response_async(client, request)

    assert response.status_code == 200

    new_response = build_response(response)

    compare_old_response_to_new_response(mode, new_response, response_path)

    ## Start the agent

    request, response_path = get_request_and_response_path("2_start_agent", test_folder)

    # Save the original stdout
    original_stdout = sys.stdout

    # Redirect stdout to a file
    with open(f"{test_folder}/write_file_output.txt", 'w') as file:
        sys.stdout = file
        response = await get_response_async(client, request)
        await sleep(45)

    # Restore the original stdout
    assert response.status_code == 200
    new_response = build_response(response)
    thread_id = new_response["body"]["thread_id"]

    await set_random_fields_to_static_values(new_response, static_fields_config)
    compare_old_response_to_new_response(mode, new_response, response_path)

    ## Stop the agent

    request, response_path = get_request_and_response_path("3_stop_agent", test_folder)
    request["body"]["thread_id"] = thread_id
    response = await get_response_async(client, request)

    ## should be 500 because the agent should
    assert response.status_code == 500
    new_response = build_response(response)
    compare_old_response_to_new_response(mode, new_response, response_path)

    sys.stdout = original_stdout
