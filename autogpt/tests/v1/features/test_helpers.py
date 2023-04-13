import inspect
import json
import os

import requests


async def set_random_fields_to_static_values(new_response, static_fields_config):
    for key, value in static_fields_config["nested_updates"].items():
        update_nested(new_response, key, value)

def get_request_and_response_path(file_prefix, test_folder):
    request_path = f"{test_folder}/{file_prefix}_request.json"
    with open(request_path, 'r') as f:
        # Load the contents of the file as a JSON object
        request = json.load(f)

    response_path = f"{test_folder}/{file_prefix}_response.json"

    return request, response_path

async def get_response_async(client, request):
    return await client.request(request["method"], request["url"], json=request["body"], headers=request["headers"])

def get_response(request):
    return requests.request(request["method"], request["url"], json=request["body"], headers=request["headers"])

def get_test_variables(config):
    calling_frame = inspect.stack()[1]

    calling_script = calling_frame.filename
    test_name = calling_frame.function
    # script_path = os.path.abspath(calling_script)
    script_dir = os.path.dirname(calling_script)
    static_fields_config_path = os.path.join(script_dir, "test_lifecycle_agents/static_fields_config.json")
    test_folder = os.path.join(script_dir, test_name)

    mode = config.getoption('mode')
    static_fields_config = json.load(open(f"{test_folder}/static_fields_config.json"))
    return mode, test_folder, static_fields_config

def build_response(response):
    return {
        "status_code": response.status_code,
        "headers": dict(response.headers),
        "body": response.json(),
    }


def compare_old_response_to_new_response(mode, new_response, response_path):
    if os.path.exists(response_path):
        with open(response_path, 'r') as f:
            old_response = json.load(f)
    else:
        old_response = {}
    if mode == 'strict':
        assert new_response == old_response
    ## overwrite the old response with the new response
    with open(response_path, 'w') as f:
        json.dump(new_response, f, indent=4)

def update_nested(in_dict, key, value):
   for k, v in in_dict.items():
       if key == k:
           in_dict[k] = value
       elif isinstance(v, dict):
           update_nested(v, key, value)
       elif isinstance(v, list):
           for o in v:
               if isinstance(o, dict):
                   update_nested(o, key, value)
