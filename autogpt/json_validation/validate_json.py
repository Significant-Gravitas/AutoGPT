import json
from jsonschema import Draft7Validator
from autogpt.config import Config
from autogpt.logs import logger

CFG = Config()


def validate_json(json_object: object, schema_name: object) -> object:
    """
    :type schema_name: object
    :param schema_name:
    :type json_object: object
    """
    with open(f"autogpt/json_schemas/{schema_name}.json", "r") as f:
        schema = json.load(f)
    validator = Draft7Validator(schema)

    if errors := sorted(validator.iter_errors(json_object), key=lambda e: e.path):
        logger.error("The JSON object is invalid.")
        if CFG.debug_mode:
            logger.error(json.dumps(json_object, indent=4))   # Replace 'json_object' with the variable containing the JSON data
            logger.error("The following issues were found:")

            for error in errors:
                logger.error(f"Error: {error.message}")
    else:
        print("The JSON object is valid.")

    return json_object
