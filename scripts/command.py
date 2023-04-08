import json
from commands import CommandManager
from json_parser import fix_and_parse_json

def get_command(response):
    try:
        response_json = fix_and_parse_json(response)
        
        if "command" not in response_json:
            return "Error:" , "Missing 'command' object in JSON"
        
        command = response_json["command"]

        if "name" not in command:
            return "Error:", "Missing 'name' field in 'command' object"
        
        command_name = command["name"]

        # Use an empty dictionary if 'args' field is not present in 'command' object
        arguments = command.get("args", {})

        if not arguments:
            arguments = {}

        return command_name, arguments
    except json.decoder.JSONDecodeError:
        return "Error:", "Invalid JSON"
    # All other errors, return "Error: + error message"
    except Exception as e:
        return "Error:", str(e)

def execute_command(command_name, arguments):
	try:
		command = CommandManager.get(command_name)
		command.execute(**arguments)

	except Exception as e:
		# All errors, return "Error: + error message"
		return "Error: "  + str(e)