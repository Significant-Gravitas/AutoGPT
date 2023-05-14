import json
from typing import Any, Dict, List, Tuple, Union


class DSLParser:
    def __init__(self):
        pass

    def parse(self, json_response: str) -> Tuple[str, Dict[str, Any], Dict[str, str]]:
        assistant_reply = json.loads(json_response)

        braindump = assistant_reply["braindump"]
        command = {
            "name": assistant_reply["command"]["name"],
            "args": assistant_reply["command"]["args"],
        }
        key_updates = assistant_reply["key_updates"]

        return braindump, command, key_updates
