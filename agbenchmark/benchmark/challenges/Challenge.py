import json

class Challenge(object):
    def __init__(self, json_data):
        self.json_data = json_data

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file) as f:
            json_data = json.load(f)
        return cls(json_data)