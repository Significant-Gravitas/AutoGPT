import base64
import json

# Load JSON data from a file
with open("secrets.json", "r") as f:
    data = json.load(f)

# Convert the JSON object into a string
json_string = json.dumps(data)

# Encode the string into bytes
json_bytes = json_string.encode("utf-8")

# Convert the bytes to a base64 string
base64_string = base64.b64encode(json_bytes).decode("utf-8")

print(base64_string)
