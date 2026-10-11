import json
import sys

with open(sys.argv[1]) as source:
    devices = json.load(source)["devices"]

candidates = [
    device
    for runtime, values in devices.items()
    if "iOS" in runtime
    for device in values
    if device["isAvailable"]
    and device["name"].startswith("iPhone")
    and device["state"] == "Shutdown"
]
if not candidates:
    raise SystemExit("No available iPhone simulator on this runner")
print(candidates[0]["udid"])
