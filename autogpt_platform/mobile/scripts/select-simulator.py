import json
import sys

with open(sys.argv[1]) as source:
    devices = json.load(source)["devices"]

runtime_id = "com.apple.CoreSimulator.SimRuntime.iOS-" + sys.argv[2].replace(".", "-")
candidates = [
    device
    for runtime, values in devices.items()
    if runtime == runtime_id
    for device in values
    if device["isAvailable"]
    and device["name"].startswith("iPhone")
    and device["state"] == "Shutdown"
]
if not candidates:
    raise SystemExit(f"No available iPhone simulator for {runtime_id} on this runner")
print(candidates[0]["udid"])
