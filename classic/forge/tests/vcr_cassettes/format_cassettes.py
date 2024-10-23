import contextlib
import json
import os
from hashlib import sha256
from pathlib import Path

import yaml
from yaml import CDumper as Dumper
from yaml import CLoader as Loader


def convert_cassette_file(filename: str | Path):
    print(f"{filename} STARTING")

    with open(filename) as c:
        cassette_content = yaml.load(c, Loader)

    # Iterate over all request+response pairs
    for interaction in cassette_content["interactions"]:
        request_body: str = interaction["request"]["body"]
        if request_body is None:
            continue

        with contextlib.suppress(json.decoder.JSONDecodeError):
            request_obj = json.loads(request_body)

            # Strip `max_tokens`, since its value doesn't matter
            #  as long as the request succeeds
            if "max_tokens" in request_obj:
                del request_obj["max_tokens"]

            # Sort the keys of the request body
            request_body = json.dumps(request_obj, sort_keys=True)

        headers = interaction["request"]["headers"]

        # Calculate hash for the request body, used for VCR lookup
        headers["X-Content-Hash"] = [
            sha256(request_body.encode(), usedforsecurity=False).hexdigest()
        ]

        # Strip auth headers
        if "AGENT-MODE" in headers:
            del headers["AGENT-MODE"]
        if "AGENT-TYPE" in headers:
            del headers["AGENT-TYPE"]
        if "OpenAI-Organization" in headers:
            del headers["OpenAI-Organization"]

        interaction["request"]["body"] = request_body

    with open(filename, "w") as c:
        c.write(yaml.dump(cassette_content, Dumper=Dumper))

    print(f"{filename} DONE")


# Iterate over all .yaml files in the current folder and its subdirectories
for dirpath, _, files in os.walk("."):
    for file in files:
        if not file.endswith(".yaml"):
            continue
        convert_cassette_file(os.path.join(dirpath, file))
