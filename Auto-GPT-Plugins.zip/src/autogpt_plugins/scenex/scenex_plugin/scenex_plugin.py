import os
import requests
from typing import List, Union


def is_api_key_set() -> bool:
    return True if os.getenv("SCENEX_API_KEY") else False


def get_api_key():
    api_key = os.getenv("SCENEX_API_KEY")
    if not api_key:
        return "Error: SCENEX_API_KEY not set in environment."
    return api_key


Algorithm = Union["Aqua", "Bolt", "Comet", "Dune", "Ember", "Flash"]


def describe_image(
    image: str,
    algorithm: Algorithm = "Dune",
    features: List[str] = [],
    languages: List[str] = [],
) -> str:
    url = "https://us-central1-causal-diffusion.cloudfunctions.net/describe"
    headers = {
        "x-api-key": f"token {get_api_key()}",
        "content-type": "application/json",
    }

    payload = {
        "data": [
            {
                "image": image,
                "algorithm": algorithm,
                "features": features,
                "languages": languages,
            }
        ]
    }

    response = requests.post(url, headers=headers, json=payload)
    result = response.json().get("result", [])
    img = result[0] if result else {}

    return {"image": image, "description": img.get("text", "")}
