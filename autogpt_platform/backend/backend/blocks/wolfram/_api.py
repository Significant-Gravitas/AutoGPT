from backend.sdk import APIKeyCredentials, Requests


async def llm_api_call(credentials: APIKeyCredentials, question: str) -> str:
    params = {"appid": credentials.api_key.get_secret_value(), "input": question}
    response = await Requests().get(
        "https://www.wolframalpha.com/api/v1/llm-api", params=params
    )
    if not response.ok:
        raise ValueError(f"API request failed: {response.status} {response.text()}")

    answer = response.text() if response.text() else ""

    return answer
