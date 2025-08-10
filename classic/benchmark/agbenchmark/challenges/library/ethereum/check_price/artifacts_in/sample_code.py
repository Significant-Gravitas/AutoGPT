import requests


def get_ethereum_price() -> float:
    url = "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        return data["ethereum"]["usd"]
    else:
        raise Exception(f"Failed to fetch data: {response.status_code}")
