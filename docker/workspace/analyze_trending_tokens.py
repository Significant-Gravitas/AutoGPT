# Use the DexGuru API to retrieve top trending tokens data
import requests

url = 'https://api-stage-lax.dex.guru/v2/tokens/trending'
headers = {'Content-type': 'application/json'}

payload = {'ids': [], 'network': 'eth,optimism,bsc,gnosis,polygon,fantom,zksync,canto,arbitrum,nova,celo,avalanche'}

response = requests.post(url, json=payload, headers=headers)

if response.status_code == 200:
    data = response.json()
    # data contains list of dict objects describing the trending tokens
    # EXTRACT INFO YOU WILL NEED TO SUGGEST INVESTMENTS HERE
    print(data)
else:
    print(f"An error occurred with status code {response.status_code}")