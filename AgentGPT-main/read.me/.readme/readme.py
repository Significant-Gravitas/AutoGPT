url = 'https://api.github.com/repos/mshafae/agentgpt-ai4dbot/readme'
headers = {'Accept': 'application/vnd.github.VERSION.raw'}
response = requests.get(url, headers=headers)

readme = response.content.decode('utf-8')
print(readme)
