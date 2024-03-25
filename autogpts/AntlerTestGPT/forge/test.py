# import json
# import requests

# url = "https://google.serper.dev/search"

# query = "market demand for e-commerce niches"

# serper_api_key = "3acbcedda11205edc8a4b4e6aee38ef82a59f1fe79b8af6146176a85b1b22ca5"

# payload = json.dumps({
#     "q": query
# })
# headers = {
#     'X-API-KEY': serper_api_key,
#     'Content-Type': 'application/json'
# }
# response = requests.request("POST", url, headers=headers, data=payload)

# print(response.text)

from selenium import webdriver
from linkedin_scraper import Person, actions

driver = webdriver.Chrome()

email = "p1231pjw@gmail.com"
password = "rhkswkwo31!"
actions.login(driver, email, password) # if email and password isnt given, it'll prompt in terminal
person = Person("https://www.linkedin.com/in/jongwon-park-247692147/", driver=driver)
bio = person.about + str(person.experiences) + str(person.educations) + str(person.interests) + str(person.accomplishments)
