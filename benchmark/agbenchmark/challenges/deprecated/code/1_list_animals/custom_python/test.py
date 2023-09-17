import os
import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# Path to the HTML file
current_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_path)
file_path = f"file://{current_directory}/animal_list.html"

# Create a new instance of the Chrome driver

chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--window-size=1024x768")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
driver = webdriver.Chrome(options=chrome_options)

# Navigate to the HTML file
driver.get(file_path)

# Wait for up to 10 seconds for the "dog" element to be available
wait = WebDriverWait(driver, 10)
dog_li = wait.until(EC.presence_of_element_located((By.ID, "dog")))

# Click on the "dog" list item
dog_li.click()

# Find the "info" div and get its text
info_div = driver.find_element(By.ID, "info")
info_text = info_div.text

# Assert that the text is what we expect
assert info_text == "Dogs are known as man's best friend!"

print(" passed!")

# Wait for 5 seconds
time.sleep(5)

# Close the browser window
driver.quit()
