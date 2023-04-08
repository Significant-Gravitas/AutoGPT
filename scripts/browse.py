
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from time import sleep

options = Options()
options.add_argument('--headless')

lastFetched = None

def fetch_url(url):
    browser = webdriver.Chrome(options=options)
    browser.get(url)

    # Wait for page to load
    # browser.implicitly_wait(10)
    sleep(5)

    # Use a more targeted XPath expression to select only elements that are likely to have meaningful text content
    xpath = "//*[self::h1 or self::h2 or self::h3 or self::h4 or self::h5 or self::h6 or self::p or self::a or self::li or self::span or self::a or self::button]"

    # Find all elements on the page
    elements = browser.find_elements(By.XPATH, xpath)

    # Extract the text content from the elements
    text_content = []
    for element in elements:
        # if element is button or link, we must include the URL
        if element.tag_name == "a" or element.tag_name == "button":
            text = element.text
            url = element.get_attribute("href")
            if text and text != "" and url and url.startswith("http"):
                text_content.append("(" + text + ")[" + url + "]")
        else:
          # Otherwise, just include the text
          text = element.text
          if text and (len(text_content) == 0 or text_content[len(text_content) - 1] != text):
              text_content.append(text)

    # Close browser
    browser.quit()

    # Build content
    content = ' '.join(text_content)

    # Store content
    global lastFetched
    lastFetched = content

def split_text(text, max_length=2048):
    # Split text into chunks of max_length
    chunks = []
    while len(text) > max_length:
        # Find the last space before the max length
        last_space = text.rfind(" ", 0, max_length)
        if last_space == -1:
            # If there is no space, just split at the max length
            last_space = max_length
        chunks.append(text[0:last_space])
        text = text[last_space + 1:]
    chunks.append(text)
    return chunks

def has_fetched():
  return lastFetched != None

def view_page(pageNumber):
  if lastFetched:
    chunks = split_text(lastFetched)
    if int(pageNumber) > len(list(chunks)):
      return "Page number out of range."

    header = "Page " + str(int(pageNumber) + 1) + " of " + str(len(list(chunks))) + ":\n"
    return header + list(chunks)[int(pageNumber)]
  else:
    return "No page fetched yet."