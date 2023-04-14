from duckduckgo_search import ddg
from selenium import webdriver
import autogpt.summary as summary
from bs4 import BeautifulSoup
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import os
import logging
from pathlib import Path
from autogpt.config import Config

file_dir = Path(__file__).parent
cfg = Config()


def browse_website(url, question):
    driver, text = scrape_text_with_selenium(url)
    add_header(driver)
    summary_text = summary.summarize_text(driver, text, question)
    links = scrape_links_with_selenium(driver)

    # Limit links to 5
    if len(links) > 5:
        links = links[:5]
    close_browser(driver)
    return f"Answer gathered from website: {summary_text} \n \n Links: {links}", driver


def scrape_text_with_selenium(url):
    logging.getLogger("selenium").setLevel(logging.CRITICAL)

    options = Options()
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.5615.49 Safari/537.36"
    )
    driver = webdriver.Chrome(
        executable_path=ChromeDriverManager().install(), options=options
    )
    driver.get(url)

    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )

    # Get the HTML content directly from the browser's DOM
    page_source = driver.execute_script("return document.body.outerHTML;")
    soup = BeautifulSoup(page_source, "html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = "\n".join(chunk for chunk in chunks if chunk)
    return driver, text


def scrape_links_with_selenium(driver):
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, "html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    hyperlinks = extract_hyperlinks(soup)

    return format_hyperlinks(hyperlinks)


def close_browser(driver):
    driver.quit()


def extract_hyperlinks(soup):
    return [(link.text, link["href"]) for link in soup.find_all("a", href=True)]


def format_hyperlinks(hyperlinks):
    return [f"{link_text} ({link_url})" for link_text, link_url in hyperlinks]


def add_header(driver):
    driver.execute_script(open(f"{file_dir}/js/overlay.js", "r").read())
