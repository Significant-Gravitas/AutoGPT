"""Selenium web scraping module."""
from __future__ import annotations

import logging
from pathlib import Path
from sys import platform

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.safari.options import Options as SafariOptions
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager

import autogpt.processing.text as summary
from autogpt.commands.command import command
from autogpt.config import Config
from autogpt.processing.html import extract_hyperlinks, format_hyperlinks
from autogpt.url_utils.validators import validate_url

from autogpt.llm.llm_utils import create_chat_completion
from autogpt.llm.token_counter import count_message_tokens

from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

import trafilatura
import re
from urllib.parse import urlparse, urljoin

FILE_DIR = Path(__file__).parent.parent
CFG = Config()
URL_MEMORY = {
    
}


@command(
    "search_website_and_extract_related_links",
    "Search website and extract related links",
    '"url": "<url>", "question": "<question_to_find>"',
)
@validate_url
def search_website_and_extract_related_links(url: str, question: str) -> str:    
    """Browse a website and return the hyperlinks related to the question

    Args:
        url (str): The url of the website to browse
        question (str): The question asked by the user

    Returns:
        str: The answer and links to the user
    """
    global URL_MEMORY
    if url in URL_MEMORY: url = URL_MEMORY[url]

    # qestion을 해결하기 위해 search가 필요할 경우 search_url로 변경합니다.    
    request_msg = f"""
        search_url is used in mobile web browsers.
        "input_url": "{url}"
        "question": "{question}"
        If a search is needed to solve it, 
        {{"search_url": "..."}}
        If a seach is not needed, please answer as below.
        {{"search_url": None}}"""
    """resp = create_chat_completion(
        model=CFG.smart_llm_model,
        messages=[{"role":"user", "content":request_msg}])
    try:
        resp = eval(resp)
        search_url = resp['search_url']
        print(f'search_url:{search_url}')
        if search_url:
            url = search_url
    except:
        pass
    """
    
    try:
        html_content, driver = get_html_content_with_selenium(url)
    except WebDriverException as e:
        msg = e.msg.split("\n")[0]
        return f"Error: {msg}", None
    
    text_link_pairs = []
    text_link_pairs.extend(get_header_text_link_pairs(html_content, url))
    text_link_pairs.extend(get_main_content_text_llink_pairs(html_content))
    text, _ = zip(*text_link_pairs)
    #add_header(driver)
    solvable_msg = is_question_solvable_using_text(text, question)    
    return_msg = get_links_related_to_question_with_chat(text_link_pairs, question)
    close_browser(driver)

    return solvable_msg + ' ' + return_msg


@command(
    "get_website_text_summary",
    "Get website text text summary",
    '"url": "<url>", "question": "<question>"',
)
def get_website_text_summary(url: str, question: str, max_len=3500) -> str:
    global URL_MEMORY
    if url in URL_MEMORY:
        print(url, URL_MEMORY[url])
        url = URL_MEMORY[url]
    try:
        html_content, driver = get_html_content_with_selenium(url)
    except WebDriverException as e:
        # These errors are often quite long and include lots of context.
        # Just grab the first line.
        msg = e.msg.split("\n")[0]
        return f"Error: {msg}"

    #add_header(driver)
    #scroll_to_percentage(driver, scroll_ratio * i)
    #summary_text = summary.summarize_text(url, text, driver)
    text = trafilatura.extract(html_content, include_formatting=True, favor_recall=False)
    
    text = text[:max_len]
    main_lang = get_main_language(text)
    request_msg = (        
        f'Please summarize the website text in {main_lang}, with reference to "{question}":\n\n'
        f'{text}'        
    )
    resp = create_chat_completion(
        model=CFG.fast_llm_model,
        messages=[{"role":"user", "content":request_msg}])
    return f'Website summary: {resp}  '


def get_header_text_link_pairs(html_content, base_url='http:'):
    # BeautifulSoup 객체 생성
    soup = BeautifulSoup(html_content, 'html.parser')

    # 이름이 'header'를 포함하거나, 'id'가 'header'를 포함하는 모든 태그 찾기
    header_tags = soup.find_all(lambda tag: 'header' in tag.name or ('header' in tag.get('id', '')))

    # 태그의 텍스트, href 속성 값
    text_link_pairs = set()  # Set으로 변경
    for header in header_tags:
        for descendant in header.descendants:
            if descendant.name == 'a' and descendant.get('href'):  # descendant가 a 태그이고 href 속성을 가지고 있으면
                url = urljoin(base_url, descendant['href'])  # 상대 URL을 절대 URL로 변환
                text_link_pairs.add((f"menu: {descendant.get_text(strip=True)}", f"{url}"))  # append 대신 add 사용
            
    return list(text_link_pairs)  # 최종적으로 다시 리스트로 변환


def get_main_content_text_llink_pairs(html_content):
    text = trafilatura.extract(html_content, include_links=True, include_formatting=True, favor_recall=True,
                         output_format='txt')
    # '[...](...)' 형식의 텍스트 추출
    # 정규표현식 패턴 정의
    pattern = r'\[(.*?)\]\((.*?)\)'

    # 텍스트에서 패턴에 맞는 부분 찾기
    matches = re.findall(pattern, text)

    # 텍스트와 링크를 담을 리스트 초기화
    t_link_pairs = set()

    # 각 매치에 대해
    for match in matches:
        # match는 (텍스트, 링크) 형태의 튜플
        t, link = match
        if t != '':
            t_link_pairs.add((t, link))

    return list(t_link_pairs)


def get_main_language(text):
    try:
        language = detect(text)
    except LangDetectException:
        language = "unknown"

    return language


def summarize_text_with_question(text: str, question: str, max_len=3500) -> str:
    text = text[:max_len]
    #request_msg = (
    #    'The following text is extracted from a webpage. Please briefly answer whether it contains the necessary information to resolve the question.\n\n'
    #    f'{text} \n\n'
    #    f'-- question:{question}'
    #)
    request_msg = (f'"""{text}""" Using the above text, answer the following'
        f' question: "{question}" -- if the question cannot be answered using the text,'
        " summarize the text. Please output in the language used in the above text."
    )
    
    resp = create_chat_completion(
        model=CFG.fast_llm_model,
        messages=[{"role":"user", "content":request_msg}])
    return f'Website summary: {resp}  '

def is_question_solvable_using_text(text: str, question: str, max_len=3500) -> str:
    text = text[:max_len]    
    request_msg = (f'"""{text}""" The above website text is extracted from the hyperlink, is it possible to answer the following question?'
        f' question: "{question}"'        
    )
    
    resp = create_chat_completion(
        model=CFG.fast_llm_model,
        messages=[{"role":"user", "content":request_msg}])
    return f'is_question_solvable_using_text: {resp}  '
    
#@command(
#    "get_hyperlinks_related_question",
#    "Get hyperlinks related to question",
#    '"url": "<url>", "question": "<what_you_want_hyperlinks_from_website>"',
#)
@validate_url
def get_hyperlinks_related_to_question(url: str, question: str) -> str:
    """Browse a website and return the hyperlinks related to the question

    Args:
        url (str): The url of the website to browse
        question (str): The question asked by the user

    Returns:
        Tuple[str, WebDriver]: The answer and links to the user and the webdriver
    """
    global URL_MEMORY
    if url in URL_MEMORY: url = URL_MEMORY[url]
    try:
        driver, text = scrape_text_with_selenium(url)
    except WebDriverException as e:
        msg = e.msg.split("\n")[0]
        return f"Error: {msg}", None

    add_header(driver)
    links = scrape_links_with_selenium(driver, url)
    return_msg = get_links_related_to_question_with_chat(links, question)
    close_browser(driver)

    return return_msg


def get_links_related_to_question_with_chat(links: list[tuple[str, str]], question: str) -> str:
    global URL_MEMORY
    link_texts, hyperlinks = zip(*links)
    cleaned_text = []
    for i, sent in enumerate(link_texts):
        sent = " ".join(sent.split())
        if len(sent) == 0: continue
        if len(sent)>20:
            sent = sent[:20] + '...'
        cleaned_text.append(f'{i}: {sent}')     
    cleaned_text = cleaned_text[:3500]
    #ntokens = count_message_tokens([{"role": "user", "content": cleaned_text}], CFG.smart_llm_model)

    request_msg = (
        'The following is the result of extracting only the text from the hyperlink and attaching a line_number to it. \n'
        f'{cleaned_text}\n'
        'Please return all appropriate line_numbers only for the following request:\n\n'
        f'{question}'
        #'If question has a specific request for the number of items, please extract 2 more.'
    )
    messages = [{"role": "user", "content": request_msg}]

    resp = create_chat_completion(model=CFG.smart_llm_model, messages=messages)

    try:
        line_numbers = eval(resp)
        line_numbers = [line_numbers] if isinstance(line_numbers, int) else line_numbers
    except:
        line_numbers = [int(w) for w in resp.split() if w.isdigit()]

    selected_links = []
    
    if line_numbers:
        for i in line_numbers:
            link = links[i][1]
            link_nick = f'URL_{len(URL_MEMORY)}'   
            URL_MEMORY[link_nick] = link
            selected_links.append(f"{cleaned_text[i]} ({link_nick})")
            
        return_msg = f"Links: {selected_links}"
    else:
        return_msg = "Links: Couldn't find any links."

    return return_msg


def get_html_content_with_selenium(url: str) -> tuple[str, str]:
    
    logging.getLogger("selenium").setLevel(logging.CRITICAL)

    options_available = {
        "chrome": ChromeOptions,
        "safari": SafariOptions,
        "firefox": FirefoxOptions,
    }

    options = options_available[CFG.selenium_web_browser]()
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.5615.49 Safari/537.36"
    )

    if CFG.selenium_web_browser == "firefox":
        if CFG.selenium_headless:
            options.headless = True
            options.add_argument("--disable-gpu")
        driver = webdriver.Firefox(
            executable_path=GeckoDriverManager().install(), options=options
        )
    elif CFG.selenium_web_browser == "safari":
        # Requires a bit more setup on the users end
        # See https://developer.apple.com/documentation/webkit/testing_with_webdriver_in_safari
        driver = webdriver.Safari(options=options)
    else:
        if platform == "linux" or platform == "linux2":
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--remote-debugging-port=9222")

        options.add_argument("--no-sandbox")
        if CFG.selenium_headless:
            options.add_argument("--headless=new")
            options.add_argument("--disable-gpu")
        
        # 모바일 버전으로 쓰기 위해서 추가
        if 0:
            user_agt = 'Mozilla/5.0 (Linux; Android 9; INE-LX1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Mobile Safari/537.36'
            options.add_argument(f'user-agent={user_agt}')
            options.add_argument("window-size=412,950")
            options.add_experimental_option("mobileEmulation",
                                            {"deviceMetrics": {"width": 360,
                                                            "height": 760,
                                                            "pixelRatio": 3.0}})

        chromium_driver_path = Path("/usr/bin/chromedriver")

        driver = webdriver.Chrome(
            executable_path=chromium_driver_path
            if chromium_driver_path.exists()
            else ChromeDriverManager().install(),
            options=options,
        )
    driver.get(url)

    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )

    # Get the HTML content directly from the browser's DOM
    page_source = driver.execute_script("return document.body.outerHTML;")
    #soup = BeautifulSoup(page_source, "html.parser")
    return page_source, driver


def scrape_text_with_selenium(url: str) -> tuple[WebDriver, str]:
    """Scrape text from a website using selenium

    Args:
        url (str): The url of the website to scrape

    Returns:
        Tuple[WebDriver, str]: The webdriver and the text scraped from the website
    """
    logging.getLogger("selenium").setLevel(logging.CRITICAL)

    options_available = {
        "chrome": ChromeOptions,
        "safari": SafariOptions,
        "firefox": FirefoxOptions,
    }

    options = options_available[CFG.selenium_web_browser]()
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.5615.49 Safari/537.36"
    )

    if CFG.selenium_web_browser == "firefox":
        if CFG.selenium_headless:
            options.headless = True
            options.add_argument("--disable-gpu")
        driver = webdriver.Firefox(
            executable_path=GeckoDriverManager().install(), options=options
        )
    elif CFG.selenium_web_browser == "safari":
        # Requires a bit more setup on the users end
        # See https://developer.apple.com/documentation/webkit/testing_with_webdriver_in_safari
        driver = webdriver.Safari(options=options)
    else:
        if platform == "linux" or platform == "linux2":
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--remote-debugging-port=9222")

        options.add_argument("--no-sandbox")
        if CFG.selenium_headless:
            options.add_argument("--headless=new")
            options.add_argument("--disable-gpu")
        
        # 모바일 버전으로 쓰기 위해서 추가
        user_agt = 'Mozilla/5.0 (Linux; Android 9; INE-LX1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Mobile Safari/537.36'
        options.add_argument(f'user-agent={user_agt}')
        options.add_argument("window-size=412,950")
        options.add_experimental_option("mobileEmulation",
                                        {"deviceMetrics": {"width": 360,
                                                        "height": 760,
                                                        "pixelRatio": 3.0}})

        chromium_driver_path = Path("/usr/bin/chromedriver")

        driver = webdriver.Chrome(
            executable_path=chromium_driver_path
            if chromium_driver_path.exists()
            else ChromeDriverManager().install(),
            options=options,
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


def scrape_links_with_selenium(driver: WebDriver, url: str) -> list[str]:
    """Scrape links from a website using selenium

    Args:
        driver (WebDriver): The webdriver to use to scrape the links

    Returns:
        List[str]: The links scraped from the website
    """
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, "html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    hyperlinks = extract_hyperlinks(soup, url)

    return hyperlinks


def close_browser(driver: WebDriver) -> None:
    """Close the browser

    Args:
        driver (WebDriver): The webdriver to close

    Returns:
        None
    """
    driver.quit()


def add_header(driver: WebDriver) -> None:
    """Add a header to the website

    Args:
        driver (WebDriver): The webdriver to use to add the header

    Returns:
        None
    """
    try:
        with open(f"{FILE_DIR}/js/overlay.js", "r") as overlay_file:
            overlay_script = overlay_file.read()
        driver.execute_script(overlay_script)
    except Exception as e:
        print(f"Error executing overlay.js: {e}")