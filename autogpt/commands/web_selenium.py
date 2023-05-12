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
import pycountry

import trafilatura
import re
from urllib.parse import urlparse, urljoin

FILE_DIR = Path(__file__).parent.parent
CFG = Config()
URL_MEMORY = {}


@command(
    "browse_website",
    "Browse website and extract related links",
    '"url": "<url>", "question": "<what_you_want_to_find_on_website>"',
)
def browse_website(url: str, question: str) -> str:    
    """Browse a website and return the hyperlinks related to the question

    Args:
        url (str): The url of the website to browse
        question (str): The question asked by the user

    Returns:
        str: The answer and links to the user
    """    
    global URL_MEMORY
    if url in URL_MEMORY: 
        #print(url, '->', URL_MEMORY[url])
        url = URL_MEMORY[url]    
    
    try:
        html_content, driver = get_html_content_with_selenium(url)
    except WebDriverException as e:
        msg = e.msg.split("\n")[0]
        return f"Error: {msg}", None
    
    text_link_pairs = []
    text_link_pairs.extend(get_header_text_link_pairs(html_content, url))
    text_link_pairs.extend(get_main_content_text_link_pairs(html_content))
    return_msg = get_links_related_to_question_with_chat(text_link_pairs, question)
    close_browser(driver)

    #print('URL_MEMORY_STATE', URL_MEMORY)    
    
    return return_msg
    #return return_msg


@command(
    "get_webpage_text_summary",
    "Get webpage text summary",
    '"url": "<url>", "question": "<question>"',
)
def get_webpage_text_summary(url: str, question: str, max_len=3500) -> str:
    global URL_MEMORY
    if url in URL_MEMORY:
        #print(url, URL_MEMORY[url])
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
        f'Please summarize the webpage text in {main_lang}", with reference to "{question}":\n\n'
        f'{text}'        
    )
    resp = create_chat_completion(
        model=CFG.fast_llm_model,
        messages=[{"role":"user", "content":request_msg}])
    return f'Webpage summary: {resp}  '


def get_header_text_link_pairs(html_content, base_url='http:'):
    # BeautifulSoup 객체 생성
    soup = BeautifulSoup(html_content, 'html.parser')

    # 이름이 'header'를 포함하거나, 'id'가 'header'를 포함하는 모든 태그 찾기
    header_tags = soup.find_all(lambda tag: 'header' in tag.name or ('header' in tag.get('id', '')))

    # 태그의 텍스트, href 속성 값
    text_link_pairs = []
    for header in header_tags:
        for descendant in header.descendants:
            if descendant.name == 'a' and descendant.get('href'):  # descendant가 a 태그이고 href 속성을 가지고 있으면
                url = urljoin(base_url, descendant['href'])  # 상대 URL을 절대 URL로 변환
                pair = (f"menu: {descendant.get_text(strip=True)}", f"{url}")
                if pair not in text_link_pairs:
                    text_link_pairs.append(pair)  # append 대신 add 사용
            
    return list(text_link_pairs)  # 최종적으로 다시 리스트로 변환


def get_main_content_text_link_pairs(html_content):
    text = trafilatura.extract(html_content, include_links=True, include_formatting=True, favor_recall=True,
                         output_format='txt')
    
    # Text를 라인 단위로 분리
    lines = text.split('\n')

    # '[...](...)' 형식의 텍스트 추출
    # 정규표현식 패턴 정의
    pattern = r'\[(.*?)\]\((.*?)\)'
    
    # 텍스트와 링크를 담을 리스트 초기화
    t_link_pairs = []
    
    # 각 라인에 대해
    for line in lines:
        # 텍스트에서 패턴에 맞는 부분 찾기
        matches = re.findall(pattern, line)        

        # 각 매치에 대해, 가장 첫번째 것만 가져온다.
        for match in matches:
            # match는 (텍스트, 링크) 형태의 튜플
            t, link = match
            if t != '':
                # 링크의 유효성 검사
                parsed_link = urlparse(link)
                if parsed_link.scheme and parsed_link.netloc and match not in t_link_pairs:
                    t_link_pairs.append(match)
    return t_link_pairs


def get_main_language(text):
    try:
        lang_code = detect(text)
    except LangDetectException:
        lang_code = "unknown"

    try:
        language = pycountry.languages.get(alpha_2=lang_code).name
    except AttributeError:
        language = "Unknown language"

    return language


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
    text = "\n".join(cleaned_text)
    text = text[:3500]
    #ntokens = count_message_tokens([{"role": "user", "content": cleaned_text}], CFG.smart_llm_model)

    request_msg = (
        f'{text}'
        '\n\nThe text(menus, contents) above is extracted from hyperlinks found on a webpage. '
        f'\n\nFor the question: "{question}",'
        '\nPlease respond like as following:'
        '\n{"related_line_numbers":[], "if_no_related_line_number_which_menu_to_click_for_retry": []}'
    )
    messages = [{"role": "user", "content": request_msg}]

    resp = create_chat_completion(model=CFG.smart_llm_model, messages=messages)    

    try:
        resp = eval(resp)
        line_numbers = resp['related_line_numbers']
        if len(line_numbers) == 0:
            line_numbers = resp['if_no_related_line_number_which_menu_to_click_for_retry']
        line_numbers = [line_numbers] if isinstance(line_numbers, int) else line_numbers
    except:
        line_numbers = [int(w) for w in resp.split() if w.isdigit()]

    selected_links = []
    
    if line_numbers:
        for i in line_numbers:
            link = links[i][1]
            link_nick = f'URL_{len(URL_MEMORY)}'   
            URL_MEMORY[link_nick] = link
            selected_links.append(f"{cleaned_text[i]} (url:{link_nick})")
            
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