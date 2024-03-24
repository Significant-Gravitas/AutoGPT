"""HTML processing functions"""
from __future__ import annotations

import random
import re

from bs4 import BeautifulSoup
from requests.compat import urljoin


def extract_tag_links(
    soup: BeautifulSoup,
    base_url: str,
    tag_type="a",
    include_keywords: list[str] = [],
    exclude_keywords: list[str] = [],
) -> list[tuple[str, str]]:
    """Extract image links and hyperlinks from a BeautifulSoup object

    Args:
        soup (BeautifulSoup): The BeautifulSoup object
        base_url (str): The base URL
        link_attr (str, optional): The attribute to extract from the tag. Defaults to "href".
        include_keywords (List[str], optional): List of keyword to include in the results. Defaults to [].
        exclude_keywords (List[str], optional): List of keyword to exclude from the results. Defaults to [].

    Returns:
        list[Tuple[str, str]]: The extracted links
    """

    exclude_full_keywords = ["blank", "transparent"]

    def contains_keyword(link_value: str, keywords: list[str]) -> bool:
        return any(keyword in link_value for keyword in keywords)

    def equals_keyword(link_value: str, keywords: list[str]) -> bool:
        return any(
            re.search(f"\\b{keyword}\\b", link_value, re.I) for keyword in keywords
        )

    def get_link_text(tag_soup) -> str:
        if tag_type == "img":
            return tag_soup.get("alt") or tag_soup.get("title", "")
        elif tag_type == "a":
            return tag_soup.text
        else:
            return ""

    link_attr = "src" if tag_type == "img" else "href"

    links = [
        (get_link_text(tag), urljoin(base_url, tag[link_attr]))
        for tag in soup.find_all(tag_type, {link_attr: True})
        if (not include_keywords or contains_keyword(tag[link_attr], include_keywords))
        and not contains_keyword(tag[link_attr], exclude_keywords)
        and not equals_keyword(tag[link_attr], exclude_full_keywords)
    ]

    random.shuffle(links)

    return links


def format_links(links: list[tuple[str, str]]) -> list[str]:
    """Format links to be displayed to the user

    Args:
        links (List[Tuple[str, str]]): The links to format

    Returns:
        List[str]: The first 50 formatted links
    """
    formatted_links = list(
        set([f"{link_text.strip()} ({link_url})" for link_text, link_url in links])
    )
    if len(formatted_links) > 50:
        formatted_links = formatted_links[:50]
    return formatted_links
