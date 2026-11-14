import re

from pydantic import BaseModel

MAX_REFERENCE_ABSTRACT_CHARS = 900
WHITESPACE_PATTERN = re.compile(r"\s+")


class ArxivReference(BaseModel):
    title: str
    authors: list[str]
    published: str
    url: str
    pdf_url: str
    summary: str
    categories: list[str]


def format_references(references: list[ArxivReference]) -> str:
    return "\n\n".join(
        format_reference(index, reference)
        for index, reference in enumerate(references, start=1)
    )


def format_reference(index: int, reference: ArxivReference) -> str:
    authors = ", ".join(reference.authors) or "Unknown authors"
    categories = ", ".join(reference.categories) or "uncategorized"
    summary = reference.summary[:MAX_REFERENCE_ABSTRACT_CHARS]
    return (
        f"[{index}] {reference.title}\n"
        f"Authors: {authors}\n"
        f"Published: {reference.published}\n"
        f"Categories: {categories}\n"
        f"URL: {reference.url}\n"
        f"PDF: {reference.pdf_url}\n"
        f"Abstract: {summary}"
    )


def reference_from_feed_entry(entry) -> ArxivReference:
    return ArxivReference(
        title=clean_text(entry.get("title", "")),
        authors=entry_authors(entry),
        published=clean_text(entry.get("published", "")),
        url=clean_text(entry.get("id", "")),
        pdf_url=entry_pdf_url(entry),
        summary=clean_text(entry.get("summary", "")),
        categories=[
            clean_text(tag.get("term", ""))
            for tag in entry.get("tags", [])
            if clean_text(tag.get("term", ""))
        ],
    )


def entry_authors(entry) -> list[str]:
    authors = [
        clean_text(author.get("name", ""))
        for author in entry.get("authors", [])
        if clean_text(author.get("name", ""))
    ]
    if authors:
        return authors

    author = clean_text(entry.get("author", ""))
    return [author] if author else []


def entry_pdf_url(entry) -> str:
    pdf_urls = [
        clean_text(link.get("href", ""))
        for link in entry.get("links", [])
        if link.get("type") == "application/pdf"
    ]
    if pdf_urls:
        return pdf_urls[0]
    return clean_text(entry.get("link", ""))


def clean_text(value: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", value).strip()
