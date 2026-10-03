"""docs/platform is part of the public docs site, so it holds only published pages.

GitBook publishes a page once its folder's SUMMARY.md lists it. A file that sits
in a published folder without being listed is one line away from going public,
which is how internal notes end up in the site navigation. Notes that are not
for the site live in docs/engineering instead.
"""

import posixpath
import re
import unittest
from pathlib import Path
from urllib.parse import unquote

DOCS = Path(__file__).resolve().parents[3] / "docs"
PUBLISHED = ("home", "platform", "integrations")
# A Markdown link target, written bare or as <a path with spaces>.
SUMMARY_LINK = re.compile(r"\]\((?:<([^>#]+\.md)[^>]*>|([^)#\s]+\.md))")
HAS_SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*:", re.IGNORECASE)


def listed_pages(folder: Path) -> set:
    """Paths of the local pages a folder's SUMMARY.md lists, relative to it."""
    summary = (folder / "SUMMARY.md").read_text(encoding="utf-8")
    links = (bracketed or bare for bracketed, bare in SUMMARY_LINK.findall(summary))
    return {
        posixpath.normpath(unquote(link))
        for link in links
        if not HAS_SCHEME.match(link)
    }


class DocsLayoutTests(unittest.TestCase):
    def test_every_platform_page_is_in_the_summary(self):
        platform = DOCS / "platform"
        listed = listed_pages(platform)
        unlisted = sorted(
            page.relative_to(platform).as_posix()
            for page in platform.rglob("*.md")
            if page.name != "SUMMARY.md"
            and page.relative_to(platform).as_posix() not in listed
        )

        self.assertEqual(
            unlisted,
            [],
            "These files are in docs/platform but not in docs/platform/SUMMARY.md. "
            "If a file is a page for the docs site, list it there. "
            "If it is an engineering note, move it to docs/engineering.",
        )

    def test_summaries_list_only_pages_in_their_own_folder(self):
        for name in PUBLISHED:
            folder = DOCS / name
            for link in sorted(listed_pages(folder)):
                page = (folder / link).resolve()
                with self.subTest(summary=f"docs/{name}/SUMMARY.md", link=link):
                    self.assertTrue(
                        folder.resolve() in page.parents,
                        "A SUMMARY.md publishes what it lists, so it may only "
                        f"list pages inside docs/{name}.",
                    )
                    self.assertTrue(page.is_file(), "The listed page does not exist.")


class ListedPagesTests(unittest.TestCase):
    def listed(self, summary: str) -> set:
        folder = MagicFolder(summary)
        return listed_pages(folder)

    def test_reads_bare_bracketed_and_encoded_links(self):
        self.assertEqual(
            self.listed(
                "* [A](a.md)\n"
                "  * [B](./sub/b.md#section)\n"
                "* [C](<with space.md>)\n"
                "* [D](with%20percent.md)\n"
            ),
            {"a.md", "sub/b.md", "with space.md", "with percent.md"},
        )

    def test_ignores_links_to_other_sites(self):
        self.assertEqual(
            self.listed("* [Elsewhere](https://example.com/page.md)\n"), set()
        )


class MagicFolder:
    """A folder whose SUMMARY.md has the given text."""

    def __init__(self, summary: str) -> None:
        self.summary = summary

    def __truediv__(self, name: str) -> "MagicFolder":
        return self

    def read_text(self, encoding: str) -> str:
        return self.summary


if __name__ == "__main__":
    unittest.main()
