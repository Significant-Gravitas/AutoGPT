"""docs/platform is part of the public docs site, so it holds only published pages.

GitBook publishes a page once its folder's SUMMARY.md lists it. A file that sits
in a published folder without being listed is one line away from going public,
which is how internal notes end up in the site navigation. Notes that are not
for the site live in docs/engineering instead.
"""

import posixpath
import re
import tempfile
import unittest
from pathlib import Path
from urllib.parse import unquote

DOCS = Path(__file__).resolve().parents[3] / "docs"
PUBLISHED = ("home", "platform", "integrations")
# A Markdown link target that is a .md file, written bare or as
# <a path with spaces>. The target has to end at ".md": "page.md.old" is not one.
SUMMARY_LINK = re.compile(r"\]\((?:<([^>#]+\.md)(?:#[^>]*)?>|([^)#\s]+\.md)(?=[)#\s]))")
HAS_SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*:", re.IGNORECASE)
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


def listed_pages(folder: Path) -> set:
    """Paths of the local pages a folder's SUMMARY.md lists, relative to it."""
    summary = (folder / "SUMMARY.md").read_text(encoding="utf-8")
    # A commented-out entry is not in the navigation, so it lists nothing.
    summary = HTML_COMMENT.sub("", summary)
    links = (bracketed or bare for bracketed, bare in SUMMARY_LINK.findall(summary))
    return {
        posixpath.normpath(unquote(link))
        for link in links
        if not HAS_SCHEME.match(link)
    }


def unlisted_pages(folder: Path) -> list:
    """Markdown files in a folder that its SUMMARY.md does not list.

    Files under .gitbook are GitBook's own (reusable content it includes into
    pages), not pages, and are never listed.
    """
    listed = listed_pages(folder)
    return sorted(
        page.relative_to(folder).as_posix()
        for page in folder.rglob("*.md")
        if page != folder / "SUMMARY.md"
        and ".gitbook" not in page.relative_to(folder).parts
        and page.relative_to(folder).as_posix() not in listed
    )


class DocsLayoutTests(unittest.TestCase):
    def test_every_platform_page_is_in_the_summary(self):
        self.assertEqual(
            unlisted_pages(DOCS / "platform"),
            [],
            "These files are in docs/platform but not in docs/platform/SUMMARY.md. "
            "docs/platform is the public docs site, so move them to "
            "docs/engineering. Only list a file in SUMMARY.md, which publishes "
            "it, if it is a page meant for the site. See docs/AGENTS.md.",
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


class SummaryParsingTests(unittest.TestCase):
    def folder(self, summary: str, *files: str) -> Path:
        """A temporary folder with the given SUMMARY.md and empty files."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        folder = Path(directory.name)
        (folder / "SUMMARY.md").write_text(summary, encoding="utf-8")
        for name in files:
            (folder / name).parent.mkdir(parents=True, exist_ok=True)
            (folder / name).write_text("", encoding="utf-8")
        return folder

    def test_reads_bare_bracketed_and_encoded_links(self):
        folder = self.folder(
            "* [A](a.md)\n"
            "  * [B](./sub/b.md#section)\n"
            "* [C](<with space.md>)\n"
            "* [D](with%20percent.md)\n"
            '* [E](titled.md "A title")\n'
            "* [F](<bracketed.md#section>)\n"
        )

        self.assertEqual(
            listed_pages(folder),
            {
                "a.md",
                "sub/b.md",
                "with space.md",
                "with percent.md",
                "titled.md",
                "bracketed.md",
            },
        )

    def test_ignores_links_to_other_sites(self):
        folder = self.folder("* [Elsewhere](https://example.com/page.md)\n")

        self.assertEqual(listed_pages(folder), set())

    def test_target_must_end_at_the_md_extension(self):
        """Listing page.md.old must not count as listing page.md."""
        folder = self.folder(
            "* [Old](page.md.old)\n* [Old](<page.md.old>)\n", "page.md"
        )

        self.assertEqual(listed_pages(folder), set())
        self.assertEqual(unlisted_pages(folder), ["page.md"])

    def test_commented_out_entry_lists_nothing(self):
        folder = self.folder(
            "* [A](a.md)\n<!-- * [B](b.md) -->\n<!--\n* [C](c.md)\n-->\n",
            "a.md",
            "b.md",
            "c.md",
        )

        self.assertEqual(listed_pages(folder), {"a.md"})
        self.assertEqual(unlisted_pages(folder), ["b.md", "c.md"])

    def test_gitbook_reusable_content_is_not_a_page(self):
        folder = self.folder(
            "* [A](a.md)\n", "a.md", ".gitbook/includes/snippet.md", "sub/b.md"
        )

        self.assertEqual(unlisted_pages(folder), ["sub/b.md"])

    def test_only_the_folders_own_summary_is_exempt(self):
        folder = self.folder(
            "* [A](a.md)\n", "a.md", "b.md", "sub/SUMMARY.md", "sub/c.md"
        )

        self.assertEqual(unlisted_pages(folder), ["b.md", "sub/SUMMARY.md", "sub/c.md"])


if __name__ == "__main__":
    unittest.main()
