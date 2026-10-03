"""docs/platform is part of the public docs site, so it holds only published pages.

GitBook publishes a page once its folder's SUMMARY.md lists it. A file that sits
in a published folder without being listed is one line away from going public,
which is how internal notes end up in the site navigation. Notes that are not
for the site live in docs/engineering instead.
"""

import re
import unittest
from pathlib import Path

DOCS = Path(__file__).resolve().parents[3] / "docs"
PUBLISHED = ("home", "platform", "integrations")
SUMMARY_LINK = re.compile(r"\]\(<?([^)>#\s]+\.md)")


def listed_pages(folder: Path) -> set:
    summary = (folder / "SUMMARY.md").read_text(encoding="utf-8")
    return set(SUMMARY_LINK.findall(summary))


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


if __name__ == "__main__":
    unittest.main()
