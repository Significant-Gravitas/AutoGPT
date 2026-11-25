import type { BrowserContext, Route } from "@playwright/test";
import { LATEST_RELEASES_URL } from "../../app/(no-navbar)/login/components/LoginWhatsNew/helpers";
import type { LatestRelease } from "../../app/api/changelog/helpers";
import {
  CHANGELOG_INDEX_MD_URL,
  STORAGE_KEY,
} from "../../components/molecules/ChangelogPopup/changelog-constants";
import { parseChangelogIndex } from "../../components/molecules/ChangelogPopup/helpers";

// The changelog routes read the published changelog from agpt.co, which the
// stack under test doesn't own, so the browser gets this index, in the
// published format, instead.
const CHANGELOG_INDEX = `
| Date | Highlights |
| ---- | ---------- |
| [June 18 – July 9](/docs/platform/changelog/changelog/june-18-july-9-2026.md) | File folders, shareable chats, Discord bot |
`;

const [LATEST] = parseChangelogIndex(CHANGELOG_INDEX);
if (!LATEST) {
  throw new Error("The changelog fixture no longer parses as an index");
}

// What the latest-releases route makes of that index. The route takes each
// title from the release's own page; this one stands in for it.
const LATEST_RELEASES: LatestRelease[] = [
  { ...LATEST, title: "File folders, shareable chats and a Discord bot" },
];

// The "What's new" popup opens over the chat composer's buttons until the
// browser has seen the latest entry, and a click's hover keeps it open.
export async function markChangelogSeen(context: BrowserContext) {
  await context.route(`**${CHANGELOG_INDEX_MD_URL}`, answerSeenIndex);
  await context.route(`**${LATEST_RELEASES_URL}`, (route) =>
    route.fulfill({ json: { releases: LATEST_RELEASES } }).catch(() => {}),
  );
}

// Marks the latest entry seen before the page reads the index, which it
// checks the marker against.
async function answerSeenIndex(route: Route) {
  await route
    .request()
    .frame()
    .evaluate(
      ([key, value]) => window.localStorage.setItem(key, value),
      [STORAGE_KEY, LATEST.slug],
    )
    .catch(() => {});
  // A test's context closing mid-answer can't be fulfilled.
  await route
    .fulfill({
      contentType: "text/markdown; charset=utf-8",
      body: CHANGELOG_INDEX,
    })
    .catch(() => {});
}
