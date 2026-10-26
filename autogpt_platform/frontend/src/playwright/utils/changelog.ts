import type { BrowserContext, Route } from "@playwright/test";
import {
  CHANGELOG_INDEX_MD_URL,
  STORAGE_KEY,
} from "../../components/molecules/ChangelogPopup/changelog-constants";
import { parseChangelogIndex } from "../../components/molecules/ChangelogPopup/helpers";

// The "What's new" popup opens over the chat composer's buttons until the
// browser has seen the latest entry, and a click's hover keeps it open.
export async function markChangelogSeen(context: BrowserContext) {
  await context.route(`**${CHANGELOG_INDEX_MD_URL}`, markSeenOnTheWay);
}

// Rides the page's own request, so a slow changelog delays only the popup's
// check, never a test; the page reads the marker after this response lands.
async function markSeenOnTheWay(route: Route) {
  let response;
  try {
    response = await route.fetch();
  } catch {
    await route.abort().catch(() => {});
    return;
  }
  const [latest] = response.ok()
    ? parseChangelogIndex(await response.text())
    : [];
  if (latest) {
    await route
      .request()
      .frame()
      .evaluate(
        ([key, value]) => window.localStorage.setItem(key, value),
        [STORAGE_KEY, latest.slug],
      )
      .catch(() => {});
  }
  await route.fulfill({ response }).catch(() => {});
}
