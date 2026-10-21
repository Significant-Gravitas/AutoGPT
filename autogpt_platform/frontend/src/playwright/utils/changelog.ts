import type { BrowserContext } from "@playwright/test";
import {
  CHANGELOG_INDEX_MD_URL,
  STORAGE_KEY,
} from "../../components/molecules/ChangelogPopup/changelog-constants";
import { parseChangelogIndex } from "../../components/molecules/ChangelogPopup/helpers";

// The "What's new" popup opens over the chat composer's buttons until the
// browser has seen the latest entry, and a click's hover keeps it open.
export async function markChangelogSeen(context: BrowserContext) {
  const slug = await latestChangelogSlug(context);
  if (!slug) return;
  await context.addInitScript(
    ({ key, value }) => {
      try {
        window.localStorage.setItem(key, value);
      } catch {
        // Opaque origins (about:blank, data:) have no storage.
      }
    },
    { key: STORAGE_KEY, value: slug },
  );
}

async function latestChangelogSlug(context: BrowserContext) {
  try {
    const response = await context.request.get(CHANGELOG_INDEX_MD_URL, {
      timeout: 15_000,
    });
    if (!response.ok()) return null;
    return parseChangelogIndex(await response.text())[0]?.slug ?? null;
  } catch {
    // The page's own fetch gets the same failure, and shows no popup.
    return null;
  }
}
