// Public docs pages the user can open (used for "open in docs" links).
export const CHANGELOG_BASE_URL =
  "https://agpt.co/docs/platform/changelog/changelog";
// The docs site 503s cross-origin browser fetches, so the markdown is fetched
// through our own same-origin proxy (src/app/api/changelog/route.ts).
export const CHANGELOG_PROXY_URL = "/api/changelog";
export const CHANGELOG_INDEX_MD_URL = CHANGELOG_PROXY_URL;
export const STORAGE_KEY = "autogpt-changelog-last-seen";
export const AUTO_DISMISS_MS = 8000;
