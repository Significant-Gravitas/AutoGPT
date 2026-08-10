// The full changelog on the public docs site (opened via "View changelog").
export const CHANGELOG_BASE_URL =
  "https://agpt.co/docs/platform/changelog/changelog";
// The release list lives on our `gitbook` branch; fetched via a cached,
// same-origin proxy (src/app/api/changelog/route.ts) so browsers hit our own
// origin, not GitHub.
export const CHANGELOG_INDEX_MD_URL = "/api/changelog";
export const STORAGE_KEY = "autogpt-changelog-last-seen";
export const AUTO_DISMISS_MS = 8000;
