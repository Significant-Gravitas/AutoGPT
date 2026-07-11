// Public docs pages the user can open (used for "open in docs" links).
export const CHANGELOG_BASE_URL =
  "https://agpt.co/docs/platform/changelog/changelog";
// The changelog source (markdown + hero images) lives on our `gitbook` branch.
// We fetch it through our own server-side, cached proxies rather than having
// browsers hit GitHub directly — see src/app/api/changelog/*.
export const CHANGELOG_PROXY_URL = "/api/changelog";
export const CHANGELOG_IMAGE_PROXY_URL = "/api/changelog/image";
export const CHANGELOG_INDEX_MD_URL = CHANGELOG_PROXY_URL;
export const STORAGE_KEY = "autogpt-changelog-last-seen";
export const AUTO_DISMISS_MS = 8000;
