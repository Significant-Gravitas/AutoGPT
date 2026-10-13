// PostHog's edge answers any User-Agent containing "HeadlessChrome" with no
// flags at all, so against dev or a preview a headless browser would see every
// flag off. Journeys present a normal Chrome instead. A context made by hand
// with browser.newContext() doesn't inherit the config; pass this to it.
export const JOURNEY_USER_AGENT =
  process.env.PLAYWRIGHT_USER_AGENT ??
  "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36";
