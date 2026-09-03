export type FlagBackend = "launchdarkly" | "posthog" | "dual";

// Read as a literal so Next/Turbopack inlines it into the client bundle; a
// dynamic ``process.env[name]`` lookup is always empty in the browser.
const configured = process.env.NEXT_PUBLIC_FEATURE_FLAG_BACKEND;

export const FLAG_BACKEND: FlagBackend =
  configured === "posthog" || configured === "dual"
    ? configured
    : "launchdarkly";

export function usesLaunchDarkly() {
  return FLAG_BACKEND !== "posthog";
}

export function usesPostHog() {
  return FLAG_BACKEND !== "launchdarkly";
}

// Deliberately not routed through ``environment``: this module is imported by
// the flag hooks, which 75 test files render with a stubbed environment.
export function isPostHogFlagsEnabled() {
  return Boolean(
    process.env.NEXT_PUBLIC_POSTHOG_KEY && process.env.NEXT_PUBLIC_POSTHOG_HOST,
  );
}
