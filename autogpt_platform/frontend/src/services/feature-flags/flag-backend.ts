// `server` asks our backend, which answers from whichever vendor it is configured for.
export type FlagBackend = "launchdarkly" | "posthog" | "dual" | "server";

// Read as a literal so Next/Turbopack inlines it into the client bundle; a
// dynamic ``process.env[name]`` lookup is always empty in the browser.
const configured = process.env.NEXT_PUBLIC_FEATURE_FLAG_BACKEND;

export const FLAG_BACKEND: FlagBackend =
  configured === "posthog" || configured === "dual" || configured === "server"
    ? configured
    : "launchdarkly";

export function usesLaunchDarkly() {
  return FLAG_BACKEND === "launchdarkly" || FLAG_BACKEND === "dual";
}

export function usesPostHog() {
  return FLAG_BACKEND === "posthog" || FLAG_BACKEND === "dual";
}

// Mirrors ``environment.isPostHogEnabled`` — including its cloud check, since
// that is what gates the PostHogProvider: claiming flags are enabled where no
// provider mounts is how a route ends up waiting on an answer that never comes.
// Deliberately not routed through ``environment`` itself: this module is
// imported by the flag hooks, which 75 test files render with a stubbed one.
export function isPostHogFlagsEnabled() {
  return Boolean(
    process.env.NEXT_PUBLIC_BEHAVE_AS === "CLOUD" &&
      process.env.NEXT_PUBLIC_POSTHOG_KEY &&
      process.env.NEXT_PUBLIC_POSTHOG_HOST,
  );
}
