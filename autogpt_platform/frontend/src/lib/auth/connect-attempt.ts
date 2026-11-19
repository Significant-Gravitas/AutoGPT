import type { LoginProvider } from "@/types/auth";

// A Google or Microsoft sign-in that also asks for the data scopes can end at
// the provider without ever coming back: a Google Workspace admin's block is
// an error page on Google's side. Starting one is noted in the tab, so the
// login and signup pages can offer the same sign-in without connecting when
// the person comes back.
const KEY = "autogpt:connect-attempt";
// Long enough for a consent screen, short enough that a later visit to the
// login page doesn't offer it out of the blue.
const RECENT_MS = 10 * 60 * 1000;

export type ConnectingProvider = Extract<LoginProvider, "google" | "microsoft">;

export function isConnectingProvider(
  value: unknown,
): value is ConnectingProvider {
  return value === "google" || value === "microsoft";
}

export function rememberConnectAttempt(provider: LoginProvider) {
  if (!isConnectingProvider(provider)) return;
  try {
    window.sessionStorage.setItem(
      KEY,
      JSON.stringify({ provider, at: Date.now() }),
    );
  } catch {
    // No storage: the page just won't offer it after a provider's dead end.
  }
}

/**
 * The tab is signed in, so a sign-in it started finished: it mustn't be
 * offered again after a sign-out.
 */
export function forgetConnectAttempt() {
  try {
    window.sessionStorage.removeItem(KEY);
  } catch {
    // No storage: nothing was noted.
  }
}

/** A connecting sign-in this tab started recently and never finished, once. */
export function takeConnectAttempt(): ConnectingProvider | null {
  try {
    const raw = window.sessionStorage.getItem(KEY);
    window.sessionStorage.removeItem(KEY);
    const attempt = raw
      ? (JSON.parse(raw) as { provider?: unknown; at?: unknown })
      : null;
    if (typeof attempt?.at !== "number") return null;
    if (Date.now() - attempt.at > RECENT_MS) return null;
    return isConnectingProvider(attempt.provider) ? attempt.provider : null;
  } catch {
    return null;
  }
}
