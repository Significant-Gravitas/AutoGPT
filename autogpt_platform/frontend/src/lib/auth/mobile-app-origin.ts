import type { BetterAuthPlugin } from "better-auth";

// A native app sends no Origin header, so the Better Auth Expo client names
// itself in this one instead.
export const MOBILE_APP_ORIGIN_HEADER = "expo-origin";

const SCHEME_PATTERN = /^[a-z][a-z0-9+.-]*$/;
const WEB_SCHEMES = ["http", "https"];

/**
 * The origins the mobile app signs in from. Empty unless
 * AUTH_MOBILE_APP_SCHEME is set, so a deployment without the app trusts
 * nothing new.
 */
export function getMobileAppOrigins(
  env: Record<string, string | undefined> = process.env,
) {
  const scheme = env.AUTH_MOBILE_APP_SCHEME?.trim().toLowerCase();
  if (!scheme) return [];
  if (!SCHEME_PATTERN.test(scheme) || WEB_SCHEMES.includes(scheme)) {
    throw new Error(
      "AUTH_MOBILE_APP_SCHEME must be an app URL scheme such as autogpt-app",
    );
  }
  // Expo Go serves a development build from exp://<host>, never the app's
  // own scheme.
  return env.NODE_ENV === "development"
    ? [`${scheme}://`, "exp://"]
    : [`${scheme}://`];
}

/**
 * Lets the mobile app pass Better Auth's origin check. It is the origin half
 * of `@better-auth/expo` only: that plugin also copies the session cookie into
 * the app link a verification email returns to, where any app that claims the
 * scheme can read it, and mounts an OAuth proxy route the app does not use.
 */
export function mobileAppOrigin(): BetterAuthPlugin {
  return {
    id: "mobile-app-origin",
    async onRequest(request) {
      if (request.headers.get("origin")) return;
      const appOrigin = request.headers.get(MOBILE_APP_ORIGIN_HEADER);
      if (!appOrigin) return;
      const headers = new Headers(request.headers);
      headers.set("origin", appOrigin);
      return { request: new Request(request, { headers }) };
    },
  };
}
