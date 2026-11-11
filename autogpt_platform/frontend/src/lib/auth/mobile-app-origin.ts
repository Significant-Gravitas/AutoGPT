import type { BetterAuthPlugin } from "better-auth";

// A native app sends no Origin header, so the Better Auth Expo client names
// itself in this one instead.
export const MOBILE_APP_ORIGIN_HEADER = "expo-origin";

const SCHEME_PATTERN = /^[a-z][a-z0-9+.-]*$/;
const WEB_SCHEMES = ["http", "https"];
const EXPO_GO_PREFIX = "exp://";

/**
 * The mobile app's URL scheme, or null unless AUTH_MOBILE_APP_SCHEME is set,
 * so a deployment without the app changes nothing.
 */
export function getMobileAppConfig(
  env: Record<string, string | undefined> = process.env,
) {
  const scheme = env.AUTH_MOBILE_APP_SCHEME?.trim().toLowerCase();
  if (!scheme) return null;
  if (!SCHEME_PATTERN.test(scheme) || WEB_SCHEMES.includes(scheme)) {
    throw new Error(
      "AUTH_MOBILE_APP_SCHEME must be an app URL scheme such as autogpt-app",
    );
  }
  // Expo Go serves a development build from exp://<host>, never the app's
  // own scheme.
  return { scheme, allowExpoGo: env.NODE_ENV === "development" };
}

type MobileAppConfig = NonNullable<ReturnType<typeof getMobileAppConfig>>;

/**
 * Lets the mobile app pass Better Auth's origin check by giving its requests
 * this site's own Origin. The scheme stays out of trustedOrigins on purpose:
 * that list also vets callback and redirect URLs, so trusting it would let
 * anyone send a password-reset link's token to whatever app claims the scheme.
 * Browsers always send Origin on such requests, so they are never rewritten.
 */
export function mobileAppOrigin(
  config: MobileAppConfig,
  baseURL: string,
): BetterAuthPlugin {
  const siteOrigin = new URL(baseURL).origin;

  function isAppOrigin(value: string) {
    if (value === `${config.scheme}://`) return true;
    return config.allowExpoGo && value.startsWith(EXPO_GO_PREFIX);
  }

  return {
    id: "mobile-app-origin",
    async onRequest(request) {
      if (request.headers.get("origin")) return;
      const appOrigin = request.headers.get(MOBILE_APP_ORIGIN_HEADER);
      if (!appOrigin || !isAppOrigin(appOrigin)) return;
      try {
        request.headers.set("origin", siteOrigin);
        return { request };
      } catch {
        // Some runtimes hand over read-only headers.
        const headers = new Headers(request.headers);
        headers.set("origin", siteOrigin);
        return {
          request: new Request(request, {
            headers,
            duplex: "half",
          } as RequestInit),
        };
      }
    },
  };
}
