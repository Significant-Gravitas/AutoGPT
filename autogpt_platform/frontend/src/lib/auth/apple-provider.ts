import { APIError } from "better-auth/api";
import { apple, type AppleOptions } from "better-auth/social-providers";
import { verifyProviderIdToken } from "@better-auth/core/oauth2";
import { appleWebCredentials } from "./apple-client-secret";
import { currentAuthEndpoint } from "./auth-endpoint";

export interface AppleAccountUser {
  email: string;
  emailVerified: boolean;
  name: string;
}

export function getAppleProvider(
  findAccountUser: (subject: string) => Promise<AppleAccountUser | null>,
) {
  const bundleID = process.env.AUTH_APPLE_APP_BUNDLE_IDENTIFIER?.trim();
  const web = appleWebCredentials();
  const clientID = web?.clientID || bundleID;
  if (!clientID) return;

  const options: AppleOptions = {
    clientId: clientID,
    audience: Array.from(new Set([clientID, bundleID ?? clientID])),
    async mapProfileToUser(profile) {
      if (profile.email) return {};
      const user = await findAccountUser(profile.sub);
      if (!user) return {};
      return {
        email: user.email,
        emailVerified: user.emailVerified,
        name: user.name,
      };
    },
  };
  const provider = apple(options);
  const callbackProvider = apple({ ...options, audience: [clientID] });
  return {
    ...options,
    async getUserInfo(tokens) {
      const endpoint = await currentAuthEndpoint();
      if (
        endpoint.path !== "/sign-in/social" &&
        (!tokens.idToken ||
          !(await verifyProviderIdToken(callbackProvider, tokens.idToken)))
      )
        return null;
      return provider.getUserInfo(tokens);
    },
  } satisfies AppleOptions;
}

export function appleTrustedOrigins(request?: Request) {
  if (
    appleWebCredentials() &&
    request?.method === "POST" &&
    new URL(request.url).pathname === "/api/auth/callback/apple"
  )
    return ["https://appleid.apple.com"];
  return [];
}

export function assertAppleSignInRequest(
  path: string,
  body: { provider?: unknown; idToken?: { nonce?: unknown } } | undefined,
) {
  if (path !== "/sign-in/social" || body?.provider !== "apple") return;
  const webConfigured = Boolean(appleWebCredentials());
  if (!process.env.AUTH_APPLE_APP_BUNDLE_IDENTIFIER?.trim() && !webConfigured) {
    return;
  }
  if (!body.idToken) {
    if (webConfigured) return;
    throw new APIError("BAD_REQUEST", {
      code: "APPLE_WEB_LOGIN_NOT_CONFIGURED",
      message: "Browser Sign in with Apple is not configured for this server.",
    });
  }
  if (typeof body.idToken.nonce !== "string" || !body.idToken.nonce.trim()) {
    throw new APIError("BAD_REQUEST", {
      code: "APPLE_NONCE_REQUIRED",
      message: "Restart Sign in with Apple and try again.",
    });
  }
}
