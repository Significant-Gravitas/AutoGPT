import { APIError } from "better-auth/api";
import type { AppleOptions } from "better-auth/social-providers";

export interface AppleAccountUser {
  email: string;
  emailVerified: boolean;
  name: string;
}

export function getAppleProvider(
  findAccountUser: (subject: string) => Promise<AppleAccountUser | null>,
): AppleOptions | undefined {
  const bundleID = process.env.AUTH_APPLE_APP_BUNDLE_IDENTIFIER?.trim();
  if (!bundleID) return;

  return {
    clientId: bundleID,
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
}

export function assertAppleSignInRequest(
  path: string,
  body: { provider?: unknown; idToken?: { nonce?: unknown } } | undefined,
) {
  if (path !== "/sign-in/social" || body?.provider !== "apple") return;
  if (!body.idToken) {
    throw new APIError("BAD_REQUEST", {
      code: "APPLE_ID_TOKEN_REQUIRED",
      message: "Use Sign in with Apple in the app to continue.",
    });
  }
  if (typeof body.idToken.nonce !== "string" || !body.idToken.nonce.trim()) {
    throw new APIError("BAD_REQUEST", {
      code: "APPLE_NONCE_REQUIRED",
      message: "Restart Sign in with Apple and try again.",
    });
  }
}
