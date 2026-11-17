import { APIError } from "better-auth/api";
import { GOOGLE_ACCESS_SCOPES } from "@/lib/google-access";
import { MICROSOFT_BASE_SCOPES } from "@/lib/microsoft-setup";

// Google and Microsoft sign-in can ask for the data scopes too, so a person
// starts with the connection the Connect flow would make: Gmail, Drive and
// Calendar, or Microsoft 365 mail, calendar and files. sign-in-connection.ts
// hands the tokens to the backend, which keeps them as that connection.
//
// Off unless AUTH_CONNECT_AT_SIGN_IN names the provider ("google,microsoft").
// Turn Google on only once login uses the backend's Google OAuth client: the
// backend can renew only tokens its own client was issued, and refuses others.
const CONNECT_SCOPES = {
  google: GOOGLE_ACCESS_SCOPES,
  // Microsoft returns a refresh token only for offline_access.
  microsoft: ["offline_access", ...MICROSOFT_BASE_SCOPES],
} as const satisfies Record<string, readonly string[]>;

export type ConnectProvider = keyof typeof CONNECT_SCOPES;

export function isConnectProvider(id: unknown): id is ConnectProvider {
  return typeof id === "string" && Object.hasOwn(CONNECT_SCOPES, id);
}

/**
 * The data scopes a sign-in with this provider asks for, or null when it asks
 * only who the person is.
 */
export function connectAtSignInScopes(
  provider: unknown,
): readonly string[] | null {
  if (!isConnectProvider(provider)) return null;
  const enabled = (process.env.AUTH_CONNECT_AT_SIGN_IN ?? "")
    .split(",")
    .map((name) => name.trim());
  return enabled.includes(provider) ? CONNECT_SCOPES[provider] : null;
}

const SCOPED_SIGN_IN_PATHS = new Set(["/sign-in/social", "/link-social"]);

/**
 * Better Auth passes a sign-in request's `scopes` straight to the provider.
 * For Google and Microsoft, whose tokens become a connection, only the data
 * scopes above may be asked for, and only while connecting at sign-in is on.
 */
export function assertSignInScopes(path: string | undefined, body: unknown) {
  if (!path || !SCOPED_SIGN_IN_PATHS.has(path)) return;
  const { provider, scopes } = (body ?? {}) as {
    provider?: unknown;
    scopes?: unknown;
  };
  if (!isConnectProvider(provider) || scopes === undefined) return;
  const allowed = connectAtSignInScopes(provider) ?? [];
  if (
    Array.isArray(scopes) &&
    scopes.every(
      (scope) => typeof scope === "string" && allowed.includes(scope),
    )
  ) {
    return;
  }
  throw new APIError("BAD_REQUEST", {
    code: "SCOPES_NOT_ALLOWED",
    message: "This sign-in can't ask for those permissions.",
  });
}

/**
 * Better Auth would keep the provider's tokens in UserAuthAccount,
 * unencrypted. A Google or Microsoft token can carry mail and file access (and
 * Google's includes every scope granted before), so it is kept only as the
 * backend's encrypted connection. Nothing reads these columns.
 */
export function withoutProviderTokens<T extends { providerId?: string }>(
  account: T,
) {
  if (!isConnectProvider(account.providerId)) return account;
  return { ...account, accessToken: null, refreshToken: null, idToken: null };
}
