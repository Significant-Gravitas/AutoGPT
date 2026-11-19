import {
  APIError,
  addOAuthServerContext,
  getOAuthState,
} from "better-auth/api";
import type { SignInConnectionClients } from "@/app/api/__generated__/models/signInConnectionClients";
import { GOOGLE_ACCESS_SCOPES } from "@/lib/google-access";
import { MICROSOFT_BASE_SCOPES } from "@/lib/microsoft-setup";
import { environment } from "@/services/environment";

// Google and Microsoft sign-in also ask for the data scopes, so a person
// starts with the connection the Connect flow would make: Gmail, Drive and
// Calendar, or Microsoft 365 mail, calendar and files. sign-in-connection.ts
// hands the tokens to the backend, which keeps them as that connection.
//
// Only when sign-in uses the OAuth client the backend's connections use: the
// backend can renew only tokens its own client was issued, and refuses others.
// It says which clients those are, so a deployment whose sign-in has a client
// of its own, or that has no such connections, asks only who the person is.
const CONNECT_SCOPES = {
  google: GOOGLE_ACCESS_SCOPES,
  // Microsoft returns a refresh token only for offline_access.
  microsoft: ["offline_access", ...MICROSOFT_BASE_SCOPES],
} as const satisfies Record<string, readonly string[]>;

export type ConnectProvider = keyof typeof CONNECT_SCOPES;

// In Microsoft's list, but not data: User.Read reads the person's own profile,
// which a sign-in that asked only who they are can come back with, and
// offline_access only makes the tokens renewable.
const NOT_DATA_SCOPES: readonly string[] = ["User.Read", "offline_access"];

// Noted in the OAuth state of a sign-in that asks for the data scopes.
const ASKED_TO_CONNECT = "connectAtSignIn";

// The OAuth client each sign-in uses (auth.ts, microsoft-login.ts).
const SIGN_IN_CLIENT_ENV = {
  google: "AUTH_GOOGLE_CLIENT_ID",
  microsoft: "AUTH_MICROSOFT_CLIENT_ID",
} as const satisfies Record<ConnectProvider, string>;

// The backend's clients change only when it is redeployed.
const CLIENTS_KEPT_MS = 5 * 60_000;
// Until the backend answers, sign-ins ask only who the person is.
const RETRY_AFTER_MS = 30_000;
// Starting a sign-in waits on the answer, so a stalled backend costs at most
// this.
const LOOKUP_TIMEOUT_MS = 2_000;

export function isConnectProvider(id: unknown): id is ConnectProvider {
  return typeof id === "string" && Object.hasOwn(CONNECT_SCOPES, id);
}

/**
 * The data scopes a sign-in with this provider asks for, or null when it asks
 * only who the person is.
 */
export async function connectAtSignInScopes(
  provider: unknown,
): Promise<readonly string[] | null> {
  if (!isConnectProvider(provider)) return null;
  const signInClient = process.env[SIGN_IN_CLIENT_ENV[provider]];
  if (!signInClient) return null;
  const clients = await connectionClients();
  return clients?.[provider] === signInClient ? CONNECT_SCOPES[provider] : null;
}

let lookup:
  | { until: number; clients: Promise<SignInConnectionClients | null> }
  | undefined;

function connectionClients() {
  if (lookup && Date.now() < lookup.until) return lookup.clients;
  const entry = {
    until: Date.now() + CLIENTS_KEPT_MS,
    clients: askBackendForClients(),
  };
  void entry.clients.then((clients) => {
    if (!clients) entry.until = Date.now() + RETRY_AFTER_MS;
  });
  lookup = entry;
  return entry.clients;
}

async function askBackendForClients(): Promise<SignInConnectionClients | null> {
  try {
    const response = await fetch(
      `${environment.getAGPTServerBaseUrl()}/api/auth/sign-in-connection`,
      { cache: "no-store", signal: AbortSignal.timeout(LOOKUP_TIMEOUT_MS) },
    );
    if (!response.ok) throw new Error(`status ${response.status}`);
    return (await response.json()) as SignInConnectionClients;
  } catch (error) {
    console.warn("Couldn't ask the backend which clients its connections use", {
      error: error instanceof Error ? error.message : String(error),
    });
    return null;
  }
}

const SCOPED_SIGN_IN_PATHS = new Set(["/sign-in/social", "/link-social"]);

/** Whether any of these scopes reaches the person's mail, files or calendar. */
export function includesDataScope(
  provider: ConnectProvider,
  scopes: readonly string[],
) {
  const connectScopes: readonly string[] = CONNECT_SCOPES[provider];
  return scopes.some(
    (scope) =>
      connectScopes.includes(scope) && !NOT_DATA_SCOPES.includes(scope),
  );
}

/**
 * Better Auth passes a sign-in request's `scopes` straight to the provider.
 * For Google and Microsoft, whose tokens become a connection, only the data
 * scopes above may be asked for, and only when the sign-in can connect. A
 * sign-in that asks for them says so in its OAuth state (askedToConnect).
 */
export async function assertSignInScopes(
  path: string | undefined,
  body: unknown,
) {
  if (!path || !SCOPED_SIGN_IN_PATHS.has(path)) return;
  const { provider, scopes } = (body ?? {}) as {
    provider?: unknown;
    scopes?: unknown;
  };
  if (!isConnectProvider(provider) || scopes === undefined) return;
  const allowed = (await connectAtSignInScopes(provider)) ?? [];
  if (
    Array.isArray(scopes) &&
    scopes.every(
      (scope) => typeof scope === "string" && allowed.includes(scope),
    )
  ) {
    if (includesDataScope(provider, scopes)) {
      await addOAuthServerContext({ [ASKED_TO_CONNECT]: true });
    }
    return;
  }
  throw new APIError("BAD_REQUEST", {
    code: "SCOPES_NOT_ALLOWED",
    message: "This sign-in can't ask for those permissions.",
  });
}

/**
 * Whether the sign-in whose callback this is asked for the data scopes. Only
 * the server writes this part of the state, so the browser can't claim it.
 */
export async function askedToConnect() {
  try {
    const state = await getOAuthState();
    return state?.serverContext?.[ASKED_TO_CONNECT] === true;
  } catch {
    // No sign-in's request, so no state: nothing to connect.
    return false;
  }
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
