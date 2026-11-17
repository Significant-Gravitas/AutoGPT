import { tryGetCurrentAuthEndpointContext } from "@better-auth/core/context";
import type { OAuth2Tokens } from "@better-auth/core/oauth2";
import { google } from "@better-auth/core/social-providers";
import type { SignInConnectionRequest } from "@/app/api/__generated__/models/signInConnectionRequest";
import { environment } from "@/services/environment";
import {
  type ConnectProvider,
  connectAtSignInScopes,
} from "./connect-at-sign-in";
import { mintServiceToken } from "./service-token";

const SIGN_IN_CONNECTION_SCOPE = "integrations:sign-in-connection";
// The sign-in's redirect waits on the handoff, so a stalled backend costs a
// few seconds, not the sign-in.
const CONNECT_TIMEOUT_MS = 8_000;
const GRAPH_PREFIX = "https://graph.microsoft.com/";

// The tokens a provider's redirect brought, by the endpoint context Better
// Auth runs the callback in: the provider's getUserInfo and the session hook
// that ends the sign-in see the same one (sign-in-connection-flow.test.ts).
const signInTokens = new WeakMap<object, SignInConnectionRequest>();

/**
 * A provider's getUserInfo that first keeps the sign-in's tokens, so they
 * can become the account's connection once the sign-in succeeds.
 */
export function keepingSignInTokens<T>(
  provider: ConnectProvider,
  getUserInfo: (tokens: OAuth2Tokens) => Promise<T>,
) {
  return async (tokens: OAuth2Tokens) => {
    keepTokens(provider, tokens);
    return getUserInfo(tokens);
  };
}

/** Google's login options: with connecting on, refresh tokens are kept. */
export function googleLoginProvider(clientId: string, clientSecret: string) {
  if (!connectAtSignInScopes("google")) return { clientId, clientSecret };
  const options = { clientId, clientSecret, accessType: "offline" as const };
  const base = google(options);
  return {
    ...options,
    getUserInfo: keepingSignInTokens("google", (tokens) =>
      base.getUserInfo(tokens),
    ),
  };
}

function keepTokens(provider: ConnectProvider, tokens: OAuth2Tokens) {
  const ctx = tryGetCurrentAuthEndpointContext() as
    | { path?: string }
    | undefined;
  // Only the provider's redirect: an ID token sent to /sign-in/social comes
  // with whatever tokens the browser chose.
  if (!ctx?.path?.startsWith("/callback/")) return;
  const asked = connectAtSignInScopes(provider);
  if (!asked || !tokens.accessToken) return;
  const granted = (tokens.scopes ?? []).map((scope) =>
    scope.replace(GRAPH_PREFIX, ""),
  );
  // A sign-in that asked who the person is, or where every data permission
  // was declined, has nothing to connect.
  if (!granted.some((scope) => asked.includes(scope))) return;
  signInTokens.set(ctx, {
    provider,
    access_token: tokens.accessToken,
    refresh_token: tokens.refreshToken || null,
    expires_at: tokens.accessTokenExpiresAt
      ? Math.floor(tokens.accessTokenExpiresAt.getTime() / 1000)
      : null,
    scope: (tokens.scopes ?? []).join(" "),
    id_token: tokens.idToken ?? null,
  });
}

/**
 * Hands the sign-in's tokens to the backend, which keeps them as the
 * account's connection (backend `sign_in_connection.py`). Called once the
 * sign-in's session exists. It never fails the sign-in: without the
 * connection, onboarding's access step offers it.
 */
export async function connectSignIn(ctx: unknown, userID: string) {
  if (typeof ctx !== "object" || ctx === null) return;
  const request = signInTokens.get(ctx);
  if (!request) return;
  signInTokens.delete(ctx);
  try {
    const token = await mintServiceToken(SIGN_IN_CONNECTION_SCOPE, {
      user_id: userID,
    });
    const response = await fetch(
      `${environment.getAGPTServerBaseUrl()}/api/auth/sign-in-connection`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(request),
        signal: AbortSignal.timeout(CONNECT_TIMEOUT_MS),
      },
    );
    if (!response.ok) {
      console.error("A sign-in's connection wasn't kept", {
        provider: request.provider,
        status: response.status,
      });
    }
  } catch (error) {
    console.error("A sign-in's connection wasn't kept", {
      provider: request.provider,
      error: error instanceof Error ? error.message : String(error),
    });
  }
}
