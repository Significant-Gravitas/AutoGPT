import {
  AbortedError,
  AuthUnavailableError,
  HttpError,
  NetworkError,
  RedirectRefusedError,
  SessionEndedError,
} from "../../api/errors";
import {
  openRequest,
  readBody,
  retryAfterSeconds,
  type HttpResponse,
} from "../../api/http";
import { asRecord, nonEmptyString } from "../../lib/json";
import type { ServerConfig } from "../servers/serverConfig";

export interface SignedInAccount {
  sessionToken: string;
  userID: string;
  email: string;
  name: string | null;
}

export type SignInFailure =
  | "credentials"
  | "unverified"
  | "blocked"
  | "rate_limited"
  | "incompatible"
  | "rejected"
  | "unavailable";

export class SignInError extends Error {
  readonly failure: SignInFailure;

  constructor(failure: SignInFailure, message: string) {
    super(message);
    this.name = "SignInError";
    this.failure = failure;
  }
}

/**
 * Email sign-in through the server's own Better Auth policies. The signed
 * session credential arrives in the `set-auth-token` header; the token in the
 * JSON body is unsigned and is deliberately ignored.
 */
export async function signInWithEmail(
  server: ServerConfig,
  email: string,
  password: string,
): Promise<SignedInAccount> {
  let response: HttpResponse;
  let release = () => {};
  try {
    ({ response, release } = await openRequest({
      url: `${server.authUrl}/api/auth/sign-in/email`,
      method: "POST",
      json: { email, password, rememberMe: true },
      timeoutMs: 30_000,
    }));
  } catch (error) {
    throw signInTransportError(error);
  }
  try {
    const body = await readBody(response).catch(() => null);
    if (!response.ok) throw signInHTTPError(response, body);
    const sessionToken = response.headers.get("set-auth-token");
    const user = asRecord(asRecord(body)?.user);
    const userID = nonEmptyString(user?.id);
    const accountEmail = nonEmptyString(user?.email);
    if (
      !sessionToken ||
      !sessionToken.includes(".") ||
      !userID ||
      !accountEmail
    ) {
      throw new SignInError(
        "incompatible",
        "This server doesn't support signing in from the app yet. Ask its administrator to update AutoGPT.",
      );
    }
    return {
      sessionToken,
      userID,
      email: accountEmail,
      name: nonEmptyString(user?.name),
    };
  } finally {
    release();
  }
}

export interface AccessToken {
  token: string;
  expiresAt: number;
  /** A re-issued signed session credential, when the auth server sends one. */
  renewedSessionToken: string | null;
}

/** Exchanges the session credential for a short-lived API access token. */
export async function exchangeAccessToken(
  server: ServerConfig,
  sessionToken: string,
  signal?: AbortSignal,
): Promise<AccessToken> {
  let response: HttpResponse;
  let release = () => {};
  try {
    ({ response, release } = await openRequest({
      url: `${server.authUrl}/api/auth/token`,
      headers: { Authorization: `Bearer ${sessionToken}` },
      signal,
    }));
  } catch (error) {
    if (error instanceof AbortedError) throw error;
    throw new AuthUnavailableError(null);
  }
  try {
    if (response.status === 401) throw new SessionEndedError();
    if (!response.ok)
      throw new AuthUnavailableError(retryAfterSeconds(response));
    const body = asRecord(await readBody(response, signal));
    const token = nonEmptyString(body?.token);
    if (!token) throw new AuthUnavailableError(null);
    const renewed = response.headers.get("set-auth-token");
    return {
      token,
      expiresAt: jwtExpiry(token) ?? Date.now() + 5 * 60_000,
      renewedSessionToken:
        renewed && renewed.includes(".") && renewed !== sessionToken
          ? renewed
          : null,
    };
  } catch (error) {
    if (
      error instanceof SessionEndedError ||
      error instanceof AuthUnavailableError
    )
      throw error;
    if (error instanceof AbortedError) throw error;
    throw new AuthUnavailableError(null);
  } finally {
    release();
  }
}

/** Best-effort server sign-out. The local credential is already gone when this runs. */
export async function revokeSession(
  server: ServerConfig,
  sessionToken: string,
): Promise<void> {
  try {
    const { release } = await openRequest({
      url: `${server.authUrl}/api/auth/sign-out`,
      method: "POST",
      json: {},
      headers: { Authorization: `Bearer ${sessionToken}` },
      timeoutMs: 10_000,
    });
    release();
  } catch {
    // The session still expires server-side; nothing else to do offline.
  }
}

function jwtExpiry(token: string): number | null {
  const payload = token.split(".")[1];
  if (!payload) return null;
  try {
    const base64 = payload.replace(/-/g, "+").replace(/_/g, "/");
    const padded = base64 + "=".repeat((4 - (base64.length % 4)) % 4);
    const claims = asRecord(JSON.parse(atob(padded)));
    return typeof claims?.exp === "number" ? claims.exp * 1000 : null;
  } catch {
    return null;
  }
}

function signInTransportError(error: unknown): SignInError {
  if (error instanceof RedirectRefusedError) {
    return new SignInError(
      "incompatible",
      "The server redirected the sign-in request. Check the sign-in address.",
    );
  }
  if (error instanceof NetworkError && error.timedOut) {
    return new SignInError(
      "unavailable",
      "The server took too long to respond. Try again.",
    );
  }
  return new SignInError(
    "unavailable",
    "Can't reach the server. Check the address and your connection.",
  );
}

function signInHTTPError(response: HttpResponse, body: unknown): SignInError {
  const record = asRecord(body);
  const code = typeof record?.code === "string" ? record.code : null;
  const serverMessage = nonEmptyString(record?.message);
  if (response.status === 401 || code === "INVALID_EMAIL_OR_PASSWORD") {
    return new SignInError(
      "credentials",
      "That email and password don't match.",
    );
  }
  if (code === "EMAIL_NOT_VERIFIED") {
    return new SignInError(
      "unverified",
      "Verify your email address first. Check your inbox for a verification link, then sign in again.",
    );
  }
  if (code === "BANNED_USER") {
    return new SignInError(
      "blocked",
      "This account can't sign in. Contact support if you think this is a mistake.",
    );
  }
  if (response.status === 429) {
    const wait = retryAfterSeconds(response);
    return new SignInError(
      "rate_limited",
      wait
        ? `Too many attempts. Try again in ${Math.ceil(wait)} seconds.`
        : "Too many attempts. Try again shortly.",
    );
  }
  if (response.status === 404) {
    return new SignInError(
      "incompatible",
      "This doesn't look like an AutoGPT sign-in address.",
    );
  }
  if (response.status >= 500) {
    return new SignInError(
      "unavailable",
      "The server couldn't sign you in right now. Try again.",
    );
  }
  const fallback = new HttpError({
    status: response.status,
    detail: body,
    retryAfterSeconds: null,
  });
  return new SignInError("rejected", serverMessage ?? fallback.message);
}
