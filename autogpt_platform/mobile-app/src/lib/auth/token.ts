import { SessionChangedError, SignedOutError } from "./errors";
import { authRequest, hasSession, sessionToken } from "./nativeSession";

const REFRESH_MARGIN_MS = 60_000;
const OFFLINE_MESSAGE = "Can't reach AutoGPT. Check your connection and try again.";
const FALLBACK_LIFETIME_MS = 50 * 60_000;

interface CachedToken {
  token: string;
  expiresAt: number;
}

let cached: CachedToken | null = null;
let pending: Promise<string> | null = null;
// Bumped on every clear, so a mint that started before a sign-out cannot
// cache the old user's token for the next one.
let generation = 0;

/** Which session a request starts under; pass it to `assertSameSession` before a retry. */
export function sessionMark() {
  return sessionToken();
}

/**
 * Throws once the session a request started under has ended, so neither its
 * token nor a retry of it goes out for whoever signs in next.
 */
export function assertSameSession(mark: string | null) {
  if (sessionToken() !== mark) throw new SessionChangedError();
}

/** The short-lived token the backend accepts, minted from the stored session. */
export async function getAccessToken() {
  if (cached && cached.expiresAt - Date.now() > REFRESH_MARGIN_MS) {
    return cached.token;
  }
  const mark = sessionMark();
  if (!pending) {
    const mint = mintAccessToken().finally(() => {
      if (pending === mint) pending = null;
    });
    pending = mint;
  }
  let token: string;
  try {
    token = await pending;
  } catch (error) {
    // The old session's refusal must not sign out whoever is signed in now.
    assertSameSession(mark);
    throw error;
  }
  // A sign-out, or a sign-in as someone else, while the token was minted.
  assertSameSession(mark);
  return token;
}

export function clearAccessToken() {
  generation += 1;
  cached = null;
  pending = null;
}

async function mintAccessToken() {
  const startedIn = generation;
  cached = null;
  if (!hasSession()) throw new SignedOutError();
  const response = await authRequest("/token").catch(() => null);
  const isDeadSession = response?.status === 401 || response?.status === 403;
  if (!response || (!response.ok && !isDeadSession)) {
    // Not a dead session (offline, or a 503 while the auth service is
    // down), so stay signed in. A plain Error, since ApiError's module
    // imports this one.
    throw new Error(OFFLINE_MESSAGE);
  }
  // A body that breaks off mid-read is a dropped connection, not a dead session.
  const data = isDeadSession
    ? null
    : ((await response.json().catch(() => {
        throw new Error(OFFLINE_MESSAGE);
      })) as { token?: unknown } | null);
  if (typeof data?.token !== "string") throw new SignedOutError();
  if (generation === startedIn) {
    cached = { token: data.token, expiresAt: readExpiry(data.token) };
  }
  return data.token;
}

function readExpiry(token: string) {
  try {
    const payload = token.split(".")[1] ?? "";
    const json = atob(payload.replace(/-/g, "+").replace(/_/g, "/"));
    const exp = (JSON.parse(json) as { exp?: unknown }).exp;
    if (typeof exp === "number") return exp * 1000;
  } catch {
    // An unreadable token still works; it is just refreshed on a fixed clock.
  }
  return Date.now() + FALLBACK_LIFETIME_MS;
}
