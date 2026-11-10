import { authClient } from "./client";

const REFRESH_MARGIN_MS = 60_000;
const FALLBACK_LIFETIME_MS = 50 * 60_000;

interface CachedToken {
  token: string;
  expiresAt: number;
}

let cached: CachedToken | null = null;
let pending: Promise<string> | null = null;

export class SignedOutError extends Error {
  constructor() {
    super("You are signed out. Sign in again to continue.");
    this.name = "SignedOutError";
  }
}

/** The short-lived token the backend accepts, minted from the stored session. */
export async function getAccessToken() {
  if (cached && cached.expiresAt - Date.now() > REFRESH_MARGIN_MS) {
    return cached.token;
  }
  if (!pending) {
    pending = mintAccessToken().finally(() => {
      pending = null;
    });
  }
  return pending;
}

export function clearAccessToken() {
  cached = null;
}

async function mintAccessToken() {
  cached = null;
  const { data, error } = await authClient.$fetch<{ token: string }>("/token");
  if (error && error.status !== 401 && error.status !== 403) {
    // Not a dead session (a 5xx, say), so stay signed in. A plain Error,
    // since ApiError's module imports this one.
    throw new Error(
      "Can't reach AutoGPT. Check your connection and try again.",
    );
  }
  if (error || !data?.token) throw new SignedOutError();
  cached = { token: data.token, expiresAt: readExpiry(data.token) };
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
