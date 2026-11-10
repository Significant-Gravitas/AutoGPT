import { authClient } from "./client";

const REFRESH_MARGIN_MS = 60_000;
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
    const mint = mintAccessToken().finally(() => {
      if (pending === mint) pending = null;
    });
    pending = mint;
  }
  return pending;
}

export function clearAccessToken() {
  generation += 1;
  cached = null;
  pending = null;
}

async function mintAccessToken() {
  const startedIn = generation;
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
