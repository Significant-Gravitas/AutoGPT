import { getIPFromHeader } from "@better-auth/core/utils/ip";

import { isForwardedForTrusted, trustedProxies } from "./client-ip";
import { mintServiceToken } from "./service-token";

export const CLIENT_IP_SCOPE = "client-ip";

// As for the country token: reuse a token for half its 60s life. Cleared when
// full, so one instance never holds more than this many visitors.
const REUSE_MS = 30_000;
const MAX_CACHED = 1_000;

const minted = new Map<string, { token: string; mintedAt: number }>();

/**
 * The visitor's IP, as a proxy in front of this server reported it, in a frontend
 * service token the backend can verify. Backend limits on public routes count per
 * visitor by it. Null when no trusted proxy reports the address, when it is not
 * usable, or when signing fails; the backend then keys on its own peer address.
 */
export async function getClientIPToken(requestHeaders: Headers) {
  const ip = readClientIP(requestHeaders);
  if (!ip) return null;
  const cached = minted.get(ip);
  if (cached && Date.now() - cached.mintedAt < REUSE_MS) return cached.token;
  try {
    const token = await mintServiceToken(CLIENT_IP_SCOPE, { ip });
    if (minted.size >= MAX_CACHED) minted.clear();
    minted.set(ip, { token, mintedAt: Date.now() });
    return token;
  } catch (error) {
    console.error("[client-ip] Could not sign the client IP", error);
    return null;
  }
}

// Through Better Auth's own resolver, so the backend counts a visitor by the same
// address the sign-in log and the auth limits use: the first hop from the right
// that TRUSTED_PROXIES does not list.
function readClientIP(requestHeaders: Headers) {
  if (!isForwardedForTrusted()) return null;
  const forwarded = requestHeaders.get("x-forwarded-for");
  if (!forwarded) return null;
  return getIPFromHeader(forwarded, { trustedProxies: trustedProxies() });
}
