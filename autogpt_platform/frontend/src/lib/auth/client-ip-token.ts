import { mintServiceToken } from "./service-token";

export const CLIENT_IP_SCOPE = "client-ip";

// As for the country token: reuse a token for half its 60s life. Cleared when
// full, so one instance never holds more than this many visitors.
const REUSE_MS = 30_000;
const MAX_CACHED = 1_000;

const minted = new Map<string, { token: string; mintedAt: number }>();

/**
 * The visitor's IP, as the edge forwarded it, in a frontend service token the
 * backend can verify. Backend limits on public routes count per visitor by it;
 * without one, every visitor arrives from this server's address. Null when the
 * request carries no usable IP or signing fails.
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

function readClientIP(requestHeaders: Headers) {
  const ip = requestHeaders.get("x-forwarded-for")?.split(",")[0]?.trim();
  return ip && /^[0-9A-Fa-f:.]{2,45}$/.test(ip) ? ip : null;
}
