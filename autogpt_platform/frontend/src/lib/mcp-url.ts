/**
 * Mirrors the backend `normalize_mcp_url` (`blocks/mcp/helpers.py`) so a stored
 * credential for `https://mcp.sentry.dev/mcp` matches a card emitted with the
 * same URL whether or not the trailing slash is present.
 */
export function normalizeMcpUrl(value: string): string {
  return value.trim().replace(/\/+$/, "");
}

/**
 * The identity of the *server*, ignoring the path.
 *
 * Used to decide whether an in-progress credential is still relevant to what
 * the user is typing. Comparing whole URLs makes every keystroke a "different
 * server", which discards the credential while the user is fixing the `/mcp`
 * suffix; a credential is issued by a host, so the host is what has to change
 * before it stops applying.
 *
 * Falls back to the normalized string for values that are not yet parseable —
 * a half-typed URL has no credential to protect.
 */
export function mcpServerIdentity(value: string): string {
  const normalized = normalizeMcpUrl(value);
  try {
    return new URL(normalized).origin.toLowerCase();
  } catch {
    return normalized.toLowerCase();
  }
}
