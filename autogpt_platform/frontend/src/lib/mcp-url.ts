/** Mirrors the backend `normalize_mcp_url` (`blocks/mcp/helpers.py`). */
export function normalizeMcpUrl(value: string): string {
  return value.trim().replace(/\/+$/, "");
}

/**
 * The origin of a server URL, so editing the path does not count as switching
 * servers. Falls back to the normalized string while the URL is unparseable.
 */
export function mcpServerIdentity(value: string): string {
  const normalized = normalizeMcpUrl(value);
  try {
    return new URL(normalized).origin.toLowerCase();
  } catch {
    return normalized.toLowerCase();
  }
}
