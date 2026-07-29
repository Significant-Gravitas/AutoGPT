/**
 * Normalizes an MCP server URL for credential matching.
 *
 * Must stay in sync with the backend's `normalize_mcp_url`
 * (backend/blocks/mcp/helpers.py) — credentials are stored under the
 * normalized URL, so comparing against a raw one silently finds nothing.
 */
export const normalizeMCPUrl = (url: string): string =>
  url.trim().replace(/\/+$/, "");

/**
 * Extracts the hostname from a URL string.
 * @param url - The URL string to extract the hostname from
 * @returns The hostname if valid, null if invalid
 */
export const getHostFromUrl = (url: string): string | null => {
  try {
    if (!url.startsWith("http://") && !url.startsWith("https://")) {
      url = "http://" + url; // Add a scheme if missing for URL parsing
    }
    const urlObj = new URL(url);
    return urlObj.hostname;
  } catch {
    return null;
  }
};
