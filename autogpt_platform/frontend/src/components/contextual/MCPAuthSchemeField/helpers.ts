import type { MCPAuthScheme } from "@/lib/mcp-auth";

/** Label for the credential input, which names what the server expects. */
export function mcpAuthTokenLabel(scheme: MCPAuthScheme): string {
  return scheme === "basic" ? "Basic authentication token" : "API token";
}

/**
 * Hint under the credential input.
 *
 * The Basic wording names the Base64 step explicitly: providers document the
 * credential as `public-key:secret-key`, and a user who pastes that pair raw
 * gets a 401 with nothing pointing at the encoding they skipped.
 */
export function mcpAuthTokenHint(scheme: MCPAuthScheme): string {
  return scheme === "basic"
    ? 'Paste the Base64 of user:password — the value after "Basic" — or the complete Authorization header.'
    : "Paste the token itself. AutoGPT sends it using Bearer authentication.";
}

/**
 * Placeholder for the credential input.
 *
 * Under a Basic hint that says "paste the Base64 of user:password", a fixed
 * "Paste API token" restates the mistake the hint exists to prevent.
 */
export function mcpAuthTokenPlaceholder(scheme: MCPAuthScheme): string {
  return scheme === "basic"
    ? "Paste Base64 of user:password"
    : "Paste API token";
}
