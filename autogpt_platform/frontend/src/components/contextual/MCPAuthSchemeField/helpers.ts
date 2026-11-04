import type { MCPAuthScheme } from "@/lib/mcp-auth";

export function mcpAuthTokenLabel(scheme: MCPAuthScheme): string {
  return scheme === "basic" ? "Basic authentication token" : "API token";
}

export function mcpAuthTokenHint(scheme: MCPAuthScheme): string {
  return scheme === "basic"
    ? 'Paste the Base64 of user:password — the value after "Basic" — or the complete Authorization header.'
    : "Paste the token itself. AutoGPT sends it using Bearer authentication.";
}

export function mcpAuthTokenPlaceholder(scheme: MCPAuthScheme): string {
  return scheme === "basic"
    ? "Paste Base64 of user:password"
    : "Paste API token";
}
