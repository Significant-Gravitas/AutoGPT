import type { MCPServerMetadata } from "@/app/api/__generated__/models/mCPServerMetadata";

export type MCPPresetAuthMethod = NonNullable<
  MCPServerMetadata["auth_methods"]
>[number];

const METHOD_LABELS: Record<MCPPresetAuthMethod, string> = {
  none: "No sign-in",
  oauth: "Sign in",
  bearer: "API token",
  basic: "Basic authentication",
};

export function getMCPPresetAuthMethods(
  server: MCPServerMetadata,
): MCPPresetAuthMethod[] {
  if (server.auth_methods?.length) return server.auth_methods;
  if (server.auth_mode === "none") return ["none"];
  if (server.auth_mode === "oauth") return ["oauth"];
  if (server.auth_mode === "token") return ["bearer", "basic"];
  return ["oauth", "bearer", "basic"];
}

export function mcpPresetMethodLabel(method: MCPPresetAuthMethod) {
  return METHOD_LABELS[method];
}

export function mcpOAuthScopes(
  defaults: string[] | null | undefined,
  optionalWrites: string[],
  allowChanges: boolean,
) {
  if (defaults == null) return undefined;
  return [...new Set([...defaults, ...(allowChanges ? optionalWrites : [])])];
}
