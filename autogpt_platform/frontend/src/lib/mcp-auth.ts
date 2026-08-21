export type MCPAuthScheme = "bearer" | "basic";

const AUTHORIZATION_HEADER_PREFIX = /^authorization\s*:\s*/i;
const SUPPORTED_SCHEME_PREFIX = /^(bearer|basic)(?:\s+|$)/i;

/** Detect an explicit Basic/Bearer scheme in a pasted credential. */
export function detectMCPAuthScheme(value: string): MCPAuthScheme | null {
  const candidate = value
    .trim()
    .replace(AUTHORIZATION_HEADER_PREFIX, "")
    .trim();
  const match = SUPPORTED_SCHEME_PREFIX.exec(candidate);
  if (!match) {
    return null;
  }
  return match[1].toLowerCase() as MCPAuthScheme;
}

/**
 * Prepare the existing token API payload without changing Bearer compatibility.
 * Bare Bearer tokens are sent unchanged; Basic credentials get an explicit prefix.
 * Explicitly prefixed values and complete Authorization headers are preserved.
 */
export function prepareMCPAuthCredential(
  value: string,
  scheme: MCPAuthScheme,
): string {
  const credential = value.trim();
  if (
    !credential ||
    AUTHORIZATION_HEADER_PREFIX.test(credential) ||
    detectMCPAuthScheme(credential)
  ) {
    return credential;
  }
  return scheme === "basic" ? `Basic ${credential}` : credential;
}
