import type { PartnerEmbedConfig } from "./types";

export function getPartnerEmbedConfig(): PartnerEmbedConfig {
  const local = process.env.NODE_ENV !== "production";
  return {
    partnerID: readConfig(
      "PARTNER_EMBED_ID",
      local ? "forwarding-digital" : undefined,
    ),
    issuer: readConfig(
      "PARTNER_EMBED_ISSUER",
      local ? "http://localhost:8787" : undefined,
    ),
    jwksURL: readConfig(
      "PARTNER_EMBED_JWKS_URL",
      local ? "http://localhost:8787/.well-known/jwks.json" : undefined,
    ),
    audience: readConfig(
      "PARTNER_EMBED_AUDIENCE",
      local ? "autogpt-partner-exchange" : undefined,
    ),
    algorithms: ["RS256"],
  };
}

function readConfig(name: string, fallback?: string): string {
  const value = process.env[name]?.trim() || fallback;
  if (!value) {
    throw new Error(`${name} is required for partner embedding`);
  }
  return value;
}
