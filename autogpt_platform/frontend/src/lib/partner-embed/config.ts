import { decodeJwt } from "jose";
import { z } from "zod";

import type { PartnerEmbedConfig } from "./types";

const configSchema = z.object({
  partnerID: z.string().min(1),
  issuer: z.string().url(),
  jwksURL: z.string().url(),
  audience: z.string().min(1),
  algorithms: z.array(z.literal("RS256")).default(["RS256"]),
  allowedCapabilities: z.array(z.string().min(1)).max(256).default([]),
});

const forwardingDigitalCapabilities = [
  "jobs.read",
  "reports.read",
  "documents.read",
  "documents.write",
  "agents.create",
  "agents.run",
  "agents.schedule",
  "autogpt:block:c0a8e994-ebf1-4a9c-a4d8-89d09c86741b",
  "autogpt:block:363ae599-353e-4804-937e-b2ee3cef3da4",
  "autogpt:block:b1ab9b19-67a6-406d-abf5-2dba76d00c79",
];

const localConfigs: PartnerEmbedConfig[] = [8787, 8788].map((port) => ({
  partnerID: "forwarding-digital",
  issuer: `http://localhost:${port}`,
  jwksURL: `http://localhost:${port}/.well-known/jwks.json`,
  audience: "autogpt-partner-exchange",
  algorithms: ["RS256"],
  allowedCapabilities: forwardingDigitalCapabilities,
}));

export class PartnerEmbedConfigurationError extends Error {}

export function getPartnerEmbedConfig(assertion: string): PartnerEmbedConfig {
  let configs: PartnerEmbedConfig[];
  try {
    configs = readConfigs();
  } catch {
    throw new PartnerEmbedConfigurationError(
      "Partner embedding is not configured",
    );
  }

  let issuer: string | undefined;
  try {
    issuer = decodeJwt(assertion).iss;
  } catch {
    throw new Error("Partner assertion is malformed");
  }
  if (!issuer) throw new Error("Partner assertion is missing an issuer");

  const config = configs.find((candidate) => candidate.issuer === issuer);
  if (!config) throw new Error("Partner assertion issuer is not configured");
  return config;
}

function readConfigs(): PartnerEmbedConfig[] {
  const serialized = process.env.PARTNER_EMBED_CONFIGS?.trim();
  if (serialized) {
    return z.array(configSchema).min(1).parse(JSON.parse(serialized));
  }

  const legacy = readLegacyConfig();
  if (legacy) return [legacy];
  if (process.env.NODE_ENV !== "production") return localConfigs;
  throw new Error("PARTNER_EMBED_CONFIGS is required for partner embedding");
}

function readLegacyConfig(): PartnerEmbedConfig | undefined {
  const partnerID = process.env.PARTNER_EMBED_ID?.trim();
  const issuer = process.env.PARTNER_EMBED_ISSUER?.trim();
  const jwksURL = process.env.PARTNER_EMBED_JWKS_URL?.trim();
  const audience = process.env.PARTNER_EMBED_AUDIENCE?.trim();
  if (!partnerID && !issuer && !jwksURL && !audience) return undefined;
  return configSchema.parse({ partnerID, issuer, jwksURL, audience });
}
