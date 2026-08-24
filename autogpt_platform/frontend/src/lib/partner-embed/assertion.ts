import { createRemoteJWKSet, jwtVerify } from "jose";
import { z } from "zod";

import type { PartnerEmbedConfig, VerifiedPartnerIdentity } from "./types";

const jwksByURL = new Map<string, ReturnType<typeof createRemoteJWKSet>>();

const partnerClaimsSchema = z.object({
  sub: z.string().min(1),
  account_id: z.string().min(1),
  email: z.string().email(),
  name: z.string().min(1),
  account_name: z.string().min(1),
  roles: z.array(z.string()).default([]),
  jti: z.string().min(1),
  exp: z.number().int().positive(),
  iat: z.number().int().positive(),
});

export async function verifyPartnerAssertion(
  assertion: string,
  config: PartnerEmbedConfig,
): Promise<VerifiedPartnerIdentity> {
  const { payload } = await jwtVerify(assertion, getJWKS(config.jwksURL), {
    issuer: config.issuer,
    audience: config.audience,
    algorithms: config.algorithms,
    maxTokenAge: "90s",
    clockTolerance: 5,
  });
  const claims = partnerClaimsSchema.parse(payload);
  return {
    partnerID: config.partnerID,
    externalSubject: claims.sub,
    externalAccountID: claims.account_id,
    email: claims.email,
    displayName: claims.name,
    accountName: claims.account_name,
    isAdmin: claims.roles.includes("admin"),
    jwtID: claims.jti,
    expiresAt: claims.exp,
  };
}

function getJWKS(url: string) {
  const existing = jwksByURL.get(url);
  if (existing) return existing;
  const jwks = createRemoteJWKSet(new URL(url));
  jwksByURL.set(url, jwks);
  return jwks;
}
