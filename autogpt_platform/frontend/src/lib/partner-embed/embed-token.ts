import { signJWT } from "better-auth/plugins/jwt";

import { JWKS_ALG } from "@/lib/auth/service-token";
import type {
  ProvisionedPartnerIdentity,
  VerifiedPartnerIdentity,
} from "./types";

export const PARTNER_EMBED_TOKEN_AUDIENCE = "autogpt-partner-embed";
export const PARTNER_EMBED_TOKEN_TTL_SECONDS = 300;
export function partnerEmbedTokenTTL(
  identity: VerifiedPartnerIdentity,
  issuedAt = Math.floor(Date.now() / 1000),
): number {
  return Math.max(
    1,
    Math.min(PARTNER_EMBED_TOKEN_TTL_SECONDS, identity.expiresAt - issuedAt),
  );
}

type SignJWTContext = Parameters<typeof signJWT>[0];

export async function mintPartnerEmbedToken(
  identity: VerifiedPartnerIdentity,
  provisioned: ProvisionedPartnerIdentity,
): Promise<string> {
  const { auth } = await import("@/lib/auth/auth");
  const context = await auth.$context;
  const issuedAt = Math.floor(Date.now() / 1000);
  const ttl = partnerEmbedTokenTTL(identity, issuedAt);
  return signJWT({ context } as unknown as SignJWTContext, {
    options: { jwks: { keyPairConfig: { alg: JWKS_ALG } } },
    payload: {
      sub: provisioned.userID,
      aud: PARTNER_EMBED_TOKEN_AUDIENCE,
      token_use: "partner_embed",
      partner_id: identity.partnerID,
      organization_id: provisioned.organizationID,
      team_id: provisioned.teamID,
      external_account_id: identity.externalAccountID,
      scope: "embed:chat",
      capabilities: identity.capabilities,
      iat: issuedAt,
      exp: issuedAt + ttl,
    },
  });
}
