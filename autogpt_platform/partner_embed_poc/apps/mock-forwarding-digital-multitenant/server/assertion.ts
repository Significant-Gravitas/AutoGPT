import { randomUUID } from "node:crypto";

import {
  calculateJwkThumbprint,
  exportJWK,
  generateKeyPair,
  SignJWT,
  type JWK,
} from "jose";

export interface PartnerIdentity {
  subject: string;
  accountID: string;
  email: string;
  name: string;
  accountName: string;
  roles: string[];
  capabilities: string[];
}

export interface PartnerAssertionIssuer {
  jwks: { keys: JWK[] };
  sign(identity: PartnerIdentity): Promise<string>;
}

export async function createPartnerAssertionIssuer(
  issuer: string,
  audience: string,
): Promise<PartnerAssertionIssuer> {
  const { privateKey, publicKey } = await generateKeyPair("RS256");
  const publicJWK = await exportJWK(publicKey);
  const keyID = await calculateJwkThumbprint(publicJWK);
  publicJWK.alg = "RS256";
  publicJWK.kid = keyID;
  publicJWK.use = "sig";

  return {
    jwks: { keys: [publicJWK] },
    async sign(identity) {
      const issuedAt = Math.floor(Date.now() / 1000);
      return new SignJWT({
        account_id: identity.accountID,
        email: identity.email,
        name: identity.name,
        account_name: identity.accountName,
        roles: identity.roles,
        capabilities: identity.capabilities,
      })
        .setProtectedHeader({ alg: "RS256", kid: keyID, typ: "JWT" })
        .setIssuer(issuer)
        .setAudience(audience)
        .setSubject(identity.subject)
        .setJti(randomUUID())
        .setIssuedAt(issuedAt)
        .setExpirationTime(issuedAt + 60)
        .sign(privateKey);
    },
  };
}
