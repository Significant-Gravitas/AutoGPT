import { randomUUID } from "node:crypto";

import {
  calculateJwkThumbprint,
  exportJWK,
  generateKeyPair,
  SignJWT,
  type JWK,
} from "jose";

export interface PartnerUser {
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
  sign(user: PartnerUser): Promise<string>;
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
    async sign(user) {
      const issuedAt = Math.floor(Date.now() / 1000);
      return new SignJWT({
        account_id: user.accountID,
        name: user.name,
        account_name: user.accountName,
        roles: user.roles,
        capabilities: user.capabilities,
      })
        .setProtectedHeader({ alg: "RS256", kid: keyID, typ: "JWT" })
        .setIssuer(issuer)
        .setAudience(audience)
        .setSubject(user.subject)
        .setJti(randomUUID())
        .setIssuedAt(issuedAt)
        .setExpirationTime(issuedAt + 60)
        .sign(privateKey);
    },
  };
}
