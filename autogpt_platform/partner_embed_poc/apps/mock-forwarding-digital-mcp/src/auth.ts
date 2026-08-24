import { createHmac, timingSafeEqual, type BinaryLike } from "node:crypto";

export interface PartnerMCPClaims {
  version: 1;
  partner_id: "forwarding-digital";
  user_id: string;
  organization_id: string;
  external_account_id: string;
  exp: number;
}

export function createAccessToken(
  claims: PartnerMCPClaims,
  secret: string,
): string {
  const payload = Buffer.from(JSON.stringify(claims)).toString("base64url");
  return payload + "." + sign(payload, secret);
}

export function verifyAccessToken(
  token: string,
  secret: string,
  now = Math.floor(Date.now() / 1000),
): PartnerMCPClaims | undefined {
  const parts = token.split(".");
  if (parts.length !== 2) return undefined;
  const [payload, signature] = parts;
  if (!payload || !signature) return undefined;
  if (!isCanonicalBase64Url(payload) || !isCanonicalBase64Url(signature)) {
    return undefined;
  }

  const expected = Buffer.from(sign(payload, secret), "base64url");
  const received = Buffer.from(signature, "base64url");
  if (
    expected.length !== received.length ||
    !timingSafeEqual(expected, received)
  ) {
    return undefined;
  }

  let value: unknown;
  try {
    value = JSON.parse(Buffer.from(payload, "base64url").toString("utf8"));
  } catch {
    return undefined;
  }
  if (!isClaims(value) || value.exp <= now) return undefined;
  return value;
}

export function bearerToken(
  authorization: string | undefined,
): string | undefined {
  if (!authorization?.startsWith("Bearer ")) return undefined;
  const token = authorization.slice("Bearer ".length).trim();
  return token || undefined;
}

function sign(payload: BinaryLike, secret: string) {
  return createHmac("sha256", secret).update(payload).digest("base64url");
}

function isCanonicalBase64Url(value: string) {
  return Buffer.from(value, "base64url").toString("base64url") === value;
}

function isClaims(value: unknown): value is PartnerMCPClaims {
  if (!value || typeof value !== "object") return false;
  const claims = value as Partial<PartnerMCPClaims>;
  return (
    claims.version === 1 &&
    claims.partner_id === "forwarding-digital" &&
    isNonEmptyString(claims.user_id) &&
    isNonEmptyString(claims.organization_id) &&
    isNonEmptyString(claims.external_account_id) &&
    typeof claims.exp === "number" &&
    Number.isInteger(claims.exp)
  );
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.length > 0;
}
