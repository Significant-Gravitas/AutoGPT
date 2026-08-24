import { afterEach, describe, expect, it } from "vitest";

import { getPartnerEmbedConfig } from "./config";

const originalConfigs = process.env.PARTNER_EMBED_CONFIGS;

afterEach(() => {
  if (originalConfigs === undefined) delete process.env.PARTNER_EMBED_CONFIGS;
  else process.env.PARTNER_EMBED_CONFIGS = originalConfigs;
});

function assertion(issuer: string) {
  return `eyJhbGciOiJSUzI1NiJ9.${Buffer.from(JSON.stringify({ iss: issuer })).toString("base64url")}.signature`;
}

describe("getPartnerEmbedConfig", () => {
  it("selects an allowlisted issuer", () => {
    process.env.PARTNER_EMBED_CONFIGS = JSON.stringify([
      {
        partnerID: "forwarding-digital",
        issuer: "https://partner.example.com",
        jwksURL: "https://partner.internal/.well-known/jwks.json",
        audience: "autogpt-partner-exchange",
      },
    ]);

    expect(
      getPartnerEmbedConfig(assertion("https://partner.example.com")),
    ).toEqual({
      partnerID: "forwarding-digital",
      issuer: "https://partner.example.com",
      jwksURL: "https://partner.internal/.well-known/jwks.json",
      audience: "autogpt-partner-exchange",
      algorithms: ["RS256"],
    });
  });

  it("rejects an issuer that is not allowlisted", () => {
    process.env.PARTNER_EMBED_CONFIGS = JSON.stringify([
      {
        partnerID: "forwarding-digital",
        issuer: "https://partner.example.com",
        jwksURL: "https://partner.internal/.well-known/jwks.json",
        audience: "autogpt-partner-exchange",
      },
    ]);

    expect(() =>
      getPartnerEmbedConfig(assertion("https://attacker.example.com")),
    ).toThrow("issuer is not configured");
  });
});
