import { createLocalJWKSet, jwtVerify } from "jose";
import { describe, expect, it } from "vitest";

import { createPartnerAssertionIssuer } from "./assertion.js";

describe("partner assertion issuer", () => {
  it("issues a short-lived, audience-bound assertion from the partner identity", async () => {
    const issuer = await createPartnerAssertionIssuer(
      "http://localhost:8787",
      "autogpt-partner-exchange",
    );
    const token = await issuer.sign({
      subject: "fd-user-1042",
      accountID: "fd-account-77",
      email: "alex@northstarfreight.com",
      name: "Alex Morgan",
      accountName: "Northstar Freight",
      roles: ["operator", "manager"],
    });

    const { payload } = await jwtVerify(token, createLocalJWKSet(issuer.jwks), {
      issuer: "http://localhost:8787",
      audience: "autogpt-partner-exchange",
      algorithms: ["RS256"],
    });

    expect(payload.sub).toBe("fd-user-1042");
    expect(payload.account_id).toBe("fd-account-77");
    expect(payload.exp! - payload.iat!).toBe(60);
  });
});
