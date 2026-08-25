import { beforeEach, describe, expect, it, vi } from "vitest";

const getConfigMock = vi.fn();
const verifyAssertionMock = vi.fn();
const provisionIdentityMock = vi.fn();
const mintTokenMock = vi.fn();

vi.mock("@/lib/partner-embed/config", () => ({
  getPartnerEmbedConfig: (...args: unknown[]) => getConfigMock(...args),
  PartnerEmbedConfigurationError: class extends Error {},
}));
vi.mock("@/lib/partner-embed/assertion", () => ({
  verifyPartnerAssertion: (...args: unknown[]) => verifyAssertionMock(...args),
}));
vi.mock("@/lib/partner-embed/provision", () => ({
  provisionPartnerIdentity: (...args: unknown[]) =>
    provisionIdentityMock(...args),
}));
vi.mock("@/lib/partner-embed/embed-token", () => ({
  mintPartnerEmbedToken: (...args: unknown[]) => mintTokenMock(...args),
  PARTNER_EMBED_TOKEN_TTL_SECONDS: 300,
  partnerEmbedTokenTTL: () => 300,
}));

import { POST } from "../route";

const config = {
  partnerID: "logistics-partner",
  issuer: "http://localhost:8787",
  jwksURL: "http://localhost:8787/.well-known/jwks.json",
  audience: "autogpt-partner-exchange",
  algorithms: ["RS256"],
};

const identity = {
  partnerID: "logistics-partner",
  externalSubject: "user-123",
  externalAccountID: "forwarder-42",
  displayName: "Jordan Avery",
  accountName: "Acme Forwarding",
  isAdmin: true,
  capabilities: ["jobs.read", "reports.read"],
  jwtID: "assertion-1",
  expiresAt: 1_788_000_000,
};

const provisioned = {
  userID: "0234dc86-e049-5c61-8b7e-826f7a7c225f",
  organizationID: "70d89c3b-2af3-5f56-8a21-2951b469ba95",
  teamID: "600e3708-3a7a-54c7-b527-53d2c62b8d5b",
};

function request(body: unknown): Request {
  return new Request("http://localhost:3000/api/embed/token", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
}

beforeEach(() => {
  getConfigMock.mockReset().mockReturnValue(config);
  verifyAssertionMock.mockReset().mockResolvedValue(identity);
  provisionIdentityMock.mockReset().mockResolvedValue(provisioned);
  mintTokenMock.mockReset().mockResolvedValue("embed-token");
});

describe("POST /api/embed/token", () => {
  it("exchanges a verified assertion for a restricted embed token", async () => {
    const response = await POST(request({ assertion: "partner-jwt" }));

    expect(response.status).toBe(200);
    expect(response.headers.get("cache-control")).toBe("no-store");
    expect(await response.json()).toEqual({
      access_token: "embed-token",
      token_type: "Bearer",
      expires_in: 300,
    });
    expect(getConfigMock).toHaveBeenCalledWith("partner-jwt");
    expect(verifyAssertionMock).toHaveBeenCalledWith("partner-jwt", config);
    expect(provisionIdentityMock).toHaveBeenCalledWith(identity);
    expect(mintTokenMock).toHaveBeenCalledWith(identity, provisioned);
  });

  it("rejects an invalid partner assertion without provisioning", async () => {
    verifyAssertionMock.mockRejectedValue(new Error("JWT verification failed"));

    const response = await POST(request({ assertion: "invalid" }));

    expect(response.status).toBe(401);
    expect(provisionIdentityMock).not.toHaveBeenCalled();
    expect(mintTokenMock).not.toHaveBeenCalled();
  });

  it("rejects an issuer that is not configured before verification", async () => {
    getConfigMock.mockImplementation(() => {
      throw new Error("Partner assertion issuer is not configured");
    });

    const response = await POST(request({ assertion: "unknown-issuer" }));

    expect(response.status).toBe(401);
    expect(verifyAssertionMock).not.toHaveBeenCalled();
  });

  it("rejects a missing assertion", async () => {
    const response = await POST(request({}));

    expect(response.status).toBe(400);
    expect(verifyAssertionMock).not.toHaveBeenCalled();
  });
});
