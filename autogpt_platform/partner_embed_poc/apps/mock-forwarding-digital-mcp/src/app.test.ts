import { afterEach, describe, expect, it } from "vitest";

import { buildApp } from "./app.js";
import { createAccessToken, type PartnerMCPClaims } from "./auth.js";

const SECRET = "forwarding-digital-mcp-test-secret";
const apps: ReturnType<typeof buildApp>[] = [];

afterEach(async () => {
  await Promise.all(apps.splice(0).map((app) => app.close()));
});

function app() {
  const instance = buildApp({ sharedSecret: SECRET });
  apps.push(instance);
  return instance;
}

function token(
  externalAccountID: string,
  overrides: Partial<PartnerMCPClaims> = {},
) {
  return createAccessToken(
    {
      version: 1,
      partner_id: "forwarding-digital",
      user_id: "autogpt-user-1",
      organization_id: "autogpt-org-1",
      external_account_id: externalAccountID,
      capabilities: ["jobs.read", "reports.read"],
      exp: Math.floor(Date.now() / 1000) + 60,
      ...overrides,
    },
    SECRET,
  );
}

async function initialize(
  instance: ReturnType<typeof buildApp>,
  bearer: string,
) {
  const response = await instance.inject({
    method: "POST",
    url: "/mcp",
    headers: { authorization: "Bearer " + bearer },
    payload: {
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: {
        protocolVersion: "2025-03-26",
        capabilities: {},
        clientInfo: { name: "test", version: "1" },
      },
    },
  });
  return {
    response,
    sessionID: response.headers["mcp-session-id"] as string,
  };
}

async function callSummary(
  instance: ReturnType<typeof buildApp>,
  bearer: string,
  sessionID: string,
  args: Record<string, unknown> = {},
) {
  return instance.inject({
    method: "POST",
    url: "/mcp",
    headers: {
      authorization: "Bearer " + bearer,
      "mcp-session-id": sessionID,
    },
    payload: {
      jsonrpc: "2.0",
      id: 2,
      method: "tools/call",
      params: { name: "get_operations_summary", arguments: args },
    },
  });
}

describe("Forwarding Digital MCP", () => {
  it("rejects requests without a valid bearer token", async () => {
    const response = await app().inject({
      method: "POST",
      url: "/mcp",
      payload: { jsonrpc: "2.0", id: 1, method: "initialize" },
    });

    expect(response.statusCode).toBe(401);
  });

  it("returns Northstar data only for the Northstar tenant token", async () => {
    const instance = app();
    const bearer = token("fd-account-77");
    const initialized = await initialize(instance, bearer);
    const response = await callSummary(instance, bearer, initialized.sessionID);
    const body = response.json();
    const result = JSON.parse(body.result.content[0].text);

    expect(initialized.response.statusCode).toBe(200);
    expect(result.account).toEqual({
      id: "fd-account-77",
      name: "Northstar Freight",
    });
    expect(result.active_jobs).toBe(148);
    expect(response.body).not.toContain("Harbour & Rail Logistics");
  });

  it("returns different data for the Harbour tenant token", async () => {
    const instance = app();
    const bearer = token("fd-account-88");
    const initialized = await initialize(instance, bearer);
    const response = await callSummary(instance, bearer, initialized.sessionID);
    const result = JSON.parse(response.json().result.content[0].text);

    expect(result.account).toEqual({
      id: "fd-account-88",
      name: "Harbour & Rail Logistics",
    });
    expect(result.active_jobs).toBe(61);
    expect(response.body).not.toContain("Northstar Freight");
  });

  it("rejects a model-supplied tenant override", async () => {
    const instance = app();
    const bearer = token("fd-account-77");
    const initialized = await initialize(instance, bearer);
    const response = await callSummary(
      instance,
      bearer,
      initialized.sessionID,
      { external_account_id: "fd-account-88" },
    );

    expect(response.json()).toMatchObject({
      error: { code: -32602, message: "Invalid params" },
    });
    expect(response.body).not.toContain("Harbour & Rail Logistics");
  });

  it("prevents an MCP session from being reused across tenants", async () => {
    const instance = app();
    const northstar = token("fd-account-77");
    const harbour = token("fd-account-88");
    const initialized = await initialize(instance, northstar);
    const response = await callSummary(
      instance,
      harbour,
      initialized.sessionID,
    );

    expect(response.statusCode).toBe(403);
    expect(response.json()).toEqual({ error: "MCP session tenant mismatch" });
  });

  it("rejects expired and tampered tokens", async () => {
    const instance = app();
    const expired = token("fd-account-77", { exp: 1 });
    const valid = token("fd-account-77");
    const tampered = valid.slice(0, -1) + (valid.endsWith("a") ? "b" : "a");

    for (const bearer of [expired, tampered]) {
      const initialized = await initialize(instance, bearer);
      expect(initialized.response.statusCode).toBe(401);
    }
  });

  it("filters and rejects tools outside the signed user capabilities", async () => {
    const instance = app();
    const bearer = token("fd-account-77", { capabilities: ["jobs.read"] });
    const initialized = await initialize(instance, bearer);
    const listed = await instance.inject({
      method: "POST",
      url: "/mcp",
      headers: {
        authorization: "Bearer " + bearer,
        "mcp-session-id": initialized.sessionID,
      },
      payload: { jsonrpc: "2.0", id: 2, method: "tools/list" },
    });
    const called = await callSummary(instance, bearer, initialized.sessionID);

    expect(
      listed.json().result.tools.map((tool: { name: string }) => tool.name),
    ).toEqual(["list_arrivals", "list_exceptions"]);
    expect(called.json()).toMatchObject({
      error: { code: -32001, message: "Capability denied" },
    });
  });
});
