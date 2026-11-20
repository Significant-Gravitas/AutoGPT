import {
  parseOAuthReturn,
  parsePendingFlow,
  PENDING_FLOW_MAX_AGE_MS,
  providerErrorMessage,
  type PendingFlow,
} from "../oauthHelpers";

describe("parseOAuthReturn", () => {
  it("reads the code, state and issuer", () => {
    expect(
      parseOAuthReturn(
        "autogpt-app://integrations/oauth?code=a%2Fb&state=native.xyz&iss=https%3A%2F%2Fauth.example.com",
      ),
    ).toEqual({
      code: "a/b",
      state: "native.xyz",
      iss: "https://auth.example.com",
      error: null,
      errorDescription: null,
    });
  });

  it("reads a provider's refusal", () => {
    expect(
      parseOAuthReturn(
        "autogpt-app://integrations/oauth?state=native.xyz&error=access_denied&error_description=User+said+no",
      ),
    ).toMatchObject({ code: null, error: "access_denied", errorDescription: "User said no" });
  });

  it("refuses any other URL, or one without a state", () => {
    expect(parseOAuthReturn("autogpt-app://somewhere?code=a&state=b")).toBeNull();
    expect(parseOAuthReturn("autogpt-app://integrations/oauthx?code=a&state=b")).toBeNull();
    expect(parseOAuthReturn("autogpt-app://integrations/oauth?code=a")).toBeNull();
    expect(parseOAuthReturn("autogpt-app://integrations/oauth?code=%E0%A4%A&state=b")).toBeNull();
  });
});

describe("parsePendingFlow", () => {
  const now = 1_000_000_000;
  const flow: PendingFlow = {
    state: "native.xyz",
    target: { kind: "provider", provider: "github", providerName: "GitHub", scopes: ["repo"] },
    sessionID: "session-1",
    cardKey: "session-1:call-1",
    rowKey: "github_credentials",
    expertID: null,
    savedAt: now - 1_000,
  };

  it("reads back a saved flow", () => {
    expect(parsePendingFlow(JSON.stringify(flow), now)).toEqual(flow);
  });

  it("reads back an MCP flow", () => {
    const mcp = { ...flow, target: { kind: "mcp", serverURL: "https://mcp.example.com", providerName: "Example" } };
    expect(parsePendingFlow(JSON.stringify(mcp), now)?.target).toEqual(mcp.target);
  });

  it("drops a flow that is too old, unreadable or incomplete", () => {
    const old = { ...flow, savedAt: now - PENDING_FLOW_MAX_AGE_MS - 1 };
    expect(parsePendingFlow(JSON.stringify(old), now)).toBeNull();
    expect(parsePendingFlow("{not json", now)).toBeNull();
    expect(parsePendingFlow(JSON.stringify({ ...flow, state: "" }), now)).toBeNull();
    expect(parsePendingFlow(JSON.stringify({ ...flow, target: { kind: "mcp" } }), now)).toBeNull();
    expect(parsePendingFlow(null, now)).toBeNull();
  });
});

describe("providerErrorMessage", () => {
  it("stays quiet when the user declined", () => {
    expect(providerErrorMessage("access_denied", "GitHub")).toBeNull();
  });

  it("names the provider otherwise", () => {
    expect(providerErrorMessage("server_error", "GitHub")).toBe(
      "GitHub didn't finish the sign-in. Try again.",
    );
  });
});
