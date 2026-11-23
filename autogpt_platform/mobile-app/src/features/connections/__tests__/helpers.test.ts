import type { CredentialMeta } from "@/lib/api/integrations";
import { FRONTEND_URL } from "@/lib/env";

import {
  connectMethod,
  credentialFits,
  hostScopeMatches,
  logoID,
  logoURL,
  parseCredentialRequests,
  requestHost,
  tileLetters,
  type CredentialRequest,
} from "../helpers";

function request(extra: Partial<CredentialRequest> = {}): CredentialRequest {
  return {
    key: "credentials",
    provider: "github",
    providerName: "GitHub",
    types: ["oauth2"],
    scopes: [],
    discriminatorValues: [],
    gap: null,
    expertGrant: null,
    ...extra,
  };
}

function credential(extra: Partial<CredentialMeta> = {}): CredentialMeta {
  return {
    id: "cred-1",
    provider: "github",
    type: "oauth2",
    title: "octocat",
    username: null,
    scopes: ["repo"],
    host: null,
    isSystem: false,
    serviceName: null,
    serviceIcon: null,
    ...extra,
  };
}

describe("parseCredentialRequests", () => {
  it("reads entries and drops ones the app can't act on", () => {
    const requests = parseCredentialRequests({
      github_credentials: {
        provider: "github",
        provider_name: "GitHub",
        types: ["oauth2", "made_up"],
        scopes: ["repo"],
        credential_gap: { kind: "missing_scopes", credential_id: "c1" },
        expert_grant: { expert_id: "e1", credentials: [{ id: "c2" }, { title: "no id" }] },
      },
      no_provider: { types: ["api_key"] },
      unknown_type: { provider: "x", types: ["telepathy"] },
    });

    expect(requests).toEqual([
      {
        key: "github_credentials",
        provider: "github",
        providerName: "GitHub",
        types: ["oauth2"],
        scopes: ["repo"],
        discriminatorValues: [],
        gap: { kind: "missing_scopes", credentialID: "c1", credentialTitle: null },
        expertGrant: { expertID: "e1", credentials: [{ id: "c2", title: "c2", type: "api_key" }] },
      },
    ]);
  });

  it("names a provider from its slug when the entry has no name", () => {
    const [entry] = parseCredentialRequests({ k: { provider: "open_weather_map", types: ["api_key"] } });
    expect(entry.providerName).toBe("Open Weather Map");
  });

  it("returns nothing for anything that isn't a record", () => {
    expect(parseCredentialRequests(null)).toEqual([]);
    expect(parseCredentialRequests([1, 2])).toEqual([]);
  });
});

describe("connectMethod", () => {
  it.each([
    [{ types: ["oauth2", "api_key"] }, "oauth"],
    [{ types: ["api_key"] }, "api_key"],
    [{ types: ["user_password"] }, "user_password"],
    [{ types: ["host_scoped"] }, "host_scoped"],
    [{ types: ["device_code"] }, "web"],
    [{ provider: "mcp", types: ["oauth2"] }, "mcp"],
    [{ provider: "codex", types: ["oauth2"] }, "web"],
    [{ provider: "ayrshare", types: ["api_key"] }, "web"],
  ] as [Partial<CredentialRequest>, string][])("%j → %s", (extra, method) => {
    expect(connectMethod(request(extra))).toBe(method);
  });
});

describe("credentialFits", () => {
  it("needs every scope the request asks for", () => {
    expect(credentialFits(credential(), request({ scopes: ["repo"] }))).toBe(true);
    expect(credentialFits(credential(), request({ scopes: ["repo", "gist"] }))).toBe(false);
  });

  it("never offers a system credential or another provider's", () => {
    expect(credentialFits(credential({ isSystem: true }), request())).toBe(false);
    expect(credentialFits(credential({ provider: "gitlab" }), request())).toBe(false);
  });

  it("matches an MCP account on its server URL", () => {
    const mcp = request({ provider: "mcp", discriminatorValues: ["https://mcp.example.com/sse"] });
    expect(
      credentialFits(credential({ provider: "mcp", host: "https://mcp.example.com/sse/" }), mcp),
    ).toBe(true);
    expect(
      credentialFits(credential({ provider: "mcp", host: "https://other.example.com" }), mcp),
    ).toBe(false);
  });

  it("matches a host-scoped key on its host", () => {
    const http = request({
      provider: "http",
      types: ["host_scoped"],
      discriminatorValues: ["https://api.example.com/v1/items"],
    });
    const key = credential({ provider: "http", type: "host_scoped", host: "api.example.com" });
    expect(credentialFits(key, http)).toBe(true);
    expect(credentialFits({ ...key, host: "api.other.com" }, http)).toBe(false);
  });
});

describe("hostScopeMatches", () => {
  it("keeps a port in the scope and matches it, as the backend does", () => {
    expect(hostScopeMatches("api.example.com:8443", "https://api.example.com:8443/data")).toBe(true);
    expect(hostScopeMatches("api.example.com:8443", "https://api.example.com/data")).toBe(false);
    expect(hostScopeMatches("api.example.com", "https://api.example.com:8443/data")).toBe(false);
    expect(hostScopeMatches("api.example.com", "https://API.example.com:443/data")).toBe(true);
  });

  it("lets a wildcard scope cover the domain and its subdomains", () => {
    expect(hostScopeMatches("*.example.com", "https://api.example.com/a")).toBe(true);
    expect(hostScopeMatches("*.example.com", "https://example.com/a")).toBe(true);
    expect(hostScopeMatches("*.example.com", "https://example.org/a")).toBe(false);
  });
});

describe("requestHost", () => {
  it("names the one host a host-scoped request is for", () => {
    const http = request({ types: ["host_scoped"], discriminatorValues: ["https://API.example.com/a"] });
    expect(requestHost(http)).toBe("api.example.com");
    expect(requestHost(request({ discriminatorValues: ["https://a.com"] }))).toBeNull();
  });

  it("keeps a port the scheme doesn't use by default, as the backend keys it", () => {
    const at = (url: string) => requestHost(request({ types: ["host_scoped"], discriminatorValues: [url] }));
    expect(at("https://api.example.com:8443/data")).toBe("api.example.com:8443");
    expect(at("https://api.example.com:443/data")).toBe("api.example.com");
    expect(at("http://api.example.com:443/data")).toBe("api.example.com:443");
  });
});

describe("tileLetters", () => {
  it.each([
    ["Gmail", "G"],
    ["GitHub", "GH"],
    ["OpenWeather", "OW"],
    ["Google Calendar", "GC"],
    ["", "?"],
  ])("%s → %s", (name, letters) => {
    expect(tileLetters(name)).toBe(letters);
  });
});

describe("logoID", () => {
  it("is the provider, lowercased", () => {
    expect(logoID("GitHub")).toBe("github");
    expect(logoID("google", "https://ignored.example.com")).toBe("google");
  });

  it("takes an MCP server's name from its host", () => {
    expect(logoID("mcp", "https://mcp.linear.app/mcp")).toBe("linear");
    expect(logoID("mcp", "https://Notion.com/sse")).toBe("notion");
  });

  it("is null for an MCP server without a host", () => {
    expect(logoID("mcp")).toBeNull();
    expect(logoID("mcp", null)).toBeNull();
    expect(logoID("mcp", "")).toBeNull();
  });
});

describe("logoURL", () => {
  it("points at the web's integration logos", () => {
    expect(logoURL("github")).toBe(`${FRONTEND_URL}/integrations/github.png`);
    expect(logoURL("../x")).toBe(`${FRONTEND_URL}/integrations/..%2Fx.png`);
  });
});
