import {
  buildCredentialBody,
  canSave,
  initialValues,
  isValidHost,
  type SheetTarget,
} from "../helpers";

function target(extra: Partial<SheetTarget>): SheetTarget {
  return {
    rowKey: "k",
    kind: "api_key",
    provider: "openweathermap",
    providerName: "OpenWeather",
    host: null,
    serverURL: null,
    ...extra,
  };
}

describe("buildCredentialBody", () => {
  it("names an API key after the service when no name was typed", () => {
    const sheet = target({});
    expect(buildCredentialBody(sheet, { ...initialValues(sheet), secret: " abc \n" })).toEqual({
      type: "api_key",
      title: "OpenWeather key",
      api_key: "abc",
    });
  });

  it("keeps a password exactly as typed", () => {
    const sheet = target({ kind: "user_password", provider: "reddit", providerName: "Reddit" });
    const body = buildCredentialBody(sheet, {
      ...initialValues(sheet),
      username: " me ",
      secret: " p@ss ",
      title: "Work",
    });
    expect(body).toEqual({ type: "user_password", title: "Work", username: "me", password: " p@ss " });
  });

  it("adds the server a database login is for", () => {
    const sheet = target({ kind: "user_password", provider: "database", providerName: "Database" });
    const values = { ...initialValues(sheet), username: "me", secret: "pw", scheme: "mysql" };
    expect(buildCredentialBody(sheet, { ...values, server: "not a host!" })).toBeNull();
    expect(buildCredentialBody(sheet, { ...values, server: "DB.example.com:3306" })).toMatchObject({
      scope: "mysql://db.example.com:3306",
    });
  });

  it("builds one header for a host", () => {
    const sheet = target({ kind: "host_scoped", provider: "http", host: "api.example.com" });
    expect(buildCredentialBody(sheet, { ...initialValues(sheet), secret: "Bearer t" })).toEqual({
      type: "host_scoped",
      title: "api.example.com",
      host: "api.example.com",
      headers: { Authorization: "Bearer t" },
    });
  });

  it("waits for every required field", () => {
    const sheet = target({});
    expect(buildCredentialBody(sheet, initialValues(sheet))).toBeNull();
  });
});

describe("canSave", () => {
  it("needs a server URL and a token for MCP", () => {
    const sheet = target({ kind: "mcp_token", provider: "mcp", serverURL: "https://mcp.example.com" });
    expect(canSave(sheet, { ...initialValues(sheet), secret: "t" })).toBe(true);
    expect(canSave({ ...sheet, serverURL: null }, { ...initialValues(sheet), secret: "t" })).toBe(false);
  });
});

describe("isValidHost", () => {
  it.each([
    ["api.example.com", true],
    ["api.example.com:8443", true],
    ["https://api.example.com", false],
    ["api.example.com/v1", false],
    ["  ", false],
  ])("%s → %s", (host, isValid) => {
    expect(isValidHost(host)).toBe(isValid);
  });
});
