import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { initMock } = vi.hoisted(() => ({ initMock: vi.fn() }));

const BODY_MARKER = "marker-4f1c";

vi.mock("@sentry/nextjs", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@sentry/nextjs")>();
  function stubIntegration(name: string) {
    return vi.fn(() => ({ name }));
  }

  return {
    init: initMock,
    requestDataIntegration: actual.requestDataIntegration,
    anrIntegration: stubIntegration("Anr"),
    captureConsoleIntegration: stubIntegration("CaptureConsole"),
    extraErrorDataIntegration: stubIntegration("ExtraErrorData"),
  };
});

beforeEach(() => {
  initMock.mockReset();
  vi.resetModules();
});

afterEach(() => {
  vi.unstubAllEnvs();
});

describe.each([
  ["edge", () => import("../../sentry.edge.config")],
  ["server", () => import("../../sentry.server.config")],
])("Sentry %s config", (_, load) => {
  it("drops the transactions of relaying a browser envelope", async () => {
    await load();

    const { beforeSendTransaction } = initMock.mock.calls[0][0];
    const serverRelay = {
      transaction: "POST /relay/snentry",
      request: { url: "https://platform.agpt.co/relay/snentry?o=1&p=2" },
    };
    const middlewareRelay = {
      transaction: "middleware POST",
      contexts: { trace: { data: { "http.target": "/relay/snentry" } } },
    };
    const page = {
      transaction: "middleware POST",
      request: { url: "https://platform.agpt.co/marketplace" },
      contexts: { trace: { data: { "http.target": "/marketplace" } } },
    };
    const malformed = { request: { url: "http://[" } };

    expect(beforeSendTransaction(serverRelay)).toBeNull();
    expect(beforeSendTransaction(middlewareRelay)).toBeNull();
    expect(beforeSendTransaction(page)).toBe(page);
    expect(beforeSendTransaction(malformed)).toBe(malformed);
  });

  it("records no request cookies", async () => {
    await load();

    const { integrations } = initMock.mock.calls[0][0];
    const requestData = integrations.find(
      (integration: { name: string }) => integration.name === "RequestData",
    );
    const event = requestData.processEvent(
      {
        sdkProcessingMetadata: {
          normalizedRequest: {
            method: "GET",
            url: "https://platform.agpt.co/library",
            headers: { cookie: "a=b", "user-agent": "ua" },
            cookies: { a: "b" },
          },
        },
      },
      {},
      { getOptions: () => ({}) },
    );

    expect(event.request.url).toBe("https://platform.agpt.co/library");
    expect(event.request.cookies).toBeUndefined();
    expect(event.request.headers).toEqual({ "user-agent": "ua" });
  });

  it("records no request body", async () => {
    await load();

    const { integrations } = initMock.mock.calls[0][0];
    const requestData = integrations.find(
      (integration: { name: string }) => integration.name === "RequestData",
    );
    const event = requestData.processEvent(
      {
        sdkProcessingMetadata: {
          normalizedRequest: {
            method: "POST",
            url: "https://platform.agpt.co/api/auth/sign-in/email",
            headers: { "content-type": "application/json" },
            data: JSON.stringify({ email: "a@b.c", password: BODY_MARKER }),
          },
        },
      },
      {},
      { getOptions: () => ({}) },
    );

    expect(event.request.url).toBe(
      "https://platform.agpt.co/api/auth/sign-in/email",
    );
    // The SDK's input, which createEventEnvelope deletes before sending.
    delete event.sdkProcessingMetadata;
    expect(JSON.stringify(event)).not.toContain(BODY_MARKER);
  });

  it("strips the body, cookies and credential headers from errors and transactions", async () => {
    await load();

    const { beforeSend, beforeSendTransaction } = initMock.mock.calls[0][0];
    function withCredentials() {
      return {
        request: {
          url: "https://platform.agpt.co/api/auth/sign-in/email",
          data: { password: BODY_MARKER },
          cookies: { a: "b" },
          headers: {
            Cookie: "a=b",
            authorization: "Basic x",
            "user-agent": "ua",
          },
        },
      };
    }

    const stripped = {
      url: "https://platform.agpt.co/api/auth/sign-in/email",
      headers: { "user-agent": "ua" },
    };
    expect(beforeSend(withCredentials(), {}).request).toEqual(stripped);
    expect(beforeSendTransaction(withCredentials()).request).toEqual(stripped);
  });

  it("tags events with the environment the deployment names for itself", async () => {
    vi.stubEnv("NEXT_PUBLIC_SENTRY_ENVIRONMENT", "pr-123");

    await load();

    expect(initMock.mock.calls[0][0].environment).toBe("pr-123");
  });

  it("tags events with the computed environment when none is named", async () => {
    vi.stubEnv("NEXT_PUBLIC_SENTRY_ENVIRONMENT", undefined);
    vi.stubEnv("NEXT_PUBLIC_APP_ENV", "local");
    vi.stubEnv("NEXT_PUBLIC_BEHAVE_AS", "LOCAL");
    vi.stubEnv("NEXT_PUBLIC_VERCEL_ENV", "");
    vi.stubEnv("VERCEL_ENV", "");

    await load();

    expect(initMock.mock.calls[0][0].environment).toBe(
      "app:local-behave:local",
    );
  });
});
