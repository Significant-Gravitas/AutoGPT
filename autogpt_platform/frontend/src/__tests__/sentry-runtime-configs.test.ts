import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { initMock } = vi.hoisted(() => ({ initMock: vi.fn() }));

const BODY_MARKER = "marker-4f1c";
const URL_MARKER = "marker-u-7d3b";

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

  it("masks the credentials in the URLs of errors, transactions and spans", async () => {
    await load();

    const { beforeSend, beforeSendTransaction, beforeSendSpan } =
      initMock.mock.calls[0][0];
    function onATokenPage() {
      return {
        type: "transaction",
        transaction: "GET /reset-password",
        request: {
          url: `https://platform.agpt.co/reset-password?token=${URL_MARKER}`,
          query_string: `token=${URL_MARKER}`,
          headers: {
            referer: `https://platform.agpt.co/link/${URL_MARKER}`,
            "user-agent": "ua",
          },
        },
        contexts: {
          trace: {
            trace_id: "a".repeat(32),
            span_id: "b".repeat(16),
            data: {
              "http.target": `/reset-password?token=${URL_MARKER}`,
              "http.route": "/reset-password",
            },
          },
        },
        spans: [
          {
            trace_id: "a".repeat(32),
            span_id: "c".repeat(16),
            start_timestamp: 1,
            description: `GET https://backend.agpt.co/api/platform-linking/tokens/${URL_MARKER}/info`,
            data: {
              "http.url": `https://backend.agpt.co/api/platform-linking/tokens/${URL_MARKER}/info`,
              "server.address": "backend.agpt.co",
            },
          },
        ],
      };
    }

    for (const event of [
      beforeSend(onATokenPage(), {}),
      beforeSendTransaction(onATokenPage()),
    ]) {
      expect(event.transaction).toBe("GET /reset-password");
      expect(event.request).toEqual({
        url: "https://platform.agpt.co/reset-password?token=[REDACTED]",
        query_string: "[REDACTED]",
        headers: {
          referer: "https://platform.agpt.co/link/[REDACTED]",
          "user-agent": "ua",
        },
      });
      expect(event.contexts.trace.data).toEqual({
        "http.target": "/reset-password?token=[REDACTED]",
        "http.route": "/reset-password",
      });
      expect(event.spans[0].description).toBe(
        "GET https://backend.agpt.co/api/platform-linking/tokens/[REDACTED]/info",
      );
      expect(event.spans[0].data).toEqual({
        "http.url":
          "https://backend.agpt.co/api/platform-linking/tokens/[REDACTED]/info",
        "server.address": "backend.agpt.co",
      });
    }

    const span = beforeSendSpan({
      trace_id: "a".repeat(32),
      span_id: "c".repeat(16),
      start_timestamp: 1,
      description: `GET /link/${URL_MARKER}`,
      data: {
        "http.target": `/link/${URL_MARKER}`,
        "http.route": "/link/[token]",
      },
    });
    expect(span.description).toBe("GET /link/[REDACTED]");
    expect(span.data).toEqual({
      "http.target": "/link/[REDACTED]",
      "http.route": "/link/[token]",
    });
  });

  it("masks the page a request's headers name, as headers and as span attributes", async () => {
    await load();

    const { beforeSend, beforeSendTransaction, beforeSendSpan } =
      initMock.mock.calls[0][0];
    const routerState = encodeURIComponent(
      JSON.stringify(["", { children: ["link", ["token", URL_MARKER, "d"]] }]),
    );
    function duringANavigation() {
      return {
        type: "transaction",
        transaction: "GET /library",
        request: {
          url: "https://platform.agpt.co/library",
          headers: {
            "next-url": `/link/${URL_MARKER}`,
            "Next-Router-State-Tree": routerState,
            "user-agent": "ua",
          },
        },
        contexts: {
          trace: {
            trace_id: "a".repeat(32),
            span_id: "b".repeat(16),
            data: {
              "http.request.header.referer": `https://platform.agpt.co/reset-password?token=${URL_MARKER}`,
              "http.request.header.next_url": `/link/${URL_MARKER}`,
              "http.request.header.next_router_state_tree": routerState,
              "http.request.header.user_agent": "ua",
              "http.route": "/library",
            },
          },
        },
      };
    }

    for (const event of [
      beforeSend(duringANavigation(), {}),
      beforeSendTransaction(duringANavigation()),
    ]) {
      expect(event.request.headers).toEqual({
        "next-url": "/link/[REDACTED]",
        "Next-Router-State-Tree": "[REDACTED]",
        "user-agent": "ua",
      });
      expect(event.contexts.trace.data).toEqual({
        "http.request.header.referer":
          "https://platform.agpt.co/reset-password?token=[REDACTED]",
        "http.request.header.next_url": "/link/[REDACTED]",
        "http.request.header.next_router_state_tree": "[REDACTED]",
        "http.request.header.user_agent": "ua",
        "http.route": "/library",
      });
    }

    const span = beforeSendSpan({
      trace_id: "a".repeat(32),
      span_id: "c".repeat(16),
      start_timestamp: 1,
      data: { "http.request.header.referer": `/link/${URL_MARKER}` },
    });
    expect(span.data).toEqual({
      "http.request.header.referer": "/link/[REDACTED]",
    });
  });

  it("masks a span's query and fragment whether or not they keep their ? or #", async () => {
    await load();

    const { beforeSendSpan } = initMock.mock.calls[0][0];
    const span = beforeSendSpan({
      trace_id: "a".repeat(32),
      span_id: "c".repeat(16),
      start_timestamp: 1,
      data: {
        "url.query": `token=${URL_MARKER}&page=2`,
        "url.fragment": `access_token=${URL_MARKER}`,
        "http.query": `?token=${URL_MARKER}`,
        "http.fragment": "#section",
      },
    });

    expect(span.data).toEqual({
      "url.query": "token=[REDACTED]&page=[REDACTED]",
      "url.fragment": "access_token=[REDACTED]",
      "http.query": "?token=[REDACTED]",
      "http.fragment": "#section",
    });
  });

  it("sends no user email or IP address", async () => {
    await load();

    const { sendDefaultPii, beforeSend, beforeSendTransaction } =
      initMock.mock.calls[0][0];
    function withPersonalData() {
      return {
        user: {
          id: "user-1",
          email: "ada@example.com",
          ip_address: "203.0.113.7",
          name: "Ada Lovelace",
          segment: "pro",
        },
        request: {
          url: "https://platform.agpt.co/library",
          headers: {
            "user-agent": "ua",
            "X-Forwarded-For": "203.0.113.7",
            "x-real-ip": "203.0.113.7",
            "x-vercel-forwarded-for": "203.0.113.7",
            "x-vercel-ip-city": "Warsaw",
            "cf-connecting-ip": "203.0.113.7",
            forwarded: "for=203.0.113.7",
          },
        },
      };
    }

    expect(sendDefaultPii).toBe(false);
    for (const event of [
      beforeSend(withPersonalData(), {}),
      beforeSendTransaction(withPersonalData()),
    ]) {
      expect(event.user).toEqual({ id: "user-1" });
      expect(event.request.headers).toEqual({ "user-agent": "ua" });
    }
  });

  it("sends errors whose trace context carries no IP address or email", async () => {
    await load();

    const { beforeSend } = initMock.mock.calls[0][0];
    const event = beforeSend(
      {
        exception: { values: [{ type: "Error", value: "boom" }] },
        contexts: {
          trace: {
            trace_id: "a".repeat(32),
            span_id: "b".repeat(16),
            data: {
              "http.client_ip": "203.0.113.7",
              "client.address": "203.0.113.7",
              "http.request.header.x_forwarded_for": "203.0.113.7",
              "user.email": "ada@example.com",
              "user.id": "user-1",
              "http.route": "/library",
            },
          },
        },
      },
      {},
    );

    expect(event.contexts.trace.data).toEqual({
      "user.id": "user-1",
      "http.route": "/library",
    });
  });

  it("sends transactions whose spans carry no IP address, email or name", async () => {
    await load();

    const { beforeSendTransaction } = initMock.mock.calls[0][0];
    const event = beforeSendTransaction({
      type: "transaction",
      transaction: "GET /library",
      contexts: {
        trace: {
          trace_id: "a".repeat(32),
          span_id: "b".repeat(16),
          data: {
            "http.client_ip": "203.0.113.7",
            "net.peer.ip": "203.0.113.7",
            "net.host.ip": "10.0.0.2",
            "client.address": "203.0.113.7",
            "network.peer.address": "203.0.113.7",
            "http.request.header.x_forwarded_for": "203.0.113.7, 10.0.0.1",
            "http.request.header.x-real-ip": "203.0.113.7",
            "http.request.header.x_vercel_ip_city": "Warsaw",
            "http.route": "/library",
            "server.address": "platform.agpt.co",
            "user.id": "user-1",
          },
        },
      },
      spans: [
        {
          trace_id: "a".repeat(32),
          span_id: "c".repeat(16),
          start_timestamp: 1,
          data: {
            "net.sock.peer.addr": "203.0.113.7",
            "net.peer.ip": "203.0.113.7",
            "user.email": "ada@example.com",
            "user.name": "ada",
            "user.id": "user-1",
            "http.route": "/api/library",
            "server.address": "backend.agpt.co",
          },
        },
      ],
    });

    expect(event.contexts.trace.data).toEqual({
      "http.route": "/library",
      "server.address": "platform.agpt.co",
      "user.id": "user-1",
    });
    expect(event.spans[0].data).toEqual({
      "user.id": "user-1",
      "http.route": "/api/library",
      "server.address": "backend.agpt.co",
    });
  });

  it("sends standalone spans with no IP address, email or name", async () => {
    await load();

    const { beforeSendSpan } = initMock.mock.calls[0][0];
    const span = beforeSendSpan({
      trace_id: "a".repeat(32),
      span_id: "c".repeat(16),
      start_timestamp: 1,
      description: "GET /library",
      data: {
        user: "ada@example.com",
        "user.email": "ada@example.com",
        "user.id": "user-1",
        "http.client_ip": "203.0.113.7",
        "client.address": "203.0.113.7",
        "net.peer.ip": "203.0.113.7",
        "http.route": "/library",
        "server.address": "platform.agpt.co",
      },
    });

    expect(span.description).toBe("GET /library");
    expect(span.data).toEqual({
      "user.id": "user-1",
      "http.route": "/library",
      "server.address": "platform.agpt.co",
    });
  });

  it("sends logs with the user id only, and no email, name or IP address", async () => {
    await load();

    const { enableLogs, beforeSendLog } = initMock.mock.calls[0][0];
    const log = beforeSendLog({
      level: "info",
      message: "rendered the library",
      attributes: {
        "user.id": "user-1",
        "user.email": "ada@example.com",
        "user.name": "ada",
        "user.ip_address": "203.0.113.7",
        "client.address": "203.0.113.7",
        "http.client_ip": "203.0.113.7",
        ip: "203.0.113.7",
        "sentry.environment": "pr-123",
      },
    });

    expect(enableLogs).toBe(true);
    expect(log).toEqual({
      level: "info",
      message: "rendered the library",
      attributes: { "user.id": "user-1", "sentry.environment": "pr-123" },
    });
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
