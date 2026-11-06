import * as Sentry from "@sentry/nextjs";
import {
  afterAll,
  afterEach,
  beforeAll,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";

import {
  logClientRequestFailure,
  resetClientRequestFailureLog,
  routeTemplate,
} from "../request-failure-log";

// A real Sentry client with the browser's console capture, so the events and
// their grouping are built exactly as they are in production.
const sent: Sentry.ErrorEvent[] = [];
const breadcrumbs: Sentry.Breadcrumb[] = [];
// Spied before init, so Sentry's console patch wraps the spy.
const consoleInfo = vi
  .spyOn(console, "info")
  .mockImplementation(() => undefined);

beforeAll(() => {
  vi.spyOn(console, "warn").mockImplementation(() => undefined);
  vi.spyOn(console, "error").mockImplementation(() => undefined);
  Sentry.init({
    dsn: "https://examplePublicKey@o0.ingest.sentry.io/0",
    transport: () => ({
      send: () => Promise.resolve({}),
      flush: () => Promise.resolve(true),
    }),
    integrations: [
      Sentry.captureConsoleIntegration({ levels: ["fatal", "error", "warn"] }),
    ],
    beforeSend(event) {
      sent.push(event);
      return null;
    },
    beforeBreadcrumb(breadcrumb) {
      if (breadcrumb.category === "http.client.failure") {
        breadcrumbs.push(breadcrumb);
      }
      return breadcrumb;
    },
  });
});

beforeEach(() => {
  sent.length = 0;
  breadcrumbs.length = 0;
  consoleInfo.mockClear();
});

afterEach(() => {
  resetClientRequestFailureLog();
  vi.useRealTimers();
});

afterAll(() => {
  vi.restoreAllMocks();
});

const SESSION_A = "9c1f6a52-3c0e-4b7e-9d0a-6f1e2b3c4d5e";
const SESSION_B = "0d4b2e61-8a7f-4c3d-b2e1-7a6f5e4d3c2b";

function fail(status: number, url: string, method = "GET") {
  logClientRequestFailure({
    status,
    method,
    url,
    errorMessage: "Backend sent no response within 30000ms",
  });
}

// Sentry's flush waits on real timers, so close the dedupe window on fake ones
// and hand back the real clock before flushing.
async function closeWindowAndFlush() {
  vi.advanceTimersByTime(10_000);
  vi.useRealTimers();
  await Sentry.flush(2000);
}

describe("client request failures in Sentry", () => {
  it("titles a server failure with its method, route and status", async () => {
    fail(504, `/api/internal/chat/sessions/${SESSION_A}/stream?resume=1`);
    await Sentry.flush(2000);

    expect(sent).toHaveLength(1);
    expect(sent[0].message).toBe(
      "Request failed on client: GET /api/internal/chat/sessions/{id}/stream 504",
    );
    expect(sent[0].level).toBe("error");
    expect(JSON.stringify(sent[0])).not.toContain("[object Object]");
    expect(sent[0].contexts?.request_failure).toMatchObject({
      route: "/api/internal/chat/sessions/{id}/stream",
      path: `/api/internal/chat/sessions/${SESSION_A}/stream`,
      status: 504,
    });
  });

  it("groups one route's server failures together, and apart from other routes", async () => {
    fail(504, `/api/internal/chat/sessions/${SESSION_A}`);
    fail(502, `/api/internal/chat/sessions/${SESSION_B}`);
    fail(504, "/api/internal/credits/trial/cancel", "POST");
    await Sentry.flush(2000);

    // Sentry may hand the events over in any order.
    const sessions = [
      "client-request-failure",
      "GET /api/internal/chat/sessions/{id}",
      "5xx",
    ];
    const trialCancel = [
      "client-request-failure",
      "POST /api/internal/credits/trial/cancel",
      "5xx",
    ];
    expect(sent).toHaveLength(3);
    expect(
      Object.fromEntries(
        sent.map((event) => [
          event.contexts?.request_failure?.path,
          event.fingerprint,
        ]),
      ),
    ).toEqual({
      [`/api/internal/chat/sessions/${SESSION_A}`]: sessions,
      [`/api/internal/chat/sessions/${SESSION_B}`]: sessions,
      "/api/internal/credits/trial/cancel": trialCancel,
    });
  });

  it("reports a request the app got wrong as a warning in its own 4xx issue", async () => {
    fail(400, `/api/internal/graphs/${SESSION_A}/execute/7`, "POST");
    fail(422, `/api/internal/graphs/${SESSION_B}/execute/9`, "POST");
    await Sentry.flush(2000);

    expect(sent.map((event) => event.level)).toEqual(["warning", "warning"]);
    expect(sent[0].fingerprint).toEqual([
      "client-request-failure",
      "POST /api/internal/graphs/{id}/execute/{id}",
      "4xx",
    ]);
    expect(sent[1].fingerprint).toEqual(sent[0].fingerprint);
  });

  it.each([
    [401, "signed out"],
    [402, "paywall"],
    [403, "permission the user lacks"],
    [404, "deleted resource"],
    [499, "user abandoned the request"],
  ])("leaves only a breadcrumb for an expected %i (%s)", async (status) => {
    fail(status, "/api/internal/experts/identities");
    await Sentry.flush(2000);

    expect(sent).toHaveLength(0);
    expect(breadcrumbs).toEqual([
      expect.objectContaining({
        level: "info",
        message: `Request failed on client: GET /api/internal/experts/identities ${status}`,
      }),
    ]);
  });

  it("sends one event for a burst of the same failure and counts the rest in the console", async () => {
    vi.useFakeTimers();
    for (let i = 0; i < 12; i++) fail(504, "/api/internal/chat/connections");
    await closeWindowAndFlush();

    expect(sent).toHaveLength(1);
    expect(consoleInfo).toHaveBeenLastCalledWith(
      "Request failed on client: GET /api/internal/chat/connections 504 ×12 in the last 10s",
    );
  });

  it("reports the same failure again once the window has passed", async () => {
    vi.useFakeTimers();
    fail(504, "/api/internal/chat/connections");
    await closeWindowAndFlush();
    fail(504, "/api/internal/chat/connections");
    await Sentry.flush(2000);

    expect(sent).toHaveLength(2);
  });

  it("does not ship a proxy's HTML error page", async () => {
    logClientRequestFailure({
      status: 502,
      method: "POST",
      url: "/api/internal/credits/trial/cancel",
      errorMessage: `<!DOCTYPE html>${"<div>cloudflare</div>".repeat(200)}`,
    });
    await Sentry.flush(2000);

    const detail = sent[0].contexts?.request_failure?.detail;
    expect(typeof detail === "string" && detail.length).toBe(300);
  });
});

describe("routeTemplate", () => {
  it.each([
    ["/api/internal/library/agents", "/api/internal/library/agents"],
    [
      `/api/internal/library/agents/${SESSION_A}`,
      "/api/internal/library/agents/{id}",
    ],
    ["/api/internal/executions/12345?page=2", "/api/internal/executions/{id}"],
    ["/api/internal/files/3f9a8b7c6d5e4f30", "/api/internal/files/{id}"],
    [
      "/api/internal/store/agents/V1StGXR8_Z5jdHi6B-myT",
      "/api/internal/store/agents/{id}",
    ],
    [
      "/api/internal/store/agents/alice/email-digest",
      "/api/internal/store/agents/alice/email-digest",
    ],
    ["/api/users/someone%40example.com", "/api/users/{id}"],
    [
      "/api/internal/integrations/codex/login",
      "/api/internal/integrations/codex/login",
    ],
  ])("%s → %s", (url, expected) => {
    expect(routeTemplate(url)).toBe(expected);
  });
});
