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
} from "@/app/api/mutators/request-failure-log";

import { isNextRSCNavigationFallback } from "../sentry-filters";

// Drives a real Sentry client with the same console capture as
// instrumentation-client.ts, so the events are built exactly as in production.
const sent: Sentry.ErrorEvent[] = [];

beforeAll(() => {
  // Spy before init so the console patch wraps the spy.
  vi.spyOn(console, "error").mockImplementation(() => undefined);
  vi.spyOn(console, "warn").mockImplementation(() => undefined);
  Sentry.init({
    dsn: "https://examplePublicKey@o0.ingest.sentry.io/0",
    // No network: a failed send is itself logged, and would be captured too.
    transport: () => ({
      send: () => Promise.resolve({}),
      flush: () => Promise.resolve(true),
    }),
    integrations: [
      Sentry.captureConsoleIntegration({ levels: ["fatal", "error", "warn"] }),
    ],
    beforeSend(event) {
      if (!isNextRSCNavigationFallback(event)) sent.push(event);
      return null;
    },
  });
});

beforeEach(() => {
  sent.length = 0;
});

afterEach(() => {
  resetClientRequestFailureLog();
});

afterAll(() => {
  vi.restoreAllMocks();
});

describe("Sentry filter for Next's RSC navigation fallback", () => {
  it("drops the handled 'Failed to fetch RSC payload' console error", async () => {
    // Verbatim from next/dist/client/components/router-reducer/fetch-server-response.
    console.error(
      "Failed to fetch RSC payload for https://platform.agpt.co/login. Falling back to browser navigation.",
      new TypeError("Failed to fetch"),
    );
    await Sentry.flush(2000);

    expect(sent).toHaveLength(0);
  });

  it("still reports a fetch failure captured from our own code", async () => {
    Sentry.captureException(new TypeError("Failed to fetch"));
    await Sentry.flush(2000);

    expect(sent).toHaveLength(1);
    expect(sent[0].exception?.values?.[0]).toMatchObject({
      type: "TypeError",
      value: "Failed to fetch",
    });
  });

  it("still reports our own console error about a failed fetch", async () => {
    console.error("Failed to fetch agents:", new TypeError("Failed to fetch"));
    await Sentry.flush(2000);

    expect(sent).toHaveLength(1);
    expect(sent[0].extra?.arguments).toEqual([
      "Failed to fetch agents:",
      expect.anything(),
    ]);
  });

  it("still reports an API request failure logged by the mutator", async () => {
    logClientRequestFailure({
      status: 500,
      method: "GET",
      url: "/api/library/agents",
      errorMessage: "Internal Server Error",
      responseData: null,
    });
    await Sentry.flush(2000);

    expect(sent).toHaveLength(1);
    expect(sent[0].message).toContain("Request failed on client");
  });

  it("still reports a message that only borrows the RSC wording", async () => {
    console.error(
      "Failed to fetch RSC payload for https://platform.agpt.co/login",
      new TypeError("Failed to fetch"),
    );
    await Sentry.flush(2000);

    expect(sent).toHaveLength(1);
  });
});
