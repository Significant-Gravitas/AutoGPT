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

import {
  CLIENT_IGNORE_ERRORS,
  isFromBot,
  isNextRSCNavigationFallback,
} from "../sentry-filters";

// Drives a real Sentry client with the same console capture as
// instrumentation-client.ts, so the events are built exactly as in production.
const sent: Sentry.ErrorEvent[] = [];

beforeAll(() => {
  // Spy before init so the console patch wraps the spy.
  vi.spyOn(console, "error").mockImplementation(() => undefined);
  vi.spyOn(console, "warn").mockImplementation(() => undefined);
  Sentry.init({
    dsn: "https://examplePublicKey@o0.ingest.sentry.io/0",
    ignoreErrors: CLIENT_IGNORE_ERRORS,
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
    });
    await Sentry.flush(2000);

    expect(sent).toHaveLength(1);
    expect(sent[0].message).toBe(
      "Request failed on client: GET /api/library/agents 500",
    );
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

describe("Sentry filter for crawler noise", () => {
  it("drops Cookiebot's 'included twice' warning", async () => {
    console.warn(
      "WARNING: Cookiebot script is included twice - please remove one instance to avoid unexpected results.",
    );
    await Sentry.flush(2000);

    expect(sent).toHaveLength(0);
  });

  it("still reports our own warnings", async () => {
    console.warn("Cookie consent banner failed to load");
    await Sentry.flush(2000);

    expect(sent).toHaveLength(1);
  });

  it.each([
    "",
    "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
    "Mozilla/5.0 (compatible; GoogleOther)",
    "Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko; compatible; GPTBot/1.0; +https://openai.com/gptbot)",
    "Mozilla/5.0 (compatible; bingbot/2.0; +http://www.bing.com/bingbot.htm)",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Linux; Android 11; moto g power (2022)) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36 Chrome-Lighthouse",
  ])("treats %j as a bot", (userAgent) => {
    expect(isFromBot(userAgent)).toBe(true);
  });

  it.each([
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 18_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.6 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:143.0) Gecko/20100101 Firefox/143.0",
    "Mozilla/5.0 (Linux; Android 10; Cubot P40) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Mobile Safari/537.36",
  ])("treats %j as a person", (userAgent) => {
    expect(isFromBot(userAgent)).toBe(false);
  });
});
