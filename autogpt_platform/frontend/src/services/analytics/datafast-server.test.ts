import * as Sentry from "@sentry/nextjs";
import { cookies, headers } from "next/headers";
import { after } from "next/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  resetConfigErrorReportingForTests,
  scheduleAccountCreatedGoal,
  wasAccountCreated,
} from "./datafast-server";

vi.mock("next/headers", () => ({ cookies: vi.fn(), headers: vi.fn() }));
vi.mock("next/server", () => ({ after: vi.fn() }));
vi.mock("@sentry/nextjs", () => ({ captureException: vi.fn() }));

const VISITOR_ID = "a3ab2331-989f-4cfa-91c6-2461c9e3c6bd";
const STATISTICS_GRANTED =
  "{stamp:%27abc==%27%2Cnecessary:true%2Cpreferences:false%2Cstatistics:true%2Cmarketing:false%2Cmethod:%27explicit%27%2Cver:1%2Cutc:1724770548958%2Cregion:%27de%27}";
const STATISTICS_DENIED = STATISTICS_GRANTED.replace(
  "statistics:true",
  "statistics:false",
);

// Mirrors a request: cookies() keeps the last of two same-named cookies,
// while the raw Cookie header carries every copy.
function mockRequestCookies(entries: Array<[name: string, value: string]>) {
  vi.mocked(cookies).mockResolvedValue({
    get: vi.fn((name: string) => {
      const entry = entries.findLast(([key]) => key === name);
      return entry ? { value: entry[1] } : undefined;
    }),
  } as never);
  vi.mocked(headers).mockResolvedValue(
    new Headers(
      entries.length
        ? {
            cookie: entries.map(([key, value]) => `${key}=${value}`).join("; "),
          }
        : {},
    ) as never,
  );
}

describe("DataFast server-side account creation tracking", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetConfigErrorReportingForTests();
    vi.stubEnv("DATAFAST_API_KEY", "df_test");
    vi.stubEnv("NEXT_PUBLIC_BEHAVE_AS", "LOCAL");
    vi.stubEnv("NEXT_PUBLIC_COOKIEBOT_CBID", "test-cbid");
    mockRequestCookies([
      ["CookieConsent", STATISTICS_GRANTED],
      ["datafast_visitor_id", VISITOR_ID],
    ]);
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response(null, { status: 200 })),
    );
  });

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("sends the signup goal after the response completes", async () => {
    let callback: (() => Promise<void>) | undefined;
    vi.mocked(after).mockImplementation((next) => {
      callback = next as () => Promise<void>;
    });

    await scheduleAccountCreatedGoal("email");
    expect(after).toHaveBeenCalledOnce();

    await callback?.();

    expect(fetch).toHaveBeenCalledWith(
      "https://datafa.st/api/v1/goals",
      expect.objectContaining({
        method: "POST",
        headers: {
          Authorization: "Bearer df_test",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          datafast_visitor_id: VISITOR_ID,
          name: "signup",
          metadata: { method: "email" },
        }),
      }),
    );
  });

  it("does not schedule tracking without analytics consent", async () => {
    mockRequestCookies([]);

    await scheduleAccountCreatedGoal("google");

    expect(after).not.toHaveBeenCalled();
    expect(fetch).not.toHaveBeenCalled();
  });

  it("does not schedule tracking when the visitor declined statistics", async () => {
    mockRequestCookies([
      ["CookieConsent", STATISTICS_DENIED],
      ["datafast_visitor_id", VISITOR_ID],
    ]);

    await scheduleAccountCreatedGoal("email");

    expect(after).not.toHaveBeenCalled();
  });

  it.each([
    ["granting copy first", [STATISTICS_GRANTED, STATISTICS_DENIED]],
    ["granting copy last", [STATISTICS_DENIED, STATISTICS_GRANTED]],
  ])(
    "does not schedule tracking when duplicate consent cookies disagree (%s)",
    async (_, answers) => {
      mockRequestCookies([
        ...answers.map((value): [string, string] => ["CookieConsent", value]),
        ["datafast_visitor_id", VISITOR_ID],
      ]);

      await scheduleAccountCreatedGoal("email");

      expect(after).not.toHaveBeenCalled();
    },
  );

  it("does not schedule tracking without a consent banner, whatever the cookie says", async () => {
    vi.stubEnv("NEXT_PUBLIC_COOKIEBOT_CBID", "");

    await scheduleAccountCreatedGoal("email");

    expect(after).not.toHaveBeenCalled();
  });

  it("does not schedule tracking without a valid visitor ID", async () => {
    mockRequestCookies([["CookieConsent", STATISTICS_GRANTED]]);

    await scheduleAccountCreatedGoal("google");

    expect(after).not.toHaveBeenCalled();
  });

  it("does not report a missing website API key outside cloud", async () => {
    vi.stubEnv("DATAFAST_API_KEY", "");

    await scheduleAccountCreatedGoal("email");

    expect(after).not.toHaveBeenCalled();
    expect(Sentry.captureException).not.toHaveBeenCalled();
  });

  it("reports a missing website API key in cloud once per server instance", async () => {
    vi.stubEnv("DATAFAST_API_KEY", "");
    vi.stubEnv("NEXT_PUBLIC_BEHAVE_AS", "CLOUD");

    await scheduleAccountCreatedGoal("email");
    await scheduleAccountCreatedGoal("google");
    await scheduleAccountCreatedGoal("email");

    expect(after).not.toHaveBeenCalled();
    expect(Sentry.captureException).toHaveBeenCalledOnce();
  });

  it("reports a malformed configured website API key", async () => {
    vi.stubEnv("DATAFAST_API_KEY", "invalid-key");

    await scheduleAccountCreatedGoal("email");

    expect(after).not.toHaveBeenCalled();
    expect(Sentry.captureException).toHaveBeenCalledOnce();
  });

  it("isolates request-context failures from account creation", async () => {
    vi.mocked(headers).mockRejectedValue(new Error("request context closed"));
    vi.mocked(cookies).mockRejectedValue(new Error("request context closed"));

    await expect(scheduleAccountCreatedGoal("email")).resolves.toBeUndefined();

    expect(Sentry.captureException).toHaveBeenCalledOnce();
    expect(after).not.toHaveBeenCalled();
  });

  it("reports DataFast failures without rejecting the post-response task", async () => {
    let callback: (() => Promise<void>) | undefined;
    vi.mocked(after).mockImplementation((next) => {
      callback = next as () => Promise<void>;
    });
    vi.mocked(fetch).mockResolvedValue(new Response(null, { status: 503 }));

    await scheduleAccountCreatedGoal("google");
    await expect(callback?.()).resolves.toBeUndefined();

    expect(Sentry.captureException).toHaveBeenCalledOnce();
  });
});

describe("wasAccountCreated", () => {
  it("only accepts the explicit backend creation header", () => {
    expect(
      wasAccountCreated({
        status: 200,
        headers: new Headers({ "X-AutoGPT-User-Created": "true" }),
      }),
    ).toBe(true);
    expect(
      wasAccountCreated({
        status: 200,
        headers: new Headers({ "X-AutoGPT-User-Created": "false" }),
      }),
    ).toBe(false);
    expect(wasAccountCreated({ status: 200, headers: new Headers() })).toBe(
      false,
    );
  });

  it("throws a status-bearing error for a resolved backend failure", () => {
    expect.assertions(2);

    try {
      wasAccountCreated({ status: 500, headers: new Headers() });
    } catch (error) {
      expect(error).toMatchObject({ status: 500 });
      expect(error).toBeInstanceOf(Error);
    }
  });
});
