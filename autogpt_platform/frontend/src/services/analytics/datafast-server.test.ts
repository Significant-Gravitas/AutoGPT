import * as Sentry from "@sentry/nextjs";
import { cookies } from "next/headers";
import { after } from "next/server";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  scheduleAccountCreatedGoal,
  wasAccountCreated,
} from "./datafast-server";

vi.mock("next/headers", () => ({ cookies: vi.fn() }));
vi.mock("next/server", () => ({ after: vi.fn() }));
vi.mock("@sentry/nextjs", () => ({ captureException: vi.fn() }));

const VISITOR_ID = "a3ab2331-989f-4cfa-91c6-2461c9e3c6bd";

describe("DataFast server-side account creation tracking", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    process.env.DATAFAST_API_KEY = "df_test";
    vi.mocked(cookies).mockResolvedValue({
      get: vi.fn((name: string) => {
        if (name === "agpt_analytics_consent") return { value: "granted" };
        if (name === "datafast_visitor_id") return { value: VISITOR_ID };
        return undefined;
      }),
    } as never);
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response(null, { status: 200 })),
    );
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
    vi.mocked(cookies).mockResolvedValue({
      get: vi.fn(() => undefined),
    } as never);

    await scheduleAccountCreatedGoal("google");

    expect(after).not.toHaveBeenCalled();
    expect(fetch).not.toHaveBeenCalled();
  });

  it("does not schedule tracking without a valid visitor ID", async () => {
    vi.mocked(cookies).mockResolvedValue({
      get: vi.fn((name: string) =>
        name === "agpt_analytics_consent" ? { value: "granted" } : undefined,
      ),
    } as never);

    await scheduleAccountCreatedGoal("google");

    expect(after).not.toHaveBeenCalled();
  });

  it("does not schedule tracking without a configured website API key", async () => {
    delete process.env.DATAFAST_API_KEY;

    await scheduleAccountCreatedGoal("email");

    expect(after).not.toHaveBeenCalled();
    expect(Sentry.captureException).toHaveBeenCalledOnce();
  });

  it("isolates request-context failures from account creation", async () => {
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
      wasAccountCreated(new Headers({ "X-AutoGPT-User-Created": "true" })),
    ).toBe(true);
    expect(
      wasAccountCreated(new Headers({ "X-AutoGPT-User-Created": "false" })),
    ).toBe(false);
    expect(wasAccountCreated(new Headers())).toBe(false);
  });
});
