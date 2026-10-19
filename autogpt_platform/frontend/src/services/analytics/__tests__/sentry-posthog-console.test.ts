import {
  configureCookiebot,
  installCookiebot,
  removeCookiebot,
} from "@/tests/integrations/cookiebot";
import * as Sentry from "@sentry/nextjs";
import posthog from "posthog-js";
import {
  afterAll,
  afterEach,
  beforeAll,
  beforeEach,
  expect,
  it,
  vi,
} from "vitest";

vi.mock("@sentry/nextjs", async () => {
  const { createRequire } = await import("node:module");
  const nextjs = createRequire(import.meta.url).resolve(
    "@sentry/nextjs/package.json",
  );
  const react = createRequire(nextjs).resolve("@sentry/react/package.json");
  const browser = createRequire(react).resolve("@sentry/browser/package.json");
  const sdk = await import(
    browser.replace("package.json", "build/npm/esm/prod/index.js")
  );
  return { ...sdk, captureRouterTransitionStart: () => undefined };
});

const sent: Sentry.ErrorEvent[] = [];
const consoleErrors = vi.fn();
const rateLimitMessage =
  "[PostHog.js] This capture call is ignored due to client rate limiting.";

beforeAll(async () => {
  vi.spyOn(console, "error").mockImplementation(consoleErrors);
  vi.stubEnv("NEXT_PUBLIC_SENTRY_DSN", "https://public@o0.ingest.sentry.io/1");
  posthog.init("phc_test", {
    api_host: "https://posthog.example.com",
    persistence: "memory",
    capture_pageview: false,
    capture_pageleave: false,
    autocapture: false,
    capture_exceptions: false,
    disable_session_recording: true,
    advanced_disable_flags: true,
    disable_external_dependency_loading: true,
    opt_out_capturing_by_default: true,
    opt_out_useragent_filter: true,
    before_send: () => null,
  });
  await import("../../../../instrumentation-client");
  vi.spyOn(Sentry.getClient()!.getTransport()!, "send").mockImplementation(
    async ([, items]) => {
      for (const [header, payload] of items) {
        if (header.type === "event") sent.push(payload as Sentry.ErrorEvent);
      }
      return {};
    },
  );
});

beforeEach(({ task }) => {
  vi.useFakeTimers({ toFake: ["Date"] });
  vi.stubEnv("NEXT_PUBLIC_BEHAVE_AS", "CLOUD");
  vi.stubEnv("NEXT_PUBLIC_POSTHOG_KEY", "phc_test");
  vi.stubEnv("NEXT_PUBLIC_POSTHOG_HOST", "https://eu.i.posthog.com");
  configureCookiebot();
  installCookiebot({ statistics: true });
  posthog.reset();
  posthog.opt_in_capturing({ captureEventName: false });
  posthog.set_config({
    rate_limiting: { events_per_second: 100, events_burst_limit: 100 },
  });
  vi.spyOn(posthog, "capture");
  consoleErrors.mockClear();
  sent.length = 0;
  Sentry.getCurrentScope().setFingerprint([task.name]);
});

afterEach(() => {
  posthog.opt_out_capturing();
  removeCookiebot();
  vi.mocked(posthog.capture).mockRestore();
  vi.useRealTimers();
});

afterAll(async () => {
  await Sentry.close(2000);
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

it("does not feed an exhausted PostHog rate limiter's console error back into capture", async () => {
  posthog.set_config({
    rate_limiting: { events_per_second: 1, events_burst_limit: 1 },
  });
  posthog.capture("fill-bucket");
  expect(posthog.is_capturing()).toBe(true);
  vi.mocked(posthog.capture).mockClear();

  Sentry.captureException(
    new Error("Application failure during an error burst"),
  );
  await Sentry.flush(2000);

  expect(consoleErrors).toHaveBeenCalledExactlyOnceWith(
    "[PostHog.js]",
    "This capture call is ignored due to client rate limiting.",
  );
  expect(sent).toHaveLength(2);
  expect(sent.find((event) => event.logger !== "console")).toMatchObject({
    exception: {
      values: [{ value: "Application failure during an error burst" }],
    },
  });
  expect(sent.find((event) => event.logger === "console")).toMatchObject({
    message: rateLimitMessage,
    level: "error",
  });
  expect(
    vi
      .mocked(posthog.capture)
      .mock.calls.filter(([name]) => name === "$exception"),
  ).toHaveLength(1);
});

it("keeps the rate-limit diagnostic in Sentry without mirroring or adding PostHog tags", async () => {
  console.error(
    "[PostHog.js]",
    "This capture call is ignored due to client rate limiting.",
  );
  await Sentry.flush(2000);

  expect(sent).toHaveLength(1);
  expect(sent[0]).toMatchObject({
    logger: "console",
    message: rateLimitMessage,
  });
  expect(sent[0].tags ?? {}).not.toHaveProperty("PostHog Person URL");
  expect(posthog.capture).not.toHaveBeenCalled();
});

it.each([
  "Workflow failed",
  "This capture call is ignored due to client rate limiting.",
])("still mirrors an application console error: %s", async (message) => {
  console.error(message);
  await Sentry.flush(2000);

  expect(sent).toHaveLength(1);
  expect(posthog.capture).toHaveBeenCalledExactlyOnceWith(
    "$exception",
    expect.objectContaining({ $sentry_event_id: sent[0].event_id }),
    expect.anything(),
  );
});

it("still mirrors a directly captured application message matching the diagnostic", async () => {
  Sentry.captureMessage(rateLimitMessage, "error");
  await Sentry.flush(2000);

  expect(sent).toHaveLength(1);
  expect(posthog.capture).toHaveBeenCalledExactlyOnceWith(
    "$exception",
    expect.objectContaining({ $sentry_event_id: sent[0].event_id }),
    expect.anything(),
  );
});
