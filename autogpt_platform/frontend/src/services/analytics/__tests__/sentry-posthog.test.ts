import type { ErrorEvent } from "@sentry/nextjs";
import posthog from "posthog-js";
import {
  afterEach,
  beforeAll,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";
import {
  answerCookiebot,
  configureCookiebot,
  installCookiebot,
  removeCookiebot,
} from "@/tests/integrations/cookiebot";
import { createPostHogSentryProcessor } from "../sentry-posthog";

const dsn =
  "https://key@o4505260022104064.ingest.us.sentry.io/4507946746380288";
const processEvent = createPostHogSentryProcessor(dsn);

function sentryEvent(level: ErrorEvent["level"] = "error"): ErrorEvent {
  return {
    type: undefined,
    event_id: "0123456789abcdef0123456789abcdef",
    level,
    exception: {
      values: [
        {
          type: "TypeError",
          value: "Could not load workflow",
          stacktrace: {
            frames: [
              { filename: "https://app.example.com/app.js", lineno: 42 },
            ],
          },
        },
      ],
    },
    tags: { environment: "test" },
  };
}

beforeAll(() => {
  posthog.init("phc_test", {
    api_host: "https://posthog.example.com",
    ui_host: "https://eu.posthog.com",
    persistence: "memory",
    capture_pageview: false,
    capture_pageleave: false,
    autocapture: false,
    capture_exceptions: false,
    disable_session_recording: true,
    advanced_disable_flags: true,
    disable_external_dependency_loading: true,
    opt_out_capturing_by_default: true,
  });
});

beforeEach(() => {
  vi.stubEnv("NEXT_PUBLIC_BEHAVE_AS", "CLOUD");
  vi.stubEnv("NEXT_PUBLIC_POSTHOG_KEY", "phc_test");
  vi.stubEnv("NEXT_PUBLIC_POSTHOG_HOST", "https://eu.i.posthog.com");
  configureCookiebot();
  installCookiebot({ statistics: true });
  vi.spyOn(posthog, "capture").mockReturnValue(undefined);
  posthog.opt_in_capturing({ captureEventName: false });
  posthog.identify("user-123");
  vi.mocked(posthog.capture).mockClear();
});

afterEach(() => {
  posthog.opt_out_capturing();
  removeCookiebot();
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
});

describe("Sentry to PostHog correlation", () => {
  it.each(["error", "fatal"] as const)(
    "mirrors one %s with exact event, person and replay links",
    (level) => {
      vi.spyOn(posthog, "sessionRecordingStarted").mockReturnValue(true);
      vi.spyOn(posthog, "get_session_replay_url").mockReturnValue(
        "https://eu.posthog.com/replay/session?t=42",
      );
      const event = sentryEvent(level);

      expect(processEvent(event)).toBe(event);

      expect(posthog.capture).toHaveBeenCalledOnce();
      expect(posthog.capture).toHaveBeenCalledWith(
        "$exception",
        expect.objectContaining({
          $sentry_event_id: event.event_id,
          $sentry_url: `https://sentry.io/organizations/significant-gravitas/issues/?project=4507946746380288&query=${event.event_id}`,
          $exception_message: "Could not load workflow",
          $exception_level: level,
          $exception_list: [
            expect.objectContaining({
              stacktrace: expect.objectContaining({
                frames: [expect.objectContaining({ lineno: 42 })],
              }),
            }),
          ],
        }),
        expect.anything(),
      );
      expect(event.tags).toEqual({
        environment: "test",
        "PostHog Person URL":
          "https://eu.posthog.com/project/phc_test/person/user-123",
        "PostHog Recording URL": "https://eu.posthog.com/replay/session?t=42",
      });
      expect(posthog.get_session_replay_url).toHaveBeenCalledWith({
        withTimestamp: true,
      });
    },
  );

  it("links anonymous visitors without inventing a recording", () => {
    posthog.reset();
    posthog.opt_in_capturing({ captureEventName: false });
    vi.mocked(posthog.capture).mockClear();
    const event = sentryEvent();
    processEvent(event);
    expect(event.tags?.["PostHog Person URL"]).toContain(
      `/person/${posthog.get_distinct_id()}`,
    );
    expect(event.tags).not.toHaveProperty("PostHog Recording URL");
    expect(posthog.capture).toHaveBeenCalledOnce();
  });

  it.each(["warning", "info", "debug", "log"] as const)(
    "does not mirror %s events",
    (level) => {
      const event = sentryEvent(level);
      expect(processEvent(event)).toBe(event);
      expect(event.tags).toEqual({ environment: "test" });
      expect(posthog.capture).not.toHaveBeenCalled();
    },
  );

  it.each(["fetch", "xhr"])(
    "does not mirror a failed %s response the HttpClient integration reports",
    (type) => {
      const event = sentryEvent();
      event.exception!.values![0].mechanism = {
        type: `auto.http.client.${type}`,
        handled: false,
      };
      expect(processEvent(event)).toBe(event);
      expect(event.tags).toEqual({ environment: "test" });
      expect(posthog.capture).not.toHaveBeenCalled();
    },
  );

  it.each([undefined, { statistics: false }, { marketing: true }])(
    "requires analytics consent: %j",
    (answer) => {
      installCookiebot(answer);
      const event = sentryEvent();
      expect(processEvent(event)).toBe(event);
      expect(event.tags).toEqual({ environment: "test" });
      expect(posthog.capture).not.toHaveBeenCalled();
    },
  );

  it("stops immediately on consent withdrawal", () => {
    processEvent(sentryEvent());
    answerCookiebot({ statistics: false });
    const event = sentryEvent();
    processEvent(event);
    expect(posthog.capture).toHaveBeenCalledOnce();
    expect(event.tags).toEqual({ environment: "test" });
  });

  it("respects SDK opt-out even with banner consent", () => {
    posthog.opt_out_capturing();
    const event = sentryEvent();
    processEvent(event);
    expect(event.tags).toEqual({ environment: "test" });
    expect(posthog.capture).not.toHaveBeenCalled();
  });

  it("does not mirror when PostHog is disabled", () => {
    vi.stubEnv("NEXT_PUBLIC_POSTHOG_KEY", "");
    const event = sentryEvent();
    processEvent(event);
    expect(event.tags).toEqual({ environment: "test" });
    expect(posthog.capture).not.toHaveBeenCalled();
  });

  it.each([
    "https://key@o4505260022104064.ingest.us.sentry.io/123",
    "https://key@o4505260022104064.ingest.sentry.io/123",
  ])("uses the configured preview project from %s", (previewDSN) => {
    createPostHogSentryProcessor(previewDSN)(sentryEvent());
    expect(posthog.capture).toHaveBeenCalledWith(
      "$exception",
      expect.objectContaining({
        $sentry_url: expect.stringContaining("project=123&query="),
      }),
      expect.anything(),
    );
  });

  it.each([
    undefined,
    "invalid",
    "https://key@o123.ingest.sentry.io/7",
    "https://key@sentry.example.com/7",
  ])("does not invent an AutoGPT Sentry link for %s", (otherDSN) => {
    createPostHogSentryProcessor(otherDSN)(sentryEvent());
    expect(posthog.capture).toHaveBeenCalledOnce();
    expect(vi.mocked(posthog.capture).mock.calls[0][1]).not.toHaveProperty(
      "$sentry_url",
    );
  });

  it("keeps Sentry reporting when PostHog fails", () => {
    vi.mocked(posthog.capture).mockImplementation(() => {
      throw new Error("PostHog unavailable");
    });
    const event = sentryEvent();
    expect(processEvent(event)).toBe(event);
    expect(posthog.capture).toHaveBeenCalledOnce();
  });
});
