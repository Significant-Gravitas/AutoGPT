import { afterEach, describe, expect, it, vi } from "vitest";

const { initMock, reportingObserverIntegrationMock } = vi.hoisted(() => ({
  initMock: vi.fn(),
  reportingObserverIntegrationMock: vi.fn((options?: { types?: string[] }) => ({
    name: "ReportingObserver",
    options,
  })),
}));

vi.mock("@sentry/nextjs", () => {
  function stubIntegration(name: string) {
    return vi.fn(() => ({ name }));
  }

  return {
    init: initMock,
    reportingObserverIntegration: reportingObserverIntegrationMock,
    captureConsoleIntegration: stubIntegration("CaptureConsole"),
    extraErrorDataIntegration: stubIntegration("ExtraErrorData"),
    browserProfilingIntegration: stubIntegration("BrowserProfiling"),
    httpClientIntegration: stubIntegration("HttpClient"),
    featureFlagsIntegration: stubIntegration("FeatureFlags"),
    replayIntegration: stubIntegration("Replay"),
    replayCanvasIntegration: stubIntegration("ReplayCanvas"),
    captureRouterTransitionStart: vi.fn(),
  };
});

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("Sentry client instrumentation", () => {
  it("does not forward browser deprecation reports to Sentry", async () => {
    await import("../../instrumentation-client");

    expect(initMock).toHaveBeenCalledTimes(1);
    expect(reportingObserverIntegrationMock).toHaveBeenCalledTimes(1);

    const types = reportingObserverIntegrationMock.mock.calls[0][0]?.types;

    expect(types).toBeDefined();
    expect(types).not.toContain("deprecation");
    expect(types).toEqual(expect.arrayContaining(["crash", "intervention"]));
  });

  it("sends envelopes to the same-origin tunnel without cookies", async () => {
    vi.resetModules();
    initMock.mockClear();
    vi.stubEnv(
      "NEXT_PUBLIC_SENTRY_DSN",
      "https://key@o4505260022104064.ingest.us.sentry.io/4507946746380288",
    );
    await import("../../instrumentation-client");
    vi.unstubAllEnvs();

    const { tunnel, transportOptions } = initMock.mock.calls[0][0];

    expect(tunnel).toBe(
      "/relay/snentry?o=4505260022104064&p=4507946746380288&r=us",
    );
    expect(transportOptions?.fetchOptions?.credentials).toBe("omit");
  });

  it("starts with no tunnel when no DSN is configured", async () => {
    vi.resetModules();
    initMock.mockClear();
    vi.stubEnv("NEXT_PUBLIC_SENTRY_DSN", "");
    await import("../../instrumentation-client");
    vi.unstubAllEnvs();

    const { enabled, tunnel } = initMock.mock.calls[0][0];

    expect(enabled).toBe(false);
    expect(tunnel).toBeUndefined();
  });

  it("drops Next's handled RSC fetch fallback but keeps other fetch errors", async () => {
    await import("../../instrumentation-client");

    const { beforeSend } = initMock.mock.calls[0][0];
    const fetchError = {
      exception: {
        values: [{ type: "TypeError", value: "Failed to fetch" }],
      },
    };
    const rscFallback = {
      ...fetchError,
      logger: "console",
      extra: {
        arguments: [
          "Failed to fetch RSC payload for https://platform.agpt.co/login. Falling back to browser navigation.",
          { name: "TypeError", message: "Failed to fetch" },
        ],
      },
    };

    expect(beforeSend).toBeTypeOf("function");
    expect(beforeSend(rscFallback, {})).toBeNull();
    expect(beforeSend(fetchError, {})).toBe(fetchError);
  });

  it("drops a browser-injected script's iframe probe but keeps our own frame errors", async () => {
    await import("../../instrumentation-client");

    const { beforeSend } = initMock.mock.calls[0][0];
    const blocked = {
      exception: {
        values: [
          {
            type: "SecurityError",
            value:
              'Blocked a frame with origin "https://platform.agpt.co" from accessing a cross-origin frame.',
          },
        ],
      },
    };
    // As HeyTap Browser's injected script logs it on /onboarding (BUILDER-943).
    const injectedProbe = {
      ...blocked,
      logger: "console",
      extra: {
        arguments: ["无法访问 iframe 的内容:", { name: "SecurityError" }],
      },
    };
    const ourWarning = {
      ...blocked,
      logger: "console",
      extra: {
        arguments: [
          "Could not read the checkout frame",
          { name: "SecurityError" },
        ],
      },
    };

    expect(beforeSend(injectedProbe, {})).toBeNull();
    expect(beforeSend(blocked, {})).toBe(blocked);
    expect(beforeSend(ourWarning, {})).toBe(ourWarning);
  });

  it("tags events with the environment the deployment names for itself", async () => {
    vi.resetModules();
    initMock.mockClear();
    vi.stubEnv("NEXT_PUBLIC_SENTRY_ENVIRONMENT", "pr-123");
    await import("../../instrumentation-client");

    expect(initMock.mock.calls[0][0].environment).toBe("pr-123");
  });

  it("tags events with the computed environment when none is named", async () => {
    vi.resetModules();
    initMock.mockClear();
    vi.stubEnv("NEXT_PUBLIC_SENTRY_ENVIRONMENT", undefined);
    vi.stubEnv("NEXT_PUBLIC_APP_ENV", "local");
    vi.stubEnv("NEXT_PUBLIC_BEHAVE_AS", "LOCAL");
    vi.stubEnv("NEXT_PUBLIC_VERCEL_ENV", "");
    vi.stubEnv("VERCEL_ENV", "");
    await import("../../instrumentation-client");

    expect(initMock.mock.calls[0][0].environment).toBe(
      "app:local-behave:local",
    );
  });
});
