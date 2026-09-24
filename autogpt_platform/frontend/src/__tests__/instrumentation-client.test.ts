import { describe, expect, it, vi } from "vitest";

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
});
