import { cleanup, renderHook } from "@testing-library/react";
import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  vi,
  type Mock,
} from "vitest";

import { useWakeResync } from "../useWakeResync";

type RefetchResult = { data?: unknown };

interface SetupOpts {
  initialSessionId?: string | null;
  refetchResult?: RefetchResult;
  refetchImpl?: () => Promise<RefetchResult>;
  initialMounted?: boolean;
}

function withVisibilityState(state: "visible" | "hidden") {
  Object.defineProperty(document, "visibilityState", {
    configurable: true,
    get: () => state,
  });
}

function fireVisibilityChange() {
  document.dispatchEvent(new Event("visibilitychange"));
}

function setup(opts: SetupOpts = {}) {
  const sessionIdRef = {
    current: "initialSessionId" in opts ? opts.initialSessionId! : "sess-1",
  };
  const isMountedRef = { current: opts.initialMounted ?? true };
  const refetchSession =
    (opts.refetchImpl as Mock<() => Promise<RefetchResult>>) ??
    vi.fn(async () => opts.refetchResult ?? { data: undefined });
  const resumeStream = vi.fn();
  const resumeStreamRef = { current: resumeStream };

  const hook = renderHook(() =>
    useWakeResync({
      sessionIdRef,
      isMountedRef,
      refetchSession,
      resumeStreamRef,
    }),
  );

  return { hook, sessionIdRef, isMountedRef, refetchSession, resumeStream };
}

describe("useWakeResync", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    withVisibilityState("visible");
  });

  afterEach(() => {
    cleanup();
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("does nothing when there is no session", async () => {
    const { refetchSession } = setup({ initialSessionId: null });

    // Hide → wait long enough → re-show.
    withVisibilityState("hidden");
    fireVisibilityChange();
    vi.advanceTimersByTime(60_000);
    withVisibilityState("visible");
    fireVisibilityChange();

    await vi.runAllTimersAsync();
    expect(refetchSession).not.toHaveBeenCalled();
  });

  it("skips re-sync when the page was hidden for less than the threshold", async () => {
    const { refetchSession } = setup();

    withVisibilityState("hidden");
    fireVisibilityChange();
    vi.advanceTimersByTime(5_000); // < 30s threshold
    withVisibilityState("visible");
    fireVisibilityChange();

    await vi.runAllTimersAsync();
    expect(refetchSession).not.toHaveBeenCalled();
  });

  it("refetches and does not resume when backend has no active stream", async () => {
    const refetchSession = vi.fn(async () => ({
      data: { status: 200, data: { active_stream: null } },
    }));
    const { resumeStream } = setup({ refetchImpl: refetchSession });

    withVisibilityState("hidden");
    fireVisibilityChange();
    vi.advanceTimersByTime(60_000); // > 30s
    withVisibilityState("visible");
    fireVisibilityChange();

    await vi.runAllTimersAsync();
    expect(refetchSession).toHaveBeenCalledTimes(1);
    expect(resumeStream).not.toHaveBeenCalled();
  });

  it("resumes the stream when the refetch reports an active backend stream", async () => {
    const refetchSession = vi.fn(async () => ({
      data: {
        status: 200,
        data: { active_stream: { turn_id: "t1", started_at: "2026-01-01" } },
      },
    }));
    const { resumeStream } = setup({ refetchImpl: refetchSession });

    withVisibilityState("hidden");
    fireVisibilityChange();
    vi.advanceTimersByTime(60_000);
    withVisibilityState("visible");
    fireVisibilityChange();

    await vi.runAllTimersAsync();
    expect(refetchSession).toHaveBeenCalledTimes(1);
    expect(resumeStream).toHaveBeenCalledTimes(1);
  });

  it("does not resume when the session changed during the refetch", async () => {
    let resolveRefetch: (v: RefetchResult) => void = () => {};
    const refetchSession = vi.fn(
      () => new Promise<RefetchResult>((r) => (resolveRefetch = r)),
    );
    const { resumeStream, sessionIdRef } = setup({
      refetchImpl: refetchSession,
    });

    withVisibilityState("hidden");
    fireVisibilityChange();
    vi.advanceTimersByTime(60_000);
    withVisibilityState("visible");
    fireVisibilityChange();

    // Simulate the user navigating to a different session before the
    // refetch resolves.
    sessionIdRef.current = "different-session";
    resolveRefetch({
      data: { status: 200, data: { active_stream: { turn_id: "t1" } } },
    });

    await vi.runAllTimersAsync();
    expect(resumeStream).not.toHaveBeenCalled();
  });

  it("does not resume when the host unmounted during the refetch", async () => {
    let resolveRefetch: (v: RefetchResult) => void = () => {};
    const refetchSession = vi.fn(
      () => new Promise<RefetchResult>((r) => (resolveRefetch = r)),
    );
    const { resumeStream, isMountedRef } = setup({
      refetchImpl: refetchSession,
    });

    withVisibilityState("hidden");
    fireVisibilityChange();
    vi.advanceTimersByTime(60_000);
    withVisibilityState("visible");
    fireVisibilityChange();

    isMountedRef.current = false;
    resolveRefetch({
      data: { status: 200, data: { active_stream: { turn_id: "t1" } } },
    });

    await vi.runAllTimersAsync();
    expect(resumeStream).not.toHaveBeenCalled();
  });

  it("swallows refetch errors and clears isSyncing in finally", async () => {
    const refetchSession = vi.fn(async () => {
      throw new Error("boom");
    });
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const { hook } = setup({ refetchImpl: refetchSession });

    withVisibilityState("hidden");
    fireVisibilityChange();
    vi.advanceTimersByTime(60_000);
    withVisibilityState("visible");
    fireVisibilityChange();

    await vi.runAllTimersAsync();
    expect(refetchSession).toHaveBeenCalledTimes(1);
    expect(warn).toHaveBeenCalled();
    expect(hook.result.current.isSyncing).toBe(false);
  });

  it("only updates the hidden-at timestamp when going to hidden, not visible", async () => {
    const refetchSession = vi.fn(async () => ({ data: undefined }));
    setup({ refetchImpl: refetchSession });

    // Hidden → visible after only 5s: skipped.
    withVisibilityState("hidden");
    fireVisibilityChange();
    vi.advanceTimersByTime(5_000);
    withVisibilityState("visible");
    fireVisibilityChange();
    await vi.runAllTimersAsync();
    expect(refetchSession).not.toHaveBeenCalled();

    // Hidden again → visible after 60s: triggers (NOT 5s + 60s = 65s
    // measured from the FIRST hidden — only the most recent hidden counts).
    withVisibilityState("hidden");
    fireVisibilityChange();
    vi.advanceTimersByTime(60_000);
    withVisibilityState("visible");
    fireVisibilityChange();
    await vi.runAllTimersAsync();
    expect(refetchSession).toHaveBeenCalledTimes(1);
  });
});
