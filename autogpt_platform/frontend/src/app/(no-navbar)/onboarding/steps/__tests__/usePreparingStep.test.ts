import { getGetBrainDumpRecommendedProvidersMockHandler200 } from "@/app/api/__generated__/endpoints/brain-dump/brain-dump.msw";
import { server } from "@/mocks/mock-server";
import { setIntroPath } from "@/services/onboarding/brain-dump-handoff";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook } from "@testing-library/react";
import { createElement, type ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { usePreparingStep } from "../usePreparingStep";

const START_DELAY_MS = 300;
const GENERIC_TOTAL_MS = 4_000;
const DUMP_TOTAL_MS = 10_000;
const RECOMMENDATIONS_MAX_WAIT_MS = 60_000;

const GENERIC_CHECKLIST = [
  "Personalizing your experience",
  "Connecting automation engines",
  "Building your space",
];
const DUMP_CHECKLIST = [
  "Reading your brain dump",
  "Briefing AutoPilot on your work",
  "Building your space",
  "Finding tools for your work",
];

function makeWrapper() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return function Wrapper({ children }: { children: ReactNode }) {
    return createElement(QueryClientProvider, { client }, children);
  };
}

function renderPreparing(args: {
  onComplete?: () => void;
  isBrainDumpEnabled?: boolean;
}) {
  return renderHook(
    () =>
      usePreparingStep({
        onComplete: args.onComplete ?? vi.fn(),
        isBrainDumpEnabled: args.isBrainDumpEnabled ?? false,
      }),
    { wrapper: makeWrapper() },
  );
}

function mockRecommendations(ready: boolean) {
  server.use(
    getGetBrainDumpRecommendedProvidersMockHandler200({
      ready,
      providers: [],
    }),
  );
}

async function advance(ms: number) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

beforeEach(() => {
  window.sessionStorage.clear();
  mockRecommendations(true);
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("usePreparingStep — checklist copy", () => {
  it("uses the honest dump copy only when the user actually dumped", () => {
    setIntroPath("A");

    const { result } = renderPreparing({ isBrainDumpEnabled: true });

    expect(result.current.checklist).toEqual(DUMP_CHECKLIST);
  });

  it("falls back to the generic copy on the skip path and with the flag off", () => {
    setIntroPath("B");
    const skipped = renderPreparing({ isBrainDumpEnabled: true });
    expect(skipped.result.current.checklist).toEqual(GENERIC_CHECKLIST);

    setIntroPath("A");
    const flagOff = renderPreparing({ isBrainDumpEnabled: false });
    expect(flagOff.result.current.checklist).toEqual(GENERIC_CHECKLIST);
  });

  it("does not consume the intro path — the copilot home still needs it", () => {
    setIntroPath("A");

    renderPreparing({ isBrainDumpEnabled: true });

    expect(window.sessionStorage.getItem("autogpt:onboarding-intro-path")).toBe(
      "A",
    );
  });
});

describe("usePreparingStep — progress", () => {
  it("holds everything still until the entrance delay elapses", async () => {
    const { result } = renderPreparing({});

    expect(result.current.started).toBe(false);
    expect(result.current.progress).toBe(0);
    expect(result.current.completedItems).toBe(0);

    await advance(START_DELAY_MS - 50);
    expect(result.current.started).toBe(false);
    expect(result.current.progress).toBe(0);

    await advance(50);
    expect(result.current.started).toBe(true);
  });

  it("fills the bar in proportion to elapsed time and ticks the checklist along", async () => {
    const { result } = renderPreparing({});

    await advance(START_DELAY_MS);
    // Halfway: past the first of three ~1.3s slices, so item two is lit.
    await advance(GENERIC_TOTAL_MS / 2);

    expect(result.current.progress).toBe(50);
    expect(result.current.completedItems).toBe(2);

    await advance(GENERIC_TOTAL_MS / 4);

    expect(result.current.progress).toBe(75);
    expect(result.current.completedItems).toBe(3);
  });

  it("caps progress at 100 and the checklist at its own length", async () => {
    const { result } = renderPreparing({});

    await advance(START_DELAY_MS);
    await advance(GENERIC_TOTAL_MS + 5_000);

    expect(result.current.progress).toBe(100);
    expect(result.current.completedItems).toBe(result.current.checklist.length);
  });
});

describe("usePreparingStep — completion", () => {
  it("calls onComplete exactly once, and not before the full duration", async () => {
    const onComplete = vi.fn();
    renderPreparing({ onComplete });

    await advance(START_DELAY_MS);
    await advance(GENERIC_TOTAL_MS - 100);
    expect(onComplete).not.toHaveBeenCalled();

    await advance(100);
    expect(onComplete).toHaveBeenCalledTimes(1);

    await advance(5_000);
    expect(onComplete).toHaveBeenCalledTimes(1);
  });

  it("calls the latest onComplete, not the one captured on the first render", async () => {
    const stale = vi.fn();
    const fresh = vi.fn();
    const { rerender } = renderHook(
      ({ onComplete }) =>
        usePreparingStep({ onComplete, isBrainDumpEnabled: false }),
      { initialProps: { onComplete: stale }, wrapper: makeWrapper() },
    );

    await advance(START_DELAY_MS + GENERIC_TOTAL_MS / 2);
    rerender({ onComplete: fresh });
    await advance(GENERIC_TOTAL_MS);

    expect(stale).not.toHaveBeenCalled();
    expect(fresh).toHaveBeenCalledTimes(1);
  });

  it("stops the ticker on unmount so a navigated-away wizard never completes", async () => {
    const onComplete = vi.fn();
    const { unmount } = renderPreparing({ onComplete });

    await advance(START_DELAY_MS + 1_000);
    unmount();
    await advance(GENERIC_TOTAL_MS);

    expect(onComplete).not.toHaveBeenCalled();
  });
});

describe("usePreparingStep — recommendation gate (dump path)", () => {
  it("completes after the full duration once recommendations are ready", async () => {
    setIntroPath("A");
    const onComplete = vi.fn();
    renderPreparing({ onComplete, isBrainDumpEnabled: true });

    await advance(START_DELAY_MS);
    await advance(DUMP_TOTAL_MS - 100);
    expect(onComplete).not.toHaveBeenCalled();

    await advance(200);
    expect(onComplete).toHaveBeenCalledTimes(1);
  });

  it("holds the last step and the bar while the recommendation job is still running", async () => {
    setIntroPath("A");
    mockRecommendations(false);
    const onComplete = vi.fn();
    const { result } = renderPreparing({
      onComplete,
      isBrainDumpEnabled: true,
    });

    await advance(START_DELAY_MS);
    await advance(DUMP_TOTAL_MS + 5_000);

    expect(onComplete).not.toHaveBeenCalled();
    expect(result.current.completedItems).toBe(DUMP_CHECKLIST.length - 1);
    expect(result.current.progress).toBeLessThanOrEqual(95);
  });

  it("releases the gate on the poll that reports the job finished", async () => {
    setIntroPath("A");
    mockRecommendations(false);
    const onComplete = vi.fn();
    const { result } = renderPreparing({
      onComplete,
      isBrainDumpEnabled: true,
    });

    await advance(START_DELAY_MS);
    await advance(DUMP_TOTAL_MS + 2_000);
    expect(onComplete).not.toHaveBeenCalled();

    mockRecommendations(true);
    // Next poll (2.5s cadence) picks up ready=true; the response settles
    // at the end of the first window, so a second window lets the ticker
    // observe it and complete the run.
    await advance(3_000);
    await advance(1_000);

    expect(onComplete).toHaveBeenCalledTimes(1);
    expect(result.current.progress).toBe(100);
    expect(result.current.completedItems).toBe(DUMP_CHECKLIST.length);
  });

  it("never strands the user: advances anyway after the max wait ceiling", async () => {
    setIntroPath("A");
    mockRecommendations(false);
    const onComplete = vi.fn();
    renderPreparing({ onComplete, isBrainDumpEnabled: true });

    await advance(START_DELAY_MS);
    await advance(RECOMMENDATIONS_MAX_WAIT_MS + 1_000);

    expect(onComplete).toHaveBeenCalledTimes(1);
  });

  it("does not gate the generic path on the recommendation job", async () => {
    setIntroPath("B");
    mockRecommendations(false);
    const onComplete = vi.fn();
    renderPreparing({ onComplete, isBrainDumpEnabled: true });

    await advance(START_DELAY_MS);
    await advance(GENERIC_TOTAL_MS + 200);

    expect(onComplete).toHaveBeenCalledTimes(1);
  });
});
