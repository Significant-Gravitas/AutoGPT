import { setIntroPath } from "@/services/onboarding/brain-dump-handoff";
import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { usePreparingStep } from "../usePreparingStep";

const START_DELAY_MS = 300;
const TOTAL_MS = 10_000;

const GENERIC_CHECKLIST = [
  "Personalizing your experience",
  "Connecting automation engines",
  "Building your space",
];
const DUMP_CHECKLIST = [
  "Reading your brain dump",
  "Briefing AutoPilot on your work",
  "Building your space",
];

async function advance(ms: number) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

beforeEach(() => {
  window.sessionStorage.clear();
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("usePreparingStep — checklist copy", () => {
  it("uses the honest dump copy only when the user actually dumped", () => {
    setIntroPath("A");

    const { result } = renderHook(() =>
      usePreparingStep({ onComplete: vi.fn(), isBrainDumpEnabled: true }),
    );

    expect(result.current.checklist).toEqual(DUMP_CHECKLIST);
  });

  it("falls back to the generic copy on the skip path and with the flag off", () => {
    setIntroPath("B");
    const skipped = renderHook(() =>
      usePreparingStep({ onComplete: vi.fn(), isBrainDumpEnabled: true }),
    );
    expect(skipped.result.current.checklist).toEqual(GENERIC_CHECKLIST);

    setIntroPath("A");
    const flagOff = renderHook(() =>
      usePreparingStep({ onComplete: vi.fn(), isBrainDumpEnabled: false }),
    );
    expect(flagOff.result.current.checklist).toEqual(GENERIC_CHECKLIST);
  });

  it("does not consume the intro path — the copilot home still needs it", () => {
    setIntroPath("A");

    renderHook(() =>
      usePreparingStep({ onComplete: vi.fn(), isBrainDumpEnabled: true }),
    );

    expect(window.sessionStorage.getItem("autogpt:onboarding-intro-path")).toBe(
      "A",
    );
  });
});

describe("usePreparingStep — progress", () => {
  it("holds everything still until the entrance delay elapses", async () => {
    const { result } = renderHook(() =>
      usePreparingStep({ onComplete: vi.fn(), isBrainDumpEnabled: false }),
    );

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
    const { result } = renderHook(() =>
      usePreparingStep({ onComplete: vi.fn(), isBrainDumpEnabled: false }),
    );

    await advance(START_DELAY_MS);
    // Halfway: past the first of three ~3.3s slices, so item two is lit.
    await advance(TOTAL_MS / 2);

    expect(result.current.progress).toBe(50);
    expect(result.current.completedItems).toBe(2);

    await advance(TOTAL_MS / 4);

    expect(result.current.progress).toBe(75);
    expect(result.current.completedItems).toBe(3);
  });

  it("caps progress at 100 and the checklist at its own length", async () => {
    const { result } = renderHook(() =>
      usePreparingStep({ onComplete: vi.fn(), isBrainDumpEnabled: false }),
    );

    await advance(START_DELAY_MS);
    await advance(TOTAL_MS + 5_000);

    expect(result.current.progress).toBe(100);
    expect(result.current.completedItems).toBe(result.current.checklist.length);
  });
});

describe("usePreparingStep — completion", () => {
  it("calls onComplete exactly once, and not before the full duration", async () => {
    const onComplete = vi.fn();
    renderHook(() =>
      usePreparingStep({ onComplete, isBrainDumpEnabled: false }),
    );

    await advance(START_DELAY_MS);
    await advance(TOTAL_MS - 100);
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
      { initialProps: { onComplete: stale } },
    );

    await advance(START_DELAY_MS + TOTAL_MS / 2);
    rerender({ onComplete: fresh });
    await advance(TOTAL_MS);

    expect(stale).not.toHaveBeenCalled();
    expect(fresh).toHaveBeenCalledTimes(1);
  });

  it("stops the ticker on unmount so a navigated-away wizard never completes", async () => {
    const onComplete = vi.fn();
    const { unmount } = renderHook(() =>
      usePreparingStep({ onComplete, isBrainDumpEnabled: false }),
    );

    await advance(START_DELAY_MS + 1_000);
    unmount();
    await advance(TOTAL_MS);

    expect(onComplete).not.toHaveBeenCalled();
  });
});
