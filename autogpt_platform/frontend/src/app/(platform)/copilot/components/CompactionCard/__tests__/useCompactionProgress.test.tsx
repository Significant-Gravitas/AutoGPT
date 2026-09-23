import { act, render, renderHook, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { formatElapsed } from "../../JobStatsBar/formatElapsed";
import { CompactionCard, SHOW_TIME_AFTER_SECONDS } from "../CompactionCard";
import { PARKED_POLL_MS, PHASE_CURVE, type CompactionPhase } from "../helpers";
import { useCompactionProgress } from "../useCompactionProgress";

let frameCallbacks: Map<number, FrameRequestCallback>;
let nextFrameId: number;
let cancelledFrames: number[];
let now: number;

beforeEach(() => {
  frameCallbacks = new Map();
  nextFrameId = 1;
  cancelledFrames = [];
  now = 0;
  vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => {
    const id = nextFrameId++;
    frameCallbacks.set(id, cb);
    return id;
  });
  vi.stubGlobal("cancelAnimationFrame", (id: number) => {
    cancelledFrames.push(id);
    frameCallbacks.delete(id);
  });
  vi.spyOn(performance, "now").mockImplementation(() => now);
  // Only the timer functions — faking requestAnimationFrame here would
  // clobber the stub above and swallow every frame.
  vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

// A parked curve leaves no frame pending — it waits on the slow timer. This
// releases that timer so the frame loop picks back up.
function runParkedPoll() {
  act(() => {
    vi.advanceTimersByTime(PARKED_POLL_MS);
  });
}

function runFrame(advanceMs: number) {
  now += advanceMs;
  const pending = Array.from(frameCallbacks.values());
  frameCallbacks.clear();
  act(() => {
    for (const cb of pending) cb(now);
  });
}

function renderProgress(phase: CompactionPhase = "summarizing") {
  return renderHook(
    (props: { phase: CompactionPhase }) =>
      useCompactionProgress(props.phase, 128_000),
    { initialProps: { phase } },
  );
}

describe("useCompactionProgress", () => {
  it("never decreases", () => {
    const { result } = renderProgress();
    let prev = result.current.progress;
    for (let i = 0; i < 50; i++) {
      runFrame(500);
      expect(result.current.progress).toBeGreaterThanOrEqual(prev);
      prev = result.current.progress;
    }
  });

  it("carries progress over into a new phase instead of resetting", () => {
    const { result, rerender } = renderProgress();
    for (let i = 0; i < 20; i++) runFrame(1_000);
    const beforeSwitch = result.current.progress;
    expect(beforeSwitch).toBeGreaterThan(0.1);

    rerender({ phase: "rebuilding" });
    runFrame(16);
    expect(result.current.progress).toBeGreaterThanOrEqual(beforeSwitch);
    runFrame(5_000);
    expect(result.current.progress).toBeGreaterThan(beforeSwitch);
  });

  it("only commits state when the visible percent changes", () => {
    let renders = 0;
    renderHook(() => {
      renders++;
      return useCompactionProgress("summarizing", 128_000);
    });
    const before = renders;
    // 30 one-millisecond frames move the curve far less than one percent
    // and stay inside the same second — no commit, no re-render.
    for (let i = 0; i < 30; i++) runFrame(1);
    expect(renders).toBe(before);
  });

  it("parks the loop once pinned at the phase ceiling and never crosses it", () => {
    const { result } = renderProgress();
    for (let i = 0; i < 60 && frameCallbacks.size > 0; i++) {
      runFrame(60_000);
    }
    expect(frameCallbacks.size).toBe(0);
    expect(result.current.progress).toBeLessThanOrEqual(
      PHASE_CURVE.summarizing.cap,
    );
    // Parked means parked: time passing schedules nothing new.
    runFrame(60_000);
    expect(frameCallbacks.size).toBe(0);
  });

  it("wakes a parked loop when the phase raises the ceiling", () => {
    const { result, rerender } = renderProgress();
    for (let i = 0; i < 60 && frameCallbacks.size > 0; i++) {
      runFrame(60_000);
    }
    expect(frameCallbacks.size).toBe(0);
    const parkedAt = result.current.progress;

    rerender({ phase: "rebuilding" });
    runParkedPoll();
    expect(frameCallbacks.size).toBe(1);
    runFrame(16);
    runFrame(10_000);
    expect(result.current.progress).toBeGreaterThan(parkedAt);
    expect(result.current.progress).toBeLessThanOrEqual(
      PHASE_CURVE.rebuilding.cap,
    );
  });

  it("keeps counting while the curve is parked at its ceiling", () => {
    // The park deliberately idles on a slow timer rather than dropping the
    // loop entirely — a rebuild that stalls for two minutes must still show
    // two minutes, not the age it had when the bar stopped moving.
    const { result } = renderProgress();
    for (let i = 0; i < 60 && frameCallbacks.size > 0; i++) runFrame(60_000);
    expect(frameCallbacks.size).toBe(0);
    const parkedProgress = result.current.progress;
    const parkedSeconds = result.current.elapsedSeconds;

    runParkedPoll();
    runFrame(30_000);

    expect(result.current.elapsedSeconds).toBe(parkedSeconds + 30);
    expect(result.current.progress).toBe(parkedProgress);
  });

  it("cancels the pending frame on unmount", () => {
    const { unmount } = renderProgress();
    runFrame(500);
    expect(frameCallbacks.size).toBe(1);
    unmount();
    expect(frameCallbacks.size).toBe(0);
    expect(cancelledFrames.length).toBeGreaterThan(0);
  });
});

// The card's elapsed readout rides the same frame clock, so it is exercised
// through the same harness rather than a second set of stubs.
describe("CompactionCard elapsed readout", () => {
  function renderCard() {
    return render(
      <CompactionCard
        phase="summarizing"
        stats={{ tokensBefore: 128_000 }}
        isSettled={false}
      />,
    );
  }

  it("stays quiet until the wait is worth mentioning", () => {
    renderCard();
    runFrame((SHOW_TIME_AFTER_SECONDS - 1) * 1_000);
    expect(
      screen.queryByText(formatElapsed(SHOW_TIME_AFTER_SECONDS - 1)),
    ).toBeNull();
  });

  it("shows the timer once past the threshold and keeps it ticking", () => {
    renderCard();
    runFrame(SHOW_TIME_AFTER_SECONDS * 1_000);
    expect(
      screen.getByText(formatElapsed(SHOW_TIME_AFTER_SECONDS)),
    ).toBeDefined();

    runFrame(45_000);
    expect(
      screen.getByText(formatElapsed(SHOW_TIME_AFTER_SECONDS + 45)),
    ).toBeDefined();
  });
});
