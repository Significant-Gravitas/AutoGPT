import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const savedMeta: Record<string, unknown>[] = [];

vi.mock("../recordingStore", () => ({
  savePart: vi.fn().mockResolvedValue(undefined),
  saveMeta: vi.fn(async (meta: Record<string, unknown>) => {
    savedMeta.push(meta);
  }),
  getMeta: vi.fn().mockResolvedValue(null),
  getParts: vi.fn().mockResolvedValue([]),
}));

vi.mock("../useUploadQueue", async () => {
  const actual =
    await vi.importActual<typeof import("../useUploadQueue")>(
      "../useUploadQueue",
    );
  return {
    buildPart: actual.buildPart,
    useUploadQueue: () => ({
      enqueue: vi.fn(),
      flush: vi.fn().mockResolvedValue(true),
      reset: vi.fn(),
      pendingCount: 0,
      isOffline: false,
    }),
  };
});

vi.mock("@/services/onboarding/brain-dump-analytics", () => ({
  trackBrainDump: vi.fn(),
}));

import { useBrainDumpRecorder } from "../useBrainDumpRecorder";

// Stopping is not instant: `onstop` fires after the encoder drains, and
// the hook then awaits the pending IndexedDB writes. This recorder makes
// that gap explicit so a test can observe what happens across it.
const STOP_GAP_MS = 4000;

class SlowStoppingMediaRecorder {
  static isTypeSupported() {
    return true;
  }
  state: "inactive" | "recording" = "inactive";
  ondataavailable: ((event: { data: Blob }) => void) | null = null;
  onstop: (() => void) | null = null;

  start() {
    this.state = "recording";
  }

  stop() {
    this.state = "inactive";
    // The clock keeps running while the encoder finishes.
    vi.advanceTimersByTime(STOP_GAP_MS);
    this.onstop?.();
  }
}

describe("useBrainDumpRecorder", () => {
  beforeEach(() => {
    savedMeta.length = 0;
    vi.useFakeTimers();
    vi.stubGlobal("MediaRecorder", SlowStoppingMediaRecorder);
    vi.stubGlobal("navigator", {
      ...navigator,
      mediaDevices: {
        getUserMedia: vi.fn().mockResolvedValue({ getTracks: () => [] }),
      },
    });
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  // The regression: the silence nudge keyed off `isSavedLocally`, which
  // flips as soon as the first chunk is persisted. MediaRecorder emits
  // one every timeslice whether or not anybody spoke, so by the nudge's
  // own 5s threshold the flag was always already true and the nudge
  // could never appear.
  it("does not report speech from a silent room", async () => {
    const { result } = renderHook(() => useBrainDumpRecorder());

    await act(async () => {
      await result.current.start();
    });
    await act(async () => {
      vi.advanceTimersByTime(10_000);
    });

    // Well past the 5s nudge threshold, and still silent — which is
    // exactly when the nudge should be showing.
    expect(result.current.elapsedSeconds).toBeGreaterThan(5);
    expect(result.current.hasSpoken).toBe(false);
  });

  // The regression: `stop()` reported `elapsedSeconds` straight off React
  // state, which is the value from the caller's own render. Every second
  // spent stopping was missing from it. The backend splits recordings
  // over 20 minutes on this number, so under-reporting it means a long
  // dump silently skips splitting and fails to transcribe.
  it("reports the duration including the time spent stopping", async () => {
    const { result } = renderHook(() => useBrainDumpRecorder());

    await act(async () => {
      await result.current.start();
    });

    const RECORDED_MS = 60_000;
    await act(async () => {
      vi.advanceTimersByTime(RECORDED_MS);
    });

    let reported = 0;
    await act(async () => {
      reported = await result.current.stop();
    });

    const expected = (RECORDED_MS + STOP_GAP_MS) / 1000;
    expect(reported).toBeCloseTo(expected, 1);
    expect(reported).toBeGreaterThanOrEqual(RECORDED_MS / 1000);

    // The same figure is what crash recovery replays from.
    const lastMeta = savedMeta.at(-1);
    expect(lastMeta?.durationSecs).toBeCloseTo(expected, 1);
  });
});
