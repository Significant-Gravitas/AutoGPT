import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const savedMeta: Record<string, unknown>[] = [];

const { store, queue } = vi.hoisted(() => ({
  store: {
    savePart: vi.fn(),
    getMeta: vi.fn(),
    getParts: vi.fn(),
  },
  queue: {
    enqueue: vi.fn(),
    flush: vi.fn(),
    reset: vi.fn(),
  },
}));

vi.mock("../recordingStore", () => ({
  savePart: store.savePart,
  saveMeta: vi.fn(async (meta: Record<string, unknown>) => {
    savedMeta.push(meta);
  }),
  getMeta: store.getMeta,
  getParts: store.getParts,
  partId: (recordingId: string, partIndex: number) =>
    `${recordingId}:${partIndex}`,
}));

vi.mock("../useUploadQueue", async () => {
  const actual =
    await vi.importActual<typeof import("../useUploadQueue")>(
      "../useUploadQueue",
    );
  return {
    buildPart: actual.buildPart,
    useUploadQueue: () => ({
      enqueue: queue.enqueue,
      flush: queue.flush,
      reset: queue.reset,
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

// Hands a chunk over on demand, so the chunk → IndexedDB → queue
// ordering can be observed a step at a time.
class ChunkingMediaRecorder {
  static isTypeSupported() {
    return true;
  }
  static latest: ChunkingMediaRecorder | null = null;
  state: "inactive" | "recording" = "inactive";
  ondataavailable: ((event: { data: Blob }) => void) | null = null;
  onstop: (() => void) | null = null;

  constructor() {
    ChunkingMediaRecorder.latest = this;
  }

  start() {
    this.state = "recording";
  }

  stop() {
    this.state = "inactive";
    this.onstop?.();
  }

  emit(blob: Blob) {
    this.ondataavailable?.({ data: blob });
  }
}

// A room whose loudness the test can change mid-take. Waveform samples
// sit either side of the 128 midpoint; `peak` is the distance from it.
function stubAudioContext(room: { peak: number }) {
  vi.stubGlobal(
    "AudioContext",
    class {
      close = vi.fn().mockResolvedValue(undefined);
      createAnalyser() {
        return {
          fftSize: 0,
          frequencyBinCount: 8,
          getByteTimeDomainData(samples: Uint8Array) {
            samples.fill(128);
            samples[0] = 128 + room.peak;
          },
        };
      }
      createMediaStreamSource() {
        return { connect: () => undefined };
      }
    },
  );
}

function stubGetUserMedia(getUserMedia: ReturnType<typeof vi.fn>) {
  vi.stubGlobal("navigator", { ...navigator, mediaDevices: { getUserMedia } });
}

describe("useBrainDumpRecorder", () => {
  beforeEach(() => {
    savedMeta.length = 0;
    vi.clearAllMocks();
    store.savePart.mockResolvedValue(undefined);
    store.getMeta.mockResolvedValue(null);
    store.getParts.mockResolvedValue([]);
    queue.flush.mockResolvedValue(true);
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

  // The regression: `start()` awaits the permission prompt, and the orb
  // stays tappable across it. A second tap reset `partIndexRef` under the
  // first take, so both recorders wrote into one recording id with
  // interleaved indices (0,2,4… and 1,3,5…) — the server's contiguity
  // check rejects that — and the first stream's tracks were never
  // stopped, leaving the mic indicator lit.
  it("ignores a second tap while the permission prompt is still open", async () => {
    vi.stubGlobal("MediaRecorder", ChunkingMediaRecorder);
    let grantPermission!: (stream: unknown) => void;
    const getUserMedia = vi.fn().mockReturnValue(
      new Promise((resolve) => {
        grantPermission = resolve;
      }),
    );
    stubGetUserMedia(getUserMedia);
    const { result } = renderHook(() => useBrainDumpRecorder());

    const first = result.current.start();
    const second = result.current.start();
    let firstStarted: boolean | undefined;
    let secondStarted: boolean | undefined;
    await act(async () => {
      grantPermission({ getTracks: () => [] });
      firstStarted = await first;
      secondStarted = await second;
    });

    expect(firstStarted).toBe(true);
    expect(secondStarted).toBe(false);
    expect(getUserMedia).toHaveBeenCalledTimes(1);

    // One take, one contiguous run of part indices.
    await act(async () => {
      ChunkingMediaRecorder.latest?.emit(new Blob(["a"]));
      ChunkingMediaRecorder.latest?.emit(new Blob(["b"]));
      await Promise.resolve();
    });
    expect(store.savePart.mock.calls.map(([part]) => part.partIndex)).toEqual([
      0, 1,
    ]);
    const recordingIds = new Set(
      store.savePart.mock.calls.map(([part]) => part.recordingId),
    );
    expect(recordingIds.size).toBe(1);
  });

  it("reports a denial so the step can offer the typed composer", async () => {
    stubGetUserMedia(
      vi.fn().mockRejectedValue(new DOMException("no", "NotAllowedError")),
    );
    const { result } = renderHook(() => useBrainDumpRecorder());

    let started: boolean | undefined;
    await act(async () => {
      started = await result.current.start();
    });

    expect(started).toBe(false);
    expect(result.current.permissionDenied).toBe(true);
    expect(result.current.phase).toBe("idle");
  });

  // A missing microphone is not a refusal, and treating it as one would
  // send the user to the typed fallback with no way back.
  it("does not call a missing microphone a denial", async () => {
    stubGetUserMedia(
      vi.fn().mockRejectedValue(new DOMException("gone", "NotFoundError")),
    );
    const { result } = renderHook(() => useBrainDumpRecorder());

    let started: boolean | undefined;
    await act(async () => {
      started = await result.current.start();
    });

    expect(started).toBe(false);
    expect(result.current.permissionDenied).toBe(false);
  });

  // The ordering is the contract: a chunk is durable before the network
  // is ever offered it.
  it("persists a chunk before handing it to the upload queue", async () => {
    vi.stubGlobal("MediaRecorder", ChunkingMediaRecorder);
    const { result } = renderHook(() => useBrainDumpRecorder());

    await act(async () => {
      await result.current.start();
    });
    await act(async () => {
      ChunkingMediaRecorder.latest?.emit(new Blob(["chunk"]));
      await Promise.resolve();
    });

    const saved = store.savePart.mock.calls[0][0];
    expect(saved).toMatchObject({ partIndex: 0, uploaded: false });
    expect(saved.recordingId).toBe(result.current.recordingId);
    expect(queue.enqueue).toHaveBeenCalledWith(saved);
    expect(store.savePart.mock.invocationCallOrder[0]).toBeLessThan(
      queue.enqueue.mock.invocationCallOrder[0],
    );
    expect(result.current.isSavedLocally).toBe(true);
  });

  it("ignores an empty chunk", async () => {
    vi.stubGlobal("MediaRecorder", ChunkingMediaRecorder);
    const { result } = renderHook(() => useBrainDumpRecorder());

    await act(async () => {
      await result.current.start();
    });
    await act(async () => {
      ChunkingMediaRecorder.latest?.emit(new Blob([]));
      await Promise.resolve();
    });

    expect(store.savePart).not.toHaveBeenCalled();
    expect(queue.enqueue).not.toHaveBeenCalled();
  });

  // Private-mode Safari and hardened profiles have no usable IndexedDB.
  // The chunk still goes out — the reassurance on screen just stays off
  // rather than lying about a backup that does not exist.
  it("still uploads a chunk it could not persist, without claiming it saved", async () => {
    vi.stubGlobal("MediaRecorder", ChunkingMediaRecorder);
    store.savePart.mockRejectedValue(new Error("no indexeddb"));
    const { result } = renderHook(() => useBrainDumpRecorder());

    await act(async () => {
      await result.current.start();
    });
    await act(async () => {
      ChunkingMediaRecorder.latest?.emit(new Blob(["chunk"]));
      await Promise.resolve();
    });

    expect(queue.enqueue).toHaveBeenCalledTimes(1);
    expect(result.current.isSavedLocally).toBe(false);
  });

  it("notices speech and never takes it back during a pause", async () => {
    const room = { peak: 40 };
    stubAudioContext(room);
    const { result } = renderHook(() => useBrainDumpRecorder());

    await act(async () => {
      await result.current.start();
    });
    await act(async () => {
      vi.advanceTimersByTime(1_000);
    });
    expect(result.current.hasSpoken).toBe(true);

    // One-way: the nudge is for someone who has not started, so a quiet
    // moment must not bring it back.
    room.peak = 0;
    await act(async () => {
      vi.advanceTimersByTime(10_000);
    });
    expect(result.current.hasSpoken).toBe(true);
  });

  it("treats room noise below the threshold as silence", async () => {
    stubAudioContext({ peak: 5 });
    const { result } = renderHook(() => useBrainDumpRecorder());

    await act(async () => {
      await result.current.start();
    });
    await act(async () => {
      vi.advanceTimersByTime(10_000);
    });

    expect(result.current.hasSpoken).toBe(false);
  });

  // 30 minutes stops the recorder but keeps every second captured.
  it("stops itself at the hard stop", async () => {
    vi.stubGlobal("MediaRecorder", ChunkingMediaRecorder);
    const { result } = renderHook(() => useBrainDumpRecorder());

    await act(async () => {
      await result.current.start();
    });
    await act(async () => {
      vi.advanceTimersByTime(1_800_000);
    });

    expect(result.current.phase).toBe("stopped");
    expect(result.current.getElapsedSeconds()).toBeGreaterThanOrEqual(1800);
    expect(savedMeta.at(-1)?.durationSecs).toBeGreaterThanOrEqual(1800);
    // The step submits on this, not on the phase: a restart stops the
    // recorder too, and that take is thrown away rather than sent.
    expect(result.current.hitTimeLimit).toBe(true);
  });

  it("does not call an ordinary stop a time limit", async () => {
    const { result } = renderHook(() => useBrainDumpRecorder());

    await act(async () => {
      await result.current.start();
    });
    await act(async () => {
      vi.advanceTimersByTime(60_000);
      await result.current.stop();
    });

    expect(result.current.phase).toBe("stopped");
    expect(result.current.hitTimeLimit).toBe(false);
  });

  // The regression: the meta row was written with `durationSecs: 0` at
  // start and only refreshed by `stop()`. A crash or a refresh — the exact
  // case recovery exists for — never gets there, so the prompt offered
  // back "the 0:00 you already recorded" and finalize submitted 0, which
  // skips the backend's splitting of long takes.
  it("keeps the stored duration current while the take runs", async () => {
    const { result } = renderHook(() => useBrainDumpRecorder());

    await act(async () => {
      await result.current.start();
    });
    expect(savedMeta.at(-1)?.durationSecs).toBe(0);

    await act(async () => {
      vi.advanceTimersByTime(60_000);
      await Promise.resolve();
    });

    const stored = savedMeta.at(-1);
    expect(stored?.durationSecs).toBeGreaterThanOrEqual(55);
    expect(stored?.finalized).toBe(false);
    expect(stored?.recordingId).toBe(result.current.recordingId);
  });

  it("reports the known duration when there is nothing to stop", async () => {
    const { result } = renderHook(() => useBrainDumpRecorder());

    let reported: number | undefined;
    await act(async () => {
      reported = await result.current.stop();
    });

    expect(reported).toBe(0);
    expect(result.current.phase).toBe("idle");
  });

  describe("recovery", () => {
    it("offers an unfinalized take back", async () => {
      const meta = {
        recordingId: "rec-crashed",
        mimeType: "audio/webm",
        startedAt: 1,
        durationSecs: 42,
        finalized: false,
      };
      store.getMeta.mockResolvedValue(meta);
      const { result } = renderHook(() => useBrainDumpRecorder());

      await expect(result.current.findRecoverable()).resolves.toEqual(meta);
    });

    // A finished take is not something to recover from.
    it("ignores a finalized take", async () => {
      store.getMeta.mockResolvedValue({
        recordingId: "rec-done",
        mimeType: "audio/webm",
        startedAt: 1,
        durationSecs: 42,
        finalized: true,
      });
      const { result } = renderHook(() => useBrainDumpRecorder());

      await expect(result.current.findRecoverable()).resolves.toBeNull();
    });

    it("offers nothing when local storage cannot be read", async () => {
      store.getMeta.mockRejectedValue(new Error("no indexeddb"));
      const { result } = renderHook(() => useBrainDumpRecorder());

      await expect(result.current.findRecoverable()).resolves.toBeNull();
    });

    // After a crash the upload queue is gone with the page, so every part
    // is replayed — including ones already marked uploaded, since the
    // server's buffer may have expired and a missing part 0 is fatal.
    it("replays every stored part when a take is adopted", async () => {
      const parts = [
        { id: "rec-crashed:0", partIndex: 0, uploaded: true },
        { id: "rec-crashed:1", partIndex: 1, uploaded: false },
      ];
      store.getParts.mockResolvedValue(parts);
      const { result } = renderHook(() => useBrainDumpRecorder());

      await act(async () => {
        await result.current.adoptRecovered("rec-crashed", "audio/mp4");
      });

      expect(queue.enqueue.mock.calls.map(([p]) => p)).toEqual(parts);
      expect(result.current.recordingId).toBe("rec-crashed");
      expect(result.current.mimeType).toBe("audio/mp4");
      expect(result.current.phase).toBe("stopped");
    });

    it("replays nothing when the stored parts cannot be read", async () => {
      store.getParts.mockRejectedValue(new Error("no indexeddb"));
      const { result } = renderHook(() => useBrainDumpRecorder());

      let replayed: unknown[] | undefined;
      await act(async () => {
        replayed = await result.current.resendAllParts("rec-crashed");
      });

      expect(replayed).toEqual([]);
      expect(queue.enqueue).not.toHaveBeenCalled();
    });
  });

  it("releases the microphone when the step unmounts", async () => {
    const track = { stop: vi.fn() };
    stubGetUserMedia(vi.fn().mockResolvedValue({ getTracks: () => [track] }));
    const { result, unmount } = renderHook(() => useBrainDumpRecorder());

    await act(async () => {
      await result.current.start();
    });
    unmount();

    expect(track.stop).toHaveBeenCalled();
  });
});
