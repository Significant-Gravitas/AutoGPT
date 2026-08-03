import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { RecordingPart } from "../recordingStore";

const uploadBrainDumpPart = vi.fn();
const markPartUploaded = vi.fn();

vi.mock("@/app/api/__generated__/endpoints/brain-dump/brain-dump", () => ({
  uploadBrainDumpPart: (...args: unknown[]) => uploadBrainDumpPart(...args),
}));

vi.mock("../recordingStore", () => ({
  markPartUploaded: (...args: unknown[]) => markPartUploaded(...args),
  partId: (recordingId: string, partIndex: number) =>
    `${recordingId}:${partIndex}`,
}));

import { buildPart, useUploadQueue } from "../useUploadQueue";

function part(index: number): RecordingPart {
  return buildPart("rec-1", index, new Blob(["x"]) as Blob);
}

// Resolves only when the test says so, standing in for an upload that is
// still on the wire when the user presses "I'm done".
function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}

describe("useUploadQueue", () => {
  beforeEach(() => {
    uploadBrainDumpPart.mockReset();
    markPartUploaded.mockReset();
    markPartUploaded.mockResolvedValue(undefined);
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("waits for an in-flight upload instead of reporting failure", async () => {
    // The regression: pressing "I'm done" while the last chunk is still
    // uploading used to make flush() await a no-op, see a non-empty queue
    // and report failure — so finalize was never called at all.
    const inFlight = deferred();
    uploadBrainDumpPart.mockReturnValueOnce(inFlight.promise);

    const { result } = renderHook(() => useUploadQueue());

    act(() => {
      result.current.enqueue(part(0));
    });

    let flushed: boolean | undefined;
    act(() => {
      void result.current.flush().then((value) => {
        flushed = value;
      });
    });

    // Let every pending microtask settle while the upload is still in
    // flight. This is the assertion that fails against the old code: a
    // no-op drain let all of flush()'s passes complete here and resolve
    // to false, so "I'm done" reported failure without calling finalize.
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    expect(flushed).toBeUndefined();

    await act(async () => {
      inFlight.resolve();
      await Promise.resolve();
    });

    expect(flushed).toBe(true);
    expect(uploadBrainDumpPart).toHaveBeenCalledTimes(1);
  });

  it("uploads every queued part in order", async () => {
    const seen: number[] = [];
    uploadBrainDumpPart.mockImplementation(
      async ({ part_index }: { part_index: number }) => {
        seen.push(part_index);
      },
    );

    const { result } = renderHook(() => useUploadQueue());

    await act(async () => {
      result.current.enqueue(part(0));
      result.current.enqueue(part(1));
      result.current.enqueue(part(2));
      await result.current.flush();
    });

    expect(seen).toEqual([0, 1, 2]);
  });

  it("reports failure when a part cannot be uploaded", async () => {
    uploadBrainDumpPart.mockRejectedValue(new Error("offline"));
    vi.useFakeTimers();

    const { result } = renderHook(() => useUploadQueue());

    let flushed: boolean | undefined;
    await act(async () => {
      result.current.enqueue(part(0));
      const pending = result.current.flush().then((value) => {
        flushed = value;
      });
      await vi.runAllTimersAsync();
      await pending;
    });

    expect(flushed).toBe(false);
    vi.useRealTimers();
  });

  it("keeps the failed part at the head so nothing overtakes it", async () => {
    const attempted: number[] = [];
    uploadBrainDumpPart.mockImplementation(
      async ({ part_index }: { part_index: number }) => {
        attempted.push(part_index);
        throw new Error("offline");
      },
    );
    vi.useFakeTimers();

    const { result } = renderHook(() => useUploadQueue());

    await act(async () => {
      result.current.enqueue(part(0));
      result.current.enqueue(part(1));
      const pending = result.current.flush();
      await vi.runAllTimersAsync();
      await pending;
    });

    expect(new Set(attempted)).toEqual(new Set([0]));
    vi.useRealTimers();
  });

  it("counts the parts still waiting to go out", async () => {
    const inFlight = deferred();
    uploadBrainDumpPart.mockReturnValueOnce(inFlight.promise);
    uploadBrainDumpPart.mockResolvedValue(undefined);

    const { result } = renderHook(() => useUploadQueue());

    act(() => {
      result.current.enqueue(part(0));
      result.current.enqueue(part(1));
    });
    expect(result.current.pendingCount).toBe(2);

    await act(async () => {
      inFlight.resolve();
      await result.current.flush();
    });

    expect(result.current.pendingCount).toBe(0);
  });

  it("marks a part uploaded once the server has it", async () => {
    uploadBrainDumpPart.mockResolvedValue(undefined);

    const { result } = renderHook(() => useUploadQueue());

    await act(async () => {
      result.current.enqueue(part(3));
      await result.current.flush();
    });

    expect(markPartUploaded).toHaveBeenCalledWith("rec-1:3");
  });

  // Bookkeeping only, and the part is already on the server. Where
  // IndexedDB is unavailable this rejects — reporting failure for a dump
  // that fully uploaded would send the user to the error screen.
  it("still reports success when the uploaded flag cannot be written", async () => {
    uploadBrainDumpPart.mockResolvedValue(undefined);
    markPartUploaded.mockRejectedValue(new Error("no indexeddb"));

    const { result } = renderHook(() => useUploadQueue());

    let flushed: boolean | undefined;
    await act(async () => {
      result.current.enqueue(part(0));
      flushed = await result.current.flush();
    });

    expect(flushed).toBe(true);
    expect(result.current.pendingCount).toBe(0);
  });

  it("starts out parked when the browser is already offline", () => {
    vi.stubGlobal("navigator", { onLine: true });
    const online = renderHook(() => useUploadQueue());
    expect(online.result.current.isOffline).toBe(false);

    vi.stubGlobal("navigator", { onLine: false });
    const offline = renderHook(() => useUploadQueue());
    expect(offline.result.current.isOffline).toBe(true);
  });

  it("parks the queue on a dropped connection and replays it on reconnect", async () => {
    uploadBrainDumpPart.mockRejectedValue(new Error("offline"));
    vi.useFakeTimers();

    const { result } = renderHook(() => useUploadQueue());

    let flushed: boolean | undefined;
    await act(async () => {
      result.current.enqueue(part(0));
      const pending = result.current.flush().then((value) => {
        flushed = value;
      });
      await vi.runAllTimersAsync();
      await pending;
    });
    expect(flushed).toBe(false);
    expect(result.current.pendingCount).toBe(1);

    act(() => {
      window.dispatchEvent(new Event("offline"));
    });
    expect(result.current.isOffline).toBe(true);

    // The part is still at the head, so coming back online replays it
    // rather than leaving a hole in the audio.
    const attemptsWhileOffline = uploadBrainDumpPart.mock.calls.length;
    uploadBrainDumpPart.mockResolvedValue(undefined);
    await act(async () => {
      window.dispatchEvent(new Event("online"));
      await vi.runAllTimersAsync();
    });

    expect(result.current.isOffline).toBe(false);
    expect(result.current.pendingCount).toBe(0);
    // Exactly one more attempt: the reconnect resumes the queue rather
    // than restarting the retry ladder from scratch.
    expect(uploadBrainDumpPart).toHaveBeenCalledTimes(attemptsWhileOffline + 1);
    expect(uploadBrainDumpPart).toHaveBeenLastCalledWith(
      expect.objectContaining({ part_index: 0, recording_id: "rec-1" }),
    );
  });

  it("drops the queue on reset so a restarted take sends nothing old", async () => {
    uploadBrainDumpPart.mockRejectedValue(new Error("offline"));
    vi.useFakeTimers();

    const { result } = renderHook(() => useUploadQueue());

    await act(async () => {
      result.current.enqueue(part(0));
      const pending = result.current.flush();
      await vi.runAllTimersAsync();
      await pending;
    });
    expect(result.current.pendingCount).toBe(1);

    act(() => {
      result.current.reset();
    });
    expect(result.current.pendingCount).toBe(0);

    uploadBrainDumpPart.mockClear();
    let flushed: boolean | undefined;
    await act(async () => {
      flushed = await result.current.flush();
    });

    expect(flushed).toBe(true);
    expect(uploadBrainDumpPart).not.toHaveBeenCalled();
  });
});
