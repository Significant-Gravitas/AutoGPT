import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { RecordingPart } from "../recordingStore";

const uploadBrainDumpPart = vi.fn();

vi.mock("@/app/api/__generated__/endpoints/brain-dump/brain-dump", () => ({
  uploadBrainDumpPart: (...args: unknown[]) => uploadBrainDumpPart(...args),
}));

vi.mock("../recordingStore", () => ({
  markPartUploaded: vi.fn().mockResolvedValue(undefined),
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
});
