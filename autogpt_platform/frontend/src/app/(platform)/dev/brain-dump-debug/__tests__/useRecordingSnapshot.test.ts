import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// The poll interval and the manual refresh both read the same store, so
// they cannot be told apart through the page UI without waiting a whole
// second per assertion. This is the one piece of the debug page that has
// to be driven directly.
interface FakeMeta {
  recordingId: string;
  mimeType: string;
  startedAt: number;
  durationSecs: number;
  finalized: boolean;
}

const { store } = vi.hoisted(() => ({
  store: {
    meta: null as FakeMeta | null,
    reads: 0,
  },
}));

vi.mock(
  "@/app/(no-navbar)/onboarding/steps/BrainDumpStep/recordingStore",
  () => ({
    isIndexedDBAvailable: () => true,
    getMeta: async () => {
      store.reads += 1;
      return store.meta;
    },
    getParts: async () => [],
  }),
);

import { SNAPSHOT_POLL_MS } from "../helpers";
import { useRecordingSnapshot } from "../useRecordingSnapshot";

function meta(recordingId: string): FakeMeta {
  return {
    recordingId,
    mimeType: "audio/webm",
    startedAt: 1,
    durationSecs: 60,
    finalized: false,
  };
}

beforeEach(() => {
  store.meta = null;
  store.reads = 0;
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("useRecordingSnapshot", () => {
  it("starts empty, then reads once on mount", async () => {
    store.meta = meta("rec-1");

    const { result } = renderHook(() => useRecordingSnapshot());

    expect(result.current.snapshot.meta).toBeNull();
    expect(result.current.snapshot.readAt).toBeNull();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(store.reads).toBe(1);
    expect(result.current.snapshot.meta?.recordingId).toBe("rec-1");
  });

  it("keeps re-reading on the poll interval and stops on unmount", async () => {
    const { unmount } = renderHook(() => useRecordingSnapshot());

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(store.reads).toBe(1);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(SNAPSHOT_POLL_MS * 3);
    });
    expect(store.reads).toBe(4);

    unmount();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(SNAPSHOT_POLL_MS * 5);
    });
    // The interval must be torn down, or the page keeps hammering
    // IndexedDB after it is gone.
    expect(store.reads).toBe(4);
  });

  it("applies the store's current contents as soon as refresh resolves", async () => {
    store.meta = meta("rec-1");
    const { result } = renderHook(() => useRecordingSnapshot());

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(result.current.snapshot.meta?.recordingId).toBe("rec-1");

    store.meta = meta("rec-2");
    const readsBeforeRefresh = store.reads;

    // No timers are advanced here, so the poll cannot be what updates it.
    await act(async () => {
      await result.current.refresh();
    });

    expect(store.reads).toBe(readsBeforeRefresh + 1);
    expect(result.current.snapshot.meta?.recordingId).toBe("rec-2");
  });
});
