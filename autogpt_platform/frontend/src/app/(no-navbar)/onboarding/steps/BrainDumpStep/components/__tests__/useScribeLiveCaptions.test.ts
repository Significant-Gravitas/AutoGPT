import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useScribeLiveCaptions } from "../useScribeLiveCaptions";

function fakeStream() {
  return { getTracks: () => [] } as unknown as MediaStream;
}

describe("useScribeLiveCaptions", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ token: "tok" }),
      }),
    );
    vi.stubGlobal(
      "WebSocket",
      class {
        onopen: (() => void) | null = null;
        onmessage: (() => void) | null = null;
        onerror: (() => void) | null = null;
        onclose: (() => void) | null = null;
        readyState = 0;
        send() {}
        close() {}
      },
    );
  });

  afterEach(() => vi.unstubAllGlobals());

  // Restarting a take stops the old stream before starting a new one, so
  // the hook is re-rendered with a null stream in between. It has to pick
  // the new one up when it arrives, otherwise the second take records
  // with no captions at all.
  it("connects once the audio stream arrives after mount", async () => {
    const { rerender } = renderHook(
      ({ audioStream }: { audioStream: MediaStream | null }) =>
        useScribeLiveCaptions({ enabled: true, audioStream }),
      { initialProps: { audioStream: null as MediaStream | null } },
    );

    expect(fetch).not.toHaveBeenCalled();

    rerender({ audioStream: fakeStream() });

    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(1));
  });

  it("stays idle while disabled even with a stream", async () => {
    renderHook(() =>
      useScribeLiveCaptions({ enabled: false, audioStream: fakeStream() }),
    );

    expect(fetch).not.toHaveBeenCalled();
  });
});
