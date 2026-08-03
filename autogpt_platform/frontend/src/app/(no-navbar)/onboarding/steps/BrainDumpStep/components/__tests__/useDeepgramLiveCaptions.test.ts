import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useDeepgramLiveCaptions } from "../useDeepgramLiveCaptions";
import {
  FakeAudioContext,
  FakeWebSocket,
  PCM_EXPECTED,
  PCM_SAMPLES,
  fakeStream,
  stubTokenFetch,
} from "./captionsTestDoubles";

function renderDeepgram(audioStream: MediaStream | null = fakeStream()) {
  return renderHook(() =>
    useDeepgramLiveCaptions({ enabled: true, audioStream }),
  );
}

async function connected() {
  const rendered = renderDeepgram();
  await waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1));
  const socket = FakeWebSocket.last();
  act(() => socket.open());
  await waitFor(() => expect(rendered.result.current.status).toBe("live"));
  return { ...rendered, socket };
}

function texts(words: { text: string }[]) {
  return words.map((word) => word.text);
}

function results(transcript: string, isFinal: boolean) {
  return {
    type: "Results",
    is_final: isFinal,
    channel: { alternatives: [{ transcript }] },
  };
}

describe("useDeepgramLiveCaptions", () => {
  beforeEach(() => {
    FakeWebSocket.reset();
    FakeAudioContext.reset();
    stubTokenFetch();
    vi.stubGlobal("WebSocket", FakeWebSocket);
    vi.stubGlobal("AudioContext", FakeAudioContext);
  });

  afterEach(() => vi.unstubAllGlobals());

  it("stays idle while disabled or without a stream", async () => {
    const { result } = renderHook(() =>
      useDeepgramLiveCaptions({ enabled: false, audioStream: fakeStream() }),
    );
    renderDeepgram(null);

    expect(fetch).not.toHaveBeenCalled();
    expect(result.current.status).toBe("idle");
  });

  // Browsers cannot set WebSocket headers, so the disposable token has to
  // ride in the subprotocol. Sending it any other way authenticates nothing.
  it("mints a Deepgram token and passes it as a bearer subprotocol", async () => {
    const { result } = renderDeepgram();

    await waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1));

    expect(fetch).toHaveBeenCalledWith(
      "/api/transcribe/live-session",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ provider: "deepgram" }),
      }),
    );
    const socket = FakeWebSocket.last();
    expect(socket.protocols).toEqual(["bearer", "tok"]);
    const url = new URL(socket.url);
    expect(url.host).toBe("api.deepgram.com");
    expect(url.searchParams.get("model")).toBe("nova-3");
    expect(url.searchParams.get("sample_rate")).toBe("24000");
    expect(result.current.status).toBe("connecting");
  });

  it("only reports live once the audio graph is running", async () => {
    const { result } = await connected();

    const context = FakeAudioContext.last();
    expect(context.sampleRate).toBe(24000);
    expect(context.processor?.connectCount).toBe(1);
    expect(result.current.status).toBe("live");
  });

  it("falls back instead of reporting live when the audio graph throws", async () => {
    vi.stubGlobal(
      "AudioContext",
      class {
        constructor() {
          throw new Error("no audio device");
        }
      },
    );
    const { result } = renderDeepgram();

    await waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1));
    act(() => FakeWebSocket.last().open());

    await waitFor(() => expect(result.current.status).toBe("failed"));
  });

  it("replaces the tail on interims and commits it when final", async () => {
    const { result, socket } = await connected();

    act(() => socket.emit(results("build me", false)));
    expect(texts(result.current.words)).toEqual(["build", "me"]);

    act(() => socket.emit(results("build me a", false)));
    expect(texts(result.current.words)).toEqual(["build", "me", "a"]);

    act(() => socket.emit(results("build me a bot.", true)));
    expect(texts(result.current.words)).toEqual(["build", "me", "a", "bot."]);

    // The finalised phrase stays while the next interim tail grows behind it.
    act(() => socket.emit(results("It should", false)));
    expect(texts(result.current.words)).toEqual([
      "build",
      "me",
      "a",
      "bot.",
      "It",
      "should",
    ]);
  });

  it("drops the interim tail on an empty final without losing the phrase", async () => {
    const { result, socket } = await connected();

    act(() => socket.emit(results("hello", false)));
    act(() => socket.emit(results("hello", true)));
    // Deepgram closes an utterance with an empty final; nothing to commit.
    act(() => socket.emit(results("", true)));

    expect(texts(result.current.words)).toEqual(["hello"]);
  });

  it("ignores non-Results frames such as metadata", async () => {
    const { result, socket } = await connected();

    act(() => socket.emit({ type: "Metadata", request_id: "abc" }));

    expect(result.current.words).toEqual([]);
    expect(result.current.status).toBe("live");
  });

  it("keeps a word's id while its text holds", async () => {
    const { result, socket } = await connected();

    act(() => socket.emit(results("hel", false)));
    const firstId = result.current.words[0].id;

    act(() => socket.emit(results("hello there", false)));

    expect(result.current.words[0].id).toBe(firstId);
    expect(result.current.words[1].id).not.toBe(firstId);
  });

  it("shows only the last 24 words", async () => {
    const { result, socket } = await connected();
    const spoken = Array.from({ length: 30 }, (_, index) => `w${index}`);

    act(() => socket.emit(results(spoken.join(" "), true)));

    expect(result.current.words).toHaveLength(24);
    expect(texts(result.current.words)).toEqual(spoken.slice(-24));
  });

  it.each(["onerror", "onclose"] as const)(
    "degrades to failed when the socket fires %s mid-recording",
    async (handler) => {
      const { result, socket } = await connected();

      act(() => socket[handler]?.());

      await waitFor(() => expect(result.current.status).toBe("failed"));
    },
  );

  it.each<[string, () => void]>([
    ["the token endpoint rejects", () => stubTokenFetch({ ok: false })],
    [
      "the token endpoint returns no token",
      () => stubTokenFetch({ token: undefined }),
    ],
    [
      "the token request throws",
      () => vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("dns"))),
    ],
  ])("fails when %s", async (_label, stub) => {
    stub();
    const { result } = renderDeepgram();

    await waitFor(() => expect(result.current.status).toBe("failed"));
    expect(FakeWebSocket.instances).toHaveLength(0);
  });

  it("streams raw PCM16 only while the socket is open", async () => {
    const { socket } = await connected();
    const processor = FakeAudioContext.last().processor!;

    socket.readyState = 0;
    processor.feed(PCM_SAMPLES);
    expect(socket.sent).toHaveLength(0);

    socket.readyState = FakeWebSocket.OPEN;
    processor.feed(PCM_SAMPLES);

    expect(socket.sent).toHaveLength(1);
    expect(Array.from(new Int16Array(socket.sent[0] as ArrayBuffer))).toEqual(
      PCM_EXPECTED,
    );
  });

  it("tears the socket and audio graph down on unmount", async () => {
    const { unmount, result, socket } = await connected();
    const context = FakeAudioContext.last();

    act(() => socket.emit(results("hi", false)));
    expect(result.current.words).toHaveLength(1);

    unmount();

    expect(socket.closeCount).toBe(1);
    expect(context.processor?.disconnectCount).toBe(1);
    expect(context.closeCount).toBe(1);
  });

  it("does not open a socket when the take ends before the token lands", async () => {
    let release!: (value: {
      ok: boolean;
      json: () => Promise<unknown>;
    }) => void;
    vi.stubGlobal(
      "fetch",
      vi.fn().mockReturnValue(
        new Promise((resolve) => {
          release = resolve;
        }),
      ),
    );

    const { unmount } = renderDeepgram();
    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(1));
    unmount();

    await act(async () => {
      release({ ok: true, json: async () => ({ token: "tok" }) });
    });

    expect(FakeWebSocket.instances).toHaveLength(0);
  });

  it("goes back to idle and closes the socket when the take stops", async () => {
    const audioStream = fakeStream();
    const { result, rerender } = renderHook(
      ({ enabled }: { enabled: boolean }) =>
        useDeepgramLiveCaptions({ enabled, audioStream }),
      { initialProps: { enabled: true } },
    );

    await waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1));

    rerender({ enabled: false });

    expect(result.current.status).toBe("idle");
    expect(FakeWebSocket.last().closeCount).toBe(1);
  });
});
