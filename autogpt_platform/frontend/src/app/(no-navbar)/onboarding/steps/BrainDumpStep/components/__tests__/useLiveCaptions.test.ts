import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  FakeAudioContext,
  FakeSpeechRecognition,
  fakeStream,
} from "./captionsTestDoubles";

type CloudState = { words: { id: number; text: string }[]; status: string };

const scribeState: CloudState = { words: [], status: "connecting" };
const deepgramState: CloudState = { words: [], status: "connecting" };
const scribeArgs = vi.fn();
const deepgramArgs = vi.fn();

vi.mock("../useScribeLiveCaptions", () => ({
  useScribeLiveCaptions: (args: unknown) => {
    scribeArgs(args);
    return scribeState;
  },
}));

vi.mock("../useDeepgramLiveCaptions", () => ({
  useDeepgramLiveCaptions: (args: unknown) => {
    deepgramArgs(args);
    return deepgramState;
  },
}));

import { type CaptionsEngine, useLiveCaptions } from "../useLiveCaptions";

const stream = fakeStream();

function render(
  props: Partial<{
    isRecording: boolean;
    audioStream: MediaStream | null;
    engine: CaptionsEngine;
  }> = {},
) {
  return renderHook(() =>
    useLiveCaptions({
      isRecording: true,
      audioStream: stream,
      ...props,
    }),
  );
}

function texts(words: { text: string }[]) {
  return words.map((word) => word.text);
}

function lastArgs(spy: typeof scribeArgs) {
  return spy.mock.lastCall?.[0] as { enabled: boolean } | undefined;
}

describe("useLiveCaptions", () => {
  beforeEach(() => {
    scribeState.words = [];
    scribeState.status = "connecting";
    deepgramState.words = [];
    deepgramState.status = "connecting";
    scribeArgs.mockClear();
    deepgramArgs.mockClear();
    FakeSpeechRecognition.reset();
    FakeAudioContext.reset();
    vi.stubGlobal("SpeechRecognition", FakeSpeechRecognition);
    vi.stubGlobal("AudioContext", FakeAudioContext);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  describe("engine selection", () => {
    it("only enables the engine the caller asked for", () => {
      render({ engine: "elevenlabs" });

      expect(lastArgs(scribeArgs)?.enabled).toBe(true);
      expect(lastArgs(deepgramArgs)?.enabled).toBe(false);
      expect(FakeSpeechRecognition.instances).toHaveLength(0);
    });

    it("routes to Deepgram when the caller picks it", () => {
      deepgramState.words = [{ id: 1, text: "deepgram" }];
      scribeState.words = [{ id: 2, text: "scribe" }];

      const { result } = render({ engine: "deepgram" });

      expect(lastArgs(deepgramArgs)?.enabled).toBe(true);
      expect(lastArgs(scribeArgs)?.enabled).toBe(false);
      expect(texts(result.current.words)).toEqual(["deepgram"]);
    });

    it("leaves both cloud engines disabled while not recording", () => {
      render({ engine: "elevenlabs", isRecording: false });

      expect(lastArgs(scribeArgs)?.enabled).toBe(false);
      expect(lastArgs(deepgramArgs)?.enabled).toBe(false);
      expect(FakeSpeechRecognition.instances).toHaveLength(0);
    });

    it("never touches a cloud engine on the browser engine", () => {
      render({ engine: "browser" });

      expect(lastArgs(scribeArgs)?.enabled).toBe(false);
      expect(lastArgs(deepgramArgs)?.enabled).toBe(false);
      expect(FakeSpeechRecognition.instances).toHaveLength(1);
    });

    // A cloud engine works wherever a mic does, so a browser without
    // SpeechRecognition must not be told captions are unsupported — that
    // would drop it to the level meter for no reason.
    it("reports supported on a cloud engine even without SpeechRecognition", () => {
      vi.stubGlobal("SpeechRecognition", undefined);
      vi.stubGlobal("webkitSpeechRecognition", undefined);

      const { result } = render({ engine: "elevenlabs" });

      expect(result.current.isSpeechSupported).toBe(true);
    });

    it("picks up the webkit-prefixed recogniser", () => {
      vi.stubGlobal("SpeechRecognition", undefined);
      vi.stubGlobal("webkitSpeechRecognition", FakeSpeechRecognition);

      const { result } = render({ engine: "browser" });

      expect(result.current.isSpeechSupported).toBe(true);
      expect(FakeSpeechRecognition.instances).toHaveLength(1);
    });
  });

  describe("fallback to the browser engine", () => {
    // The whole point of the fallback chain: no key, dead socket or a
    // provider error must not leave the user staring at an empty caption
    // box for the rest of the take.
    it("starts the browser recogniser when the cloud engine gives up", () => {
      scribeState.words = [{ id: 9, text: "stale" }];
      const { result, rerender } = renderHook(() =>
        useLiveCaptions({
          isRecording: true,
          audioStream: stream,
          engine: "elevenlabs",
        }),
      );

      expect(FakeSpeechRecognition.instances).toHaveLength(0);
      expect(texts(result.current.words)).toEqual(["stale"]);

      scribeState.status = "failed";
      rerender();

      expect(FakeSpeechRecognition.instances).toHaveLength(1);
      act(() => FakeSpeechRecognition.last().say("hello again"));
      expect(texts(result.current.words)).toEqual(["hello", "again"]);
    });

    it("degrades to the level meter when neither engine is available", () => {
      vi.stubGlobal("SpeechRecognition", undefined);
      vi.stubGlobal("webkitSpeechRecognition", undefined);
      deepgramState.status = "failed";

      const { result } = render({ engine: "deepgram" });

      expect(result.current.isSpeechSupported).toBe(false);
      expect(FakeAudioContext.instances).toHaveLength(1);
    });
  });

  describe("browser recogniser", () => {
    it("configures a continuous session with interim results", () => {
      render({ engine: "browser" });

      const recognition = FakeSpeechRecognition.last();
      expect(recognition.continuous).toBe(true);
      expect(recognition.interimResults).toBe(true);
      expect(recognition.startCount).toBe(1);
    });

    it("does not listen while the take is stopped", () => {
      render({ engine: "browser", isRecording: false });

      expect(FakeSpeechRecognition.instances).toHaveLength(0);
    });

    it("joins every utterance of a session into one growing line", () => {
      const { result } = render({ engine: "browser" });
      const recognition = FakeSpeechRecognition.last();

      act(() => recognition.say("I want", "a bot"));

      expect(texts(result.current.words)).toEqual(["I", "want", "a", "bot"]);
    });

    // Each restarted session reports results from an empty list, so words
    // already committed have to be kept as a base or the line snaps back
    // to a single word every time the recogniser cycles.
    it("keeps earlier words when the recogniser restarts itself", () => {
      const { result } = render({ engine: "browser" });
      const recognition = FakeSpeechRecognition.last();

      act(() => recognition.say("first phrase"));
      act(() => recognition.onend?.());
      act(() => recognition.say("second"));

      expect(texts(result.current.words)).toEqual([
        "first",
        "phrase",
        "second",
      ]);
    });

    it("keeps a word's id while its text holds", () => {
      const { result } = render({ engine: "browser" });
      const recognition = FakeSpeechRecognition.last();

      act(() => recognition.say("hel"));
      const firstId = result.current.words[0].id;

      act(() => recognition.say("hello there"));

      expect(result.current.words[0].id).toBe(firstId);
      expect(result.current.words[1].id).not.toBe(firstId);
    });

    it("shows only the last 24 words", () => {
      const { result } = render({ engine: "browser" });
      const spoken = Array.from({ length: 30 }, (_, index) => `w${index}`);

      act(() => FakeSpeechRecognition.last().say(spoken.join(" ")));

      expect(result.current.words).toHaveLength(24);
      expect(texts(result.current.words)).toEqual(spoken.slice(-24));
    });

    // A session that dies the instant it starts means a fatal condition
    // (mic revoked, no speech service). Retrying forever would spin the
    // tab for the rest of the recording.
    it("gives up after five restarts that die immediately", () => {
      vi.useFakeTimers();
      render({ engine: "browser" });
      const recognition = FakeSpeechRecognition.last();

      for (let attempt = 0; attempt < 8; attempt++) {
        act(() => recognition.onend?.());
      }

      expect(recognition.startCount).toBe(6);
    });

    it("keeps restarting sessions that survived long enough", () => {
      vi.useFakeTimers();
      render({ engine: "browser" });
      const recognition = FakeSpeechRecognition.last();

      for (let attempt = 0; attempt < 8; attempt++) {
        vi.advanceTimersByTime(2000);
        act(() => recognition.onend?.());
      }

      expect(recognition.startCount).toBe(9);
    });

    it("forgives rapid restarts once words come through again", () => {
      vi.useFakeTimers();
      render({ engine: "browser" });
      const recognition = FakeSpeechRecognition.last();

      for (let attempt = 0; attempt < 4; attempt++) {
        act(() => recognition.onend?.());
      }
      act(() => recognition.say("still here"));
      for (let attempt = 0; attempt < 4; attempt++) {
        act(() => recognition.onend?.());
      }

      // 1 initial + 4 + 4 restarts: the counter reset means the second
      // burst never hits the give-up threshold.
      expect(recognition.startCount).toBe(9);
    });

    it("survives a recogniser error without dropping the line", () => {
      const { result } = render({ engine: "browser" });
      const recognition = FakeSpeechRecognition.last();

      act(() => recognition.say("already said"));
      act(() => recognition.onerror?.());

      expect(texts(result.current.words)).toEqual(["already", "said"]);
    });

    it("stops the recogniser and clears the line on unmount", () => {
      const { result, unmount } = render({ engine: "browser" });
      const recognition = FakeSpeechRecognition.last();

      act(() => recognition.say("goodbye"));
      expect(result.current.words).toHaveLength(1);

      unmount();

      expect(recognition.stopCount).toBe(1);
      // A restart queued by the disposed session must not resurrect it.
      act(() => recognition.onend?.());
      expect(recognition.startCount).toBe(1);
    });
  });

  describe("level meter", () => {
    function stubFrames() {
      const frames: FrameRequestCallback[] = [];
      const cancel = vi.fn();
      vi.stubGlobal(
        "requestAnimationFrame",
        (callback: FrameRequestCallback) => {
          frames.push(callback);
          return frames.length;
        },
      );
      vi.stubGlobal("cancelAnimationFrame", cancel);
      return { frames, cancel };
    }

    beforeEach(() => {
      vi.stubGlobal("SpeechRecognition", undefined);
      vi.stubGlobal("webkitSpeechRecognition", undefined);
    });

    it("tracks the loudest sample in the frame", () => {
      const { frames } = stubFrames();
      const { result } = render({ engine: "browser" });

      const analyser = FakeAudioContext.last().analyser!;
      expect(analyser.fftSize).toBe(256);
      expect(result.current.level).toBe(0);

      analyser.amplitude = 32;
      act(() => frames.at(-1)!(0));
      expect(result.current.level).toBe(0.5);

      // Anything past the 64 reference stays pinned at full scale.
      analyser.amplitude = 120;
      act(() => frames.at(-1)!(0));
      expect(result.current.level).toBe(1);
    });

    it("does not open an audio context without a stream", () => {
      stubFrames();
      render({ engine: "browser", audioStream: null });

      expect(FakeAudioContext.instances).toHaveLength(0);
    });

    it("cancels the frame loop and closes the context on unmount", () => {
      const { frames, cancel } = stubFrames();
      const { unmount } = render({ engine: "browser" });
      const context = FakeAudioContext.last();

      act(() => frames.at(-1)!(0));
      unmount();

      expect(cancel).toHaveBeenCalledWith(frames.length);
      expect(context.closeCount).toBe(1);
    });
  });
});

describe("DEFAULT_CAPTIONS_ENGINE", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.resetModules();
  });

  it.each([
    ["deepgram", "deepgram"],
    ["browser", "browser"],
    ["something-else", "elevenlabs"],
    [undefined, "elevenlabs"],
  ])("resolves %s to %s", async (configured, expected) => {
    vi.stubEnv("NEXT_PUBLIC_LIVE_CAPTIONS_ENGINE", configured);
    vi.resetModules();

    const module = await import("../useLiveCaptions");

    expect(module.DEFAULT_CAPTIONS_ENGINE).toBe(expected);
  });
});
