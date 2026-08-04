import { act, render, screen, waitFor } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LiveCaptions } from "../LiveCaptions";
import {
  FakeAudioContext,
  FakeSpeechRecognition,
  FakeWebSocket,
  fakeStream,
  stubTokenFetch,
} from "./captionsTestDoubles";

const stream = fakeStream();

// happy-dom has no layout, so every width is 0 and the marquee's "how much
// did the line grow" maths short-circuits. Give words a width so the growth
// branch actually runs while the captions are asserted.
const WORD_WIDTH = 40;

function stubLayout() {
  const original = {
    scrollWidth: Object.getOwnPropertyDescriptor(
      HTMLElement.prototype,
      "scrollWidth",
    ),
    offsetWidth: Object.getOwnPropertyDescriptor(
      HTMLElement.prototype,
      "offsetWidth",
    ),
  };
  Object.defineProperty(HTMLElement.prototype, "offsetWidth", {
    configurable: true,
    get() {
      return WORD_WIDTH;
    },
  });
  Object.defineProperty(HTMLElement.prototype, "scrollWidth", {
    configurable: true,
    get(this: HTMLElement) {
      return this.children.length * WORD_WIDTH;
    },
  });
  return function restore() {
    for (const [name, descriptor] of Object.entries(original)) {
      if (descriptor)
        Object.defineProperty(HTMLElement.prototype, name, descriptor);
      else Reflect.deleteProperty(HTMLElement.prototype, name);
    }
  };
}

function bars() {
  return Array.from(screen.getByTestId("brain-dump-level-meter").children);
}

describe("LiveCaptions", () => {
  let restoreLayout: () => void;

  beforeEach(() => {
    FakeSpeechRecognition.reset();
    FakeAudioContext.reset();
    restoreLayout = stubLayout();
    vi.stubGlobal("AudioContext", FakeAudioContext);
  });

  afterEach(() => {
    restoreLayout();
    vi.unstubAllGlobals();
  });

  it("shows nothing until the take starts", () => {
    vi.stubGlobal("SpeechRecognition", FakeSpeechRecognition);

    const { container } = render(
      <LiveCaptions
        isRecording={false}
        audioStream={stream}
        engine="browser"
      />,
    );

    expect(container.innerHTML).toBe("");
    expect(FakeSpeechRecognition.instances).toHaveLength(0);
  });

  it("renders the words the recogniser hears", () => {
    vi.stubGlobal("SpeechRecognition", FakeSpeechRecognition);

    render(<LiveCaptions isRecording audioStream={stream} engine="browser" />);

    act(() => FakeSpeechRecognition.last().say("build me a bot"));

    for (const word of ["build", "me", "a", "bot"]) {
      expect(screen.getByText(word)).toBeDefined();
    }
    expect(screen.queryByTestId("brain-dump-level-meter")).toBeNull();
  });

  // Stable ids exist so a word being revised is not torn down and
  // re-mounted, which would replay its fade-in on every interim result.
  it("keeps the same element for a word whose text has not changed", () => {
    vi.stubGlobal("SpeechRecognition", FakeSpeechRecognition);

    render(<LiveCaptions isRecording audioStream={stream} engine="browser" />);
    const recognition = FakeSpeechRecognition.last();

    act(() => recognition.say("hello wor"));
    const firstWord = screen.getByText("hello");

    act(() => recognition.say("hello world"));

    expect(screen.getByText("hello")).toBe(firstWord);
    expect(screen.getByText("world")).toBeDefined();
  });

  it("never shows more than the last 24 words", () => {
    vi.stubGlobal("SpeechRecognition", FakeSpeechRecognition);

    const { container } = render(
      <LiveCaptions isRecording audioStream={stream} engine="browser" />,
    );
    const spoken = Array.from({ length: 30 }, (_, index) => `w${index}`);

    act(() => FakeSpeechRecognition.last().say(spoken.join(" ")));

    expect(container.querySelectorAll("[data-word-id]")).toHaveLength(24);
    expect(screen.queryByText("w5")).toBeNull();
    expect(screen.getByText("w29")).toBeDefined();
  });

  // Production never passes `engine`, so the default — ElevenLabs Scribe —
  // is the configuration that actually ships. Every other test here pins
  // the browser engine, which nothing renders.
  describe("on the shipped default engine", () => {
    beforeEach(() => {
      FakeWebSocket.reset();
      stubTokenFetch();
      vi.stubGlobal("WebSocket", FakeWebSocket);
      vi.stubGlobal("SpeechRecognition", FakeSpeechRecognition);
    });

    it("renders the words the cloud engine sends", async () => {
      render(<LiveCaptions isRecording audioStream={stream} />);

      await waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1));
      const socket = FakeWebSocket.last();
      act(() => socket.open());
      act(() =>
        socket.emit({ message_type: "partial_transcript", text: "build a bot" }),
      );

      for (const word of ["build", "a", "bot"]) {
        expect(screen.getByText(word)).toBeDefined();
      }
      // The browser recogniser stays out of the way while the cloud one
      // is working.
      expect(FakeSpeechRecognition.instances).toHaveLength(0);
    });

    it("swaps in the browser recogniser when the socket dies", async () => {
      render(<LiveCaptions isRecording audioStream={stream} />);

      await waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1));
      const socket = FakeWebSocket.last();
      act(() => socket.open());
      act(() => socket.onclose?.());

      await waitFor(() =>
        expect(FakeSpeechRecognition.instances).toHaveLength(1),
      );
      act(() => FakeSpeechRecognition.last().say("still listening"));

      expect(screen.getByText("still")).toBeDefined();
      expect(screen.getByText("listening")).toBeDefined();
    });
  });

  describe("without a speech recogniser", () => {
    const frames: FrameRequestCallback[] = [];

    beforeEach(() => {
      frames.length = 0;
      vi.stubGlobal("SpeechRecognition", undefined);
      vi.stubGlobal("webkitSpeechRecognition", undefined);
      vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => {
        frames.push(cb);
        return frames.length;
      });
      vi.stubGlobal("cancelAnimationFrame", vi.fn());
    });

    // Firefox and some Safari builds have no SpeechRecognition. The level
    // meter carries the same "we can hear you" signal without pretending
    // to transcribe.
    it("falls back to a level meter that follows the mic", () => {
      render(
        <LiveCaptions isRecording audioStream={stream} engine="browser" />,
      );

      const silent = bars().map((bar) => (bar as HTMLElement).style.height);
      expect(silent).toEqual(Array(7).fill("8px"));

      FakeAudioContext.last().analyser!.amplitude = 64;
      act(() => frames.at(-1)!(0));

      const loud = bars().map((bar) => (bar as HTMLElement).style.height);
      // Every third bar swings the full 24px; the rest are damped to 60%.
      expect(loud[0]).toBe("32px");
      expect(loud[1]).toBe("22.4px");
      expect(loud[3]).toBe("32px");
    });

    it("shows the meter instead of a caption line", () => {
      const { container } = render(
        <LiveCaptions isRecording audioStream={stream} engine="browser" />,
      );

      expect(screen.getByTestId("brain-dump-level-meter")).toBeDefined();
      expect(container.querySelectorAll("[data-word-id]")).toHaveLength(0);
    });
  });
});
