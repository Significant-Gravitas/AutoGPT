import { act, renderHook, waitFor } from "@testing-library/react";
import type { UIMessage } from "ai";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const vad = {
  onSpeechStart: () => undefined as void,
  onSpeechEnd: (_wav: Blob) => undefined as void,
  onMisfire: () => undefined as void,
  pause: vi.fn(),
  resume: vi.fn(),
  destroy: vi.fn(async () => undefined),
};

vi.mock("../vadSession", () => ({
  startVadSession: vi.fn(async (callbacks) => {
    Object.assign(vad, callbacks);
    return { pause: vad.pause, resume: vad.resume, destroy: vad.destroy };
  }),
}));

const spoken: string[] = [];
let transcript = "Build me a Slack agent";

vi.mock("../speechApi", () => ({
  synthesizeSpeech: vi.fn(async (text: string) => {
    spoken.push(text);
    return new Blob([text]);
  }),
  transcribeUtterance: vi.fn(async () => transcript),
}));

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toast: vi.fn() }),
}));

import { useVoiceMode } from "../useVoiceMode";

describe("useVoiceMode", () => {
  beforeEach(() => {
    spoken.length = 0;
    transcript = "Build me a Slack agent";
    vi.spyOn(window.HTMLMediaElement.prototype, "play").mockImplementation(
      function (this: HTMLAudioElement) {
        queueMicrotask(() => this.dispatchEvent(new Event("ended")));
        return Promise.resolve();
      },
    );
    global.URL.createObjectURL = vi.fn(() => "blob:chunk");
    global.URL.revokeObjectURL = vi.fn();
  });

  afterEach(() => vi.restoreAllMocks());

  it("runs a whole turn and gives the mic back", async () => {
    const onSend = vi.fn();
    const view = render({ onSend });

    await enable(view);
    expect(view.result.current.state).toBe("listening");

    await speak();
    await waitFor(() => expect(onSend).toHaveBeenCalledWith(transcript));
    expect(view.result.current.state).toBe("thinking");
    // The mic is shut the moment the user stops talking, not when the reply
    // starts — that is what keeps the assistant out of its own transcript.
    expect(vad.pause).toHaveBeenCalled();

    await reply(view, "On it. Building that now.");
    await waitFor(() => expect(view.result.current.state).toBe("listening"));
    expect(vad.resume).toHaveBeenCalled();
  });

  it("acknowledges before the transcript has even arrived", async () => {
    const view = render({});
    await enable(view);
    await speak();

    expect(spoken.length).toBeGreaterThan(0);
    expect(spoken[0]).not.toContain("Slack");
  });

  it("speaks the reply one sentence at a time as it streams", async () => {
    const view = render({});
    await enable(view);
    await speak();

    await act(async () => {
      view.rerender({
        messages: assistant("First sentence. Second half"),
        isStreaming: true,
      });
    });

    expect(spoken).toContain("First sentence.");
    expect(spoken).not.toContain("Second half");
  });

  it("never reads fenced code aloud", async () => {
    const view = render({});
    await enable(view);
    await speak();

    await reply(view, "Here it is.\n```python\nprint(1)\n```\nDone.");
    expect(spoken).toContain("Done.");
    expect(spoken.some((chunk) => chunk.includes("print"))).toBe(false);
  });

  it("drops a filler-only transcript without sending a turn", async () => {
    transcript = "Uh, um...";
    const onSend = vi.fn();
    const view = render({ onSend });

    await enable(view);
    await speak();

    expect(onSend).not.toHaveBeenCalled();
    await waitFor(() => expect(view.result.current.state).toBe("listening"));
  });

  it("returns to listening when the VAD misfires", async () => {
    const view = render({});
    await enable(view);

    await act(async () => vad.onSpeechStart());
    expect(view.result.current.state).toBe("hearing");
    await act(async () => vad.onMisfire());

    expect(view.result.current.state).toBe("listening");
  });

  it("stops playback and listens again when interrupted", async () => {
    const view = render({});
    await enable(view);
    await speak();
    await act(async () => {
      view.rerender({
        messages: assistant("A long answer. "),
        isStreaming: true,
      });
    });

    await act(async () => view.result.current.interrupt());

    expect(view.result.current.state).toBe("listening");
  });

  it("closes the mic after the silence timeout", async () => {
    vi.useFakeTimers();
    const view = render({ silenceTimeoutMs: 8000 });
    await enable(view);

    await act(async () => {
      vi.advanceTimersByTime(8001);
    });

    expect(view.result.current.state).toBe("off");
    expect(view.result.current.isActive).toBe(false);
    vi.useRealTimers();
  });

  it("shuts itself down when the flag goes off", async () => {
    const view = render({});
    await enable(view);

    await act(async () => view.rerender({ enabled: false }));

    expect(view.result.current.state).toBe("off");
    expect(vad.destroy).toHaveBeenCalled();
  });
});

type Props = Partial<Parameters<typeof useVoiceMode>[0]>;

function render(overrides: Props) {
  let props: Parameters<typeof useVoiceMode>[0] = {
    enabled: true,
    messages: [],
    isStreaming: false,
    sessionId: "session-1",
    onSend: vi.fn(),
    ...overrides,
  };
  const view = renderHook(() => useVoiceMode(props));
  return {
    get result() {
      return view.result;
    },
    rerender(next: Props) {
      props = { ...props, ...next };
      view.rerender();
    },
  };
}

async function enable(view: ReturnType<typeof render>) {
  await act(async () => view.result.current.toggle());
}

/** One utterance: speech starts, then ends, then the transcript resolves. */
async function speak() {
  await act(async () => vad.onSpeechStart());
  await act(async () => {
    vad.onSpeechEnd(new Blob(["wav"]));
  });
  await act(async () => undefined);
}

/** Stream a whole reply in, then end the stream. */
async function reply(view: ReturnType<typeof render>, text: string) {
  await act(async () => {
    view.rerender({ messages: assistant(text), isStreaming: true });
  });
  await act(async () => {
    view.rerender({ messages: assistant(text), isStreaming: false });
  });
}

function assistant(text: string): UIMessage[] {
  return [
    {
      id: "a1",
      role: "assistant",
      parts: [{ type: "text", text }],
    } as UIMessage,
  ];
}
