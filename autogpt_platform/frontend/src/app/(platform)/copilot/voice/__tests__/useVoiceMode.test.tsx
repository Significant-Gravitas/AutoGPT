import { act, renderHook, waitFor } from "@testing-library/react";
import type { UIMessage } from "ai";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const vad = {
  onSpeechStart: () => undefined as void,
  onSpeechEnd: (_wav: Blob) => undefined as void,
  onMisfire: () => undefined as void,
  pause: vi.fn(),
  resume: vi.fn(),
};

/** Every session the hook has started, so a leaked one is visible. */
const sessions: { destroy: ReturnType<typeof vi.fn> }[] = [];
/** Stalls `startVadSession` the way the real model download does. */
let vadLoad: Promise<void> = Promise.resolve();

vi.mock("../vadSession", () => ({
  startVadSession: vi.fn(async (callbacks) => {
    await vadLoad;
    Object.assign(vad, callbacks);
    const session = {
      pause: vad.pause,
      resume: vad.resume,
      destroy: vi.fn(async () => undefined),
    };
    sessions.push(session);
    return session;
  }),
}));

const spoken: string[] = [];
let transcript = "Build me a Slack agent";

let transcribe: () => Promise<string> = async () => transcript;

vi.mock("../speechApi", () => ({
  synthesizeSpeech: vi.fn(async (text: string) => {
    spoken.push(text);
    return new Blob([text]);
  }),
  transcribeUtterance: vi.fn(() => transcribe()),
}));

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toast: vi.fn() }),
}));

import { requestVoiceStart, takeVoiceStart } from "../pendingVoiceStart";
import { useVoiceMode } from "../useVoiceMode";

describe("useVoiceMode", () => {
  beforeEach(() => {
    takeVoiceStart();
    spoken.length = 0;
    sessions.length = 0;
    vadLoad = Promise.resolve();
    transcribe = async () => transcript;
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

  it("speaks exactly one acknowledgement, however long the reply takes", async () => {
    vi.useFakeTimers();
    const view = render({});
    await enable(view);
    await speak();
    expect(spoken).toHaveLength(1);

    // The model's first token is a median 13.9 s out. Nothing may fill that
    // gap with a second phrase — two in a row read as a glitch.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(20_000);
    });

    expect(spoken).toHaveLength(1);
    vi.useRealTimers();
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

  it("leaves voice mode when the reply is stopped", async () => {
    const view = render({});
    await enable(view);
    await speak();
    await act(async () => {
      view.rerender({
        messages: assistant("A long answer. "),
        isStreaming: true,
      });
    });

    await act(async () => view.result.current.stop());

    // Stop means "I am done", not "skip this bit" — the mic closes with it.
    expect(view.result.current.state).toBe("off");
    expect(sessions[0].destroy).toHaveBeenCalled();
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

  it("starts one mic session however fast the button is clicked", async () => {
    let finishLoading!: () => void;
    vadLoad = new Promise<void>((resolve) => (finishLoading = resolve));
    const view = render({});

    act(() => view.result.current.toggle());
    act(() => view.result.current.toggle());
    await act(async () => finishLoading());
    await act(async () => undefined);

    // Whichever way the second click is read, no session may outlive the UI.
    const live = sessions.filter((s) => s.destroy.mock.calls.length === 0);
    expect(live.length).toBeLessThanOrEqual(1);
    if (view.result.current.state === "off") expect(live).toHaveLength(0);
  });

  it("destroys a mic session that finishes starting after the user left", async () => {
    let finishLoading!: () => void;
    vadLoad = new Promise<void>((resolve) => (finishLoading = resolve));
    const onSend = vi.fn();
    const view = render({ onSend });

    act(() => view.result.current.toggle());
    act(() => view.result.current.toggle());
    await act(async () => finishLoading());
    await act(async () => undefined);

    expect(view.result.current.state).toBe("off");
    expect(sessions).toHaveLength(1);
    expect(sessions[0].destroy).toHaveBeenCalled();
  });

  it("does not send a transcript for an utterance the user opted out of", async () => {
    vi.useFakeTimers();
    let finishTranscribing!: (text: string) => void;
    transcribe = () => new Promise((resolve) => (finishTranscribing = resolve));
    const onSend = vi.fn();
    const view = render({ onSend });

    await enable(view);
    await act(async () => vad.onSpeechStart());
    await act(async () => vad.onSpeechEnd(new Blob(["wav"])));
    expect(view.result.current.state).toBe("transcribing");

    await act(async () => view.result.current.toggle());
    expect(view.result.current.state).toBe("off");
    await act(async () => finishTranscribing("Delete all my agents"));
    await act(async () => undefined);

    expect(onSend).not.toHaveBeenCalled();
    const spokenAfterOff = spoken.length;
    await act(async () => {
      await vi.advanceTimersByTimeAsync(7000);
    });
    expect(spoken).toHaveLength(spokenAfterOff);
    vi.useRealTimers();
  });

  it("never speaks the leftover of a reply the user stopped", async () => {
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

    await act(async () => view.result.current.stop());
    const spokenAfterStop = spoken.length;
    await act(async () => {
      view.rerender({
        messages: assistant("First sentence. Second half"),
        isStreaming: false,
      });
    });
    await act(async () => undefined);

    expect(spoken).toHaveLength(spokenAfterStop);
    expect(view.result.current.state).toBe("off");
  });

  it("starts itself on the mount that follows creating the chat", async () => {
    // The empty composer asks for voice mode, then the session is created and
    // this whole subtree is re-keyed and remounted with the new id.
    requestVoiceStart();
    const view = render({ sessionId: "session-created-just-now" });
    await act(async () => undefined);

    expect(view.result.current.state).toBe("listening");
    expect(sessions).toHaveLength(1);
  });

  it("does not start itself on an ordinary mount", async () => {
    const view = render({});
    await act(async () => undefined);

    expect(view.result.current.state).toBe("off");
    expect(sessions).toHaveLength(0);
  });

  it("only honours the request once", async () => {
    requestVoiceStart();
    const first = render({});
    await act(async () => undefined);
    await act(async () => first.result.current.toggle());

    const second = render({});
    await act(async () => undefined);
    expect(second.result.current.state).toBe("off");
  });

  it("shuts itself down when the flag goes off", async () => {
    const view = render({});
    await enable(view);

    await act(async () => view.rerender({ enabled: false }));

    expect(view.result.current.state).toBe("off");
    expect(sessions[0].destroy).toHaveBeenCalled();
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
