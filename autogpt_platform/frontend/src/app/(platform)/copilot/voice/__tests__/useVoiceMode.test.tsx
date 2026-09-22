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

let transcribe: (audio: Blob) => Promise<string> = async () => transcript;
/** Every blob handed to the transcriber, so a retry can be proved identical. */
const transcribed: Blob[] = [];

vi.mock("../speechApi", () => ({
  synthesizeSpeech: vi.fn(async (text: string) => {
    spoken.push(text);
    return new Blob([text]);
  }),
  transcribeUtterance: vi.fn((audio: Blob) => {
    transcribed.push(audio);
    return transcribe(audio);
  }),
}));

const downloaded: Blob[] = [];
vi.mock("../downloadRecording", () => ({
  downloadRecording: (blob: Blob) => downloaded.push(blob),
}));

const clicks: string[] = [];
vi.mock("../clickSound", () => ({
  playClickSound: () => clicks.push("play"),
}));

vi.mock("../audioContext", () => ({
  primeAudioContext: () => clicks.push("prime"),
  runningAudioContext: () => null,
}));

// Hoisted: `vi.mock` factories run before module-level consts exist.
const { toast } = vi.hoisted(() => ({ toast: vi.fn() }));
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toast }),
}));

const tracked: [string, Record<string, unknown> | undefined][] = [];
vi.mock("@/services/copilot/voice-mode-analytics", () => ({
  trackVoiceMode: (event: string, props?: Record<string, unknown>) =>
    tracked.push([event, props]),
}));

import { synthesizeSpeech } from "../speechApi";
import {
  isVoiceTurn,
  requestVoiceStart,
  setVoiceTurnActive,
  takeVoiceStart,
} from "../pendingVoiceStart";
import { useVoiceMode } from "../useVoiceMode";

describe("useVoiceMode", () => {
  beforeEach(() => {
    tracked.length = 0;
    setVoiceTurnActive(false);
    takeVoiceStart();
    spoken.length = 0;
    clicks.length = 0;
    toast.mockClear();
    // A rejection set by one test would otherwise outlive it: restoreAllMocks
    // does not reach an implementation set on a module mock.
    vi.mocked(synthesizeSpeech).mockImplementation(async (text: string) => {
      spoken.push(text);
      return new Blob([text]);
    });
    sessions.length = 0;
    vadLoad = Promise.resolve();
    transcribed.length = 0;
    downloaded.length = 0;
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

  afterEach(() => {
    // Timers first: a fake-timer test that fails before its own cleanup
    // would otherwise leak them into every test after it.
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

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

  it("clicks the moment the user stops, before the transcript arrives", async () => {
    const view = render({});
    await enable(view);
    await speak();

    expect(clicks).toContain("play");
  });

  it("buys no speech while the model thinks", async () => {
    // The model's first token is a median 13.9 s out. Nothing may fill that
    // gap: a second cue reads as a glitch, and synthesis is real money.
    vi.useFakeTimers();
    const view = render({});
    await enable(view);
    await speak();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(20_000);
    });

    expect(spoken).toHaveLength(0);
    expect(clicks.filter((c) => c === "play")).toHaveLength(1);
    vi.useRealTimers();
  });

  it("speaks a turn replayed under a fresh id once", async () => {
    // A stream reconnect replays the running turn from its start as a new
    // assistant message. Keyed on the id, that read the whole turn again on
    // every reconnect — three times during one long agent build.
    const view = render({});
    await enable(view);
    await speak();
    await act(async () => {
      view.rerender({
        messages: assistant("First sentence. Second sentence. ", "a1"),
        isStreaming: true,
      });
    });
    await act(async () => {
      view.rerender({
        messages: assistant("First sentence. Second sentence. ", "a2"),
        isStreaming: true,
      });
    });
    await act(async () => {
      view.rerender({
        messages: assistant(
          "First sentence. Second sentence. Third sentence. ",
          "a2",
        ),
        isStreaming: true,
      });
    });
    await act(async () => {
      view.rerender({
        messages: assistant(
          "First sentence. Second sentence. Third sentence. ",
          "a2",
        ),
        isStreaming: false,
      });
    });
    await act(async () => undefined);

    expect(spoken).toEqual([
      "First sentence.",
      "Second sentence.",
      "Third sentence.",
    ]);
  });

  it("speaks a message restarted from empty under the same id once", async () => {
    const view = render({});
    await enable(view);
    await speak();
    await act(async () => {
      view.rerender({
        messages: assistant("Hello there. "),
        isStreaming: true,
      });
    });
    await act(async () => {
      view.rerender({ messages: assistant(""), isStreaming: true });
    });
    await act(async () => {
      view.rerender({ messages: assistant("Hello "), isStreaming: true });
    });
    await act(async () => {
      view.rerender({
        messages: assistant("Hello there. And again. "),
        isStreaming: true,
      });
    });
    await act(async () => {
      view.rerender({
        messages: assistant("Hello there. And again. "),
        isStreaming: false,
      });
    });
    await act(async () => undefined);

    expect(spoken).toEqual(["Hello there.", "And again."]);
  });

  it("keeps the mic shut while a closed stream is probed and reconnected", async () => {
    // Between the close and the resume the SDK reports "not streaming" —
    // through the finish probe and then the scheduled reconnect. Treating
    // that as the reply's end reopened the mic mid-turn and left the rest
    // of the turn unspoken.
    const view = render({});
    await enable(view);
    await speak();
    await act(async () => {
      view.rerender({
        messages: assistant("A long answer. "),
        isStreaming: true,
      });
    });

    await act(async () => {
      view.rerender({ isStreaming: false, isFinishProbing: true });
    });
    await act(async () => undefined);
    expect(view.result.current.state).not.toBe("listening");

    await act(async () => {
      view.rerender({ isFinishProbing: false, isReconnecting: true });
    });
    await act(async () => undefined);
    expect(view.result.current.state).not.toBe("listening");

    await act(async () => {
      view.rerender({
        messages: assistant("A long answer. Continued. ", "a2"),
        isStreaming: true,
        isReconnecting: false,
      });
    });
    await act(async () => {
      view.rerender({
        messages: assistant("A long answer. Continued. ", "a2"),
        isStreaming: false,
      });
    });
    await waitFor(() => expect(view.result.current.state).toBe("listening"));
    expect(spoken).toEqual(["A long answer.", "Continued."]);
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

  it("reports a synthesis outage once a turn, not once a sentence", async () => {
    // Every sentence is its own request, so whatever refuses the first
    // refuses them all — and 429 (over the usage cap) is exactly what the
    // spend pre-flight returns.
    vi.mocked(synthesizeSpeech).mockRejectedValue(
      new Error("You've reached your usage limit"),
    );
    const view = render({});
    await enable(view);
    await speak();

    await reply(view, "First sentence. Second sentence. Third sentence.");
    await act(async () => undefined);

    expect(toast).toHaveBeenCalledTimes(1);
    expect(toast.mock.calls[0][0].description).toContain("usage limit");
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

    await act(async () => view.result.current.toggle());

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

  it("destroys a mic session that finishes starting after the host unmounted", async () => {
    // Navigating away mid-start: nothing to destroy yet, so unless the start
    // is invalidated the session opens the mic on a page that is gone and
    // marks later text turns as voice turns.
    let finishLoading!: () => void;
    vadLoad = new Promise<void>((resolve) => (finishLoading = resolve));
    const view = render({});

    act(() => view.result.current.toggle());
    view.unmount();
    await act(async () => finishLoading());
    await act(async () => undefined);

    expect(sessions).toHaveLength(1);
    expect(sessions[0].destroy).toHaveBeenCalled();
    expect(isVoiceTurn()).toBe(false);
  });

  it("stays quiet when an abandoned mic start fails", async () => {
    // The page was left while the mic was still starting; the failure that
    // follows belongs to nobody, and a toast for it lands on the next page.
    let failLoading!: (error: Error) => void;
    vadLoad = new Promise<void>((_, reject) => (failLoading = reject));
    const view = render({});

    act(() => view.result.current.toggle());
    view.unmount();
    await act(async () => failLoading(new Error("mic went away")));
    await act(async () => undefined);

    expect(toast).not.toHaveBeenCalled();
    expect(tracked.map(([event]) => event)).not.toContain("voice_mode_error");
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

    await act(async () => view.result.current.toggle());
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

  it("reports a completed turn to the funnel", async () => {
    const view = render({});
    await enable(view);
    await speak();
    await reply(view, "On it. Building that now.");
    await waitFor(() => expect(view.result.current.state).toBe("listening"));

    const events = tracked.map(([e]) => e);
    expect(events).toContain("voice_mode_started");
    expect(events).toContain("voice_transcribe_latency_ms");
    expect(events).toContain("voice_turn_sent");
    expect(events).toContain("voice_turn_completed");
    // Without this the funnel cannot tell one turn from ten.
    const sent = tracked.find(([e]) => e === "voice_turn_sent");
    expect(sent?.[1]?.turn_index).toBe(1);
  });

  it("distinguishes a silence timeout from the user leaving", async () => {
    vi.useFakeTimers();
    const view = render({ silenceTimeoutMs: 8000 });
    await enable(view);
    await act(async () => {
      vi.advanceTimersByTime(8001);
    });

    expect(tracked.map(([e]) => e)).toContain("voice_mode_timed_out");
    expect(tracked.map(([e]) => e)).not.toContain("voice_mode_stopped");
    vi.useRealTimers();
  });

  it("reports why an utterance was thrown away", async () => {
    transcript = "Uh, um...";
    const view = render({});
    await enable(view);
    await speak();

    const dropped = tracked.find(([e]) => e === "voice_turn_dropped");
    expect(dropped?.[1]?.reason).toBe("filler_or_empty");
  });

  it("marks turns as voice turns only while voice mode is on", async () => {
    // The transport reads this to ask the reply to speak before it works.
    expect(isVoiceTurn()).toBe(false);
    const view = render({});
    await enable(view);
    expect(isVoiceTurn()).toBe(true);

    await act(async () => view.result.current.toggle());
    expect(isVoiceTurn()).toBe(false);
  });

  it("speaks the answer that follows a tool round", async () => {
    // Tool rounds start a new assistant message. Treating that as a rewrite
    // of the first left the real answer silent.
    const view = render({});
    await enable(view);
    await speak();

    await act(async () => {
      view.rerender({
        messages: assistant("Let me check that.", "a1"),
        isStreaming: true,
      });
    });
    // Trailing space because a live stream keeps going; a terminator at the
    // very end of the buffer is deliberately held back.
    await act(async () => {
      view.rerender({
        messages: assistant("Here is what I found. ", "a2"),
        isStreaming: true,
      });
    });

    // The pre-tool line and the answer both get said, in order, and are not
    // run together into one utterance.
    expect(spoken.slice(-2)).toEqual([
      "Let me check that.",
      "Here is what I found.",
    ]);
  });

  it("does not give the mic back while a long tool chain is still running", async () => {
    vi.useFakeTimers();
    const view = render({});
    await enable(view);
    await speak();

    // A tool chain that outlasts the old 45s watchdog, showing signs of life.
    for (let i = 0; i < 4; i++) {
      await act(async () => {
        await vi.advanceTimersByTimeAsync(30_000);
      });
      await act(async () => {
        view.rerender({
          messages: assistant(`Working... ${i}`, "a1"),
          isStreaming: true,
        });
      });
    }

    expect(view.result.current.state).not.toBe("listening");
    vi.useRealTimers();
  });

  it("speaks the pre-tool line without waiting for the tool chain", async () => {
    // The acknowledgement and the tool calls share one message, so there is
    // no message boundary to flush on. Holding it means silence for as long
    // as the tools run — which was 27-36s in practice.
    vi.useFakeTimers();
    const view = render({});
    await enable(view);
    await speak();

    await act(async () => {
      view.rerender({
        messages: assistant("Great question — let me look that up.", "a1"),
        isStreaming: true,
      });
    });
    const beforeWait = spoken.length;

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1200);
    });

    expect(spoken.slice(beforeWait)).toEqual([
      "Great question — let me look that up.",
    ]);
    vi.useRealTimers();
  });

  it("keeps the recording when transcription fails", async () => {
    // The whole bug: a transient 500 used to drop the audio on the floor and
    // leave the user with nothing but a toast.
    const onSend = vi.fn();
    transcribe = async () => {
      throw new Error("Transcription failed");
    };
    const view = render({ onSend });

    await enable(view);
    await speak();

    expect(onSend).not.toHaveBeenCalled();
    expect(view.result.current.failure).toEqual({
      message: "Transcription failed",
    });
    // The mic comes back, as before — the failure is offered, not forced.
    expect(view.result.current.state).toBe("listening");
  });

  it("falls back to a readable message when the failure is not an Error", async () => {
    transcribe = async () => {
      throw "the network went away";
    };
    const view = render({});

    await enable(view);
    await speak();

    expect(view.result.current.failure).toEqual({
      message: "Transcription failed",
    });
  });

  it("retries the same recording, byte for byte, and finishes the turn", async () => {
    const onSend = vi.fn();
    transcribe = async () => {
      throw new Error("Transcription failed");
    };
    const view = render({ onSend });

    await enable(view);
    await speak();
    expect(transcribed).toHaveLength(1);

    transcribe = async () => transcript;
    await act(async () => view.result.current.retryFailedUtterance());

    expect(transcribed).toHaveLength(2);
    expect(transcribed[1]).toBe(transcribed[0]);
    await waitFor(() => expect(onSend).toHaveBeenCalledWith(transcript));
    expect(view.result.current.failure).toBeNull();
    expect(view.result.current.state).toBe("thinking");
  });

  it("still has the recording after a retry fails too", async () => {
    transcribe = async () => {
      throw new Error("Transcription failed");
    };
    const view = render({});

    await enable(view);
    await speak();
    await act(async () => view.result.current.retryFailedUtterance());

    expect(transcribed).toHaveLength(2);
    expect(view.result.current.failure).not.toBeNull();
    expect(view.result.current.state).toBe("listening");

    await act(async () => view.result.current.downloadFailedUtterance());
    expect(downloaded).toEqual([transcribed[0]]);
  });

  it("does not close the session out from under the error", async () => {
    // The mic closes after 8s of silence. Reading an error and deciding takes
    // longer than that, and closing takes the recording with it.
    vi.useFakeTimers();
    transcribe = async () => {
      throw new Error("Transcription failed");
    };
    const view = render({ silenceTimeoutMs: 8_000 });

    await enable(view);
    await speak();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(30_000);
    });

    expect(view.result.current.state).toBe("listening");
    expect(view.result.current.failure).not.toBeNull();

    // It is held open, not held open forever: the session cap still ends it.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5 * 60 * 1000);
    });
    expect(view.result.current.state).toBe("off");
    vi.useRealTimers();
  });

  it("drops a failure that arrives after the user left voice mode", async () => {
    // Stop while the transcript is in flight, then let it reject. Writing the
    // failure back from a dead turn used to resurrect the old audio under the
    // next activation, with a Retry that would send it.
    let failTranscribing!: (error: Error) => void;
    transcribe = () => new Promise((_, reject) => (failTranscribing = reject));
    const view = render({});

    await enable(view);
    await act(async () => vad.onSpeechStart());
    await act(async () => vad.onSpeechEnd(new Blob(["wav"])));
    expect(view.result.current.state).toBe("transcribing");

    await act(async () => view.result.current.toggle());
    expect(view.result.current.state).toBe("off");
    await act(async () => failTranscribing(new Error("Transcription failed")));
    await act(async () => undefined);

    expect(view.result.current.failure).toBeNull();

    // Restarting gets a clean session, not the last one's error and audio.
    await enable(view);
    expect(view.result.current.state).toBe("listening");
    expect(view.result.current.failure).toBeNull();
    await act(async () => view.result.current.downloadFailedUtterance());
    expect(downloaded).toHaveLength(0);

    // The dead turn is not counted against the new session either.
    expect(
      tracked.filter(([event]) => event === "voice_turn_dropped"),
    ).toHaveLength(0);
  });

  it("forgets the failed recording once the user speaks again", async () => {
    // Otherwise the stale row sits over a live mic, and Retry sends audio the
    // user has already moved on from.
    transcribe = async () => {
      throw new Error("Transcription failed");
    };
    const view = render({});

    await enable(view);
    await speak();
    expect(view.result.current.failure).not.toBeNull();

    await act(async () => vad.onSpeechStart());

    expect(view.result.current.failure).toBeNull();
    await act(async () => view.result.current.downloadFailedUtterance());
    expect(downloaded).toHaveLength(0);
  });

  it("does not retry over an utterance already in flight", async () => {
    transcribe = async () => {
      throw new Error("Transcription failed");
    };
    const onSend = vi.fn();
    const view = render({ onSend });

    await enable(view);
    await speak();
    await act(async () => vad.onSpeechStart());

    await act(async () => view.result.current.retryFailedUtterance());

    expect(transcribed).toHaveLength(1);
    expect(view.result.current.state).toBe("hearing");
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
    unmount: view.unmount,
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

function assistant(text: string, id = "a1"): UIMessage[] {
  return [
    {
      id,
      role: "assistant",
      parts: [{ type: "text", text }],
    } as UIMessage,
  ];
}
