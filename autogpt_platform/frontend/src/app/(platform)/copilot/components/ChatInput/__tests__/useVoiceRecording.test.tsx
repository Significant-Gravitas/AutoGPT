import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useVoiceRecording } from "../useVoiceRecording";

vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: { ONBOARDING_BRAIN_DUMP: "onboarding-brain-dump" },
  useGetFlag: () => false,
}));

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toast: vi.fn() }),
}));

const downloaded: Blob[] = [];
vi.mock("../../../voice/downloadRecording", () => ({
  downloadRecording: (blob: Blob) => downloaded.push(blob),
}));

const getUserMedia = vi.fn();
const recorderStop = vi.fn();
/** Only the transcription tests want a real stop; the rest assert on the call. */
let firesOnStop = false;

function keydown(key: string, isComposing = false) {
  return {
    key,
    nativeEvent: { key, isComposing },
    preventDefault: vi.fn(),
  } as unknown as React.KeyboardEvent<HTMLTextAreaElement>;
}

function spaceKeydown(isComposing = false) {
  return keydown(" ", isComposing);
}

// Enough of the MediaRecorder surface for startRecording to reach the
// isRecordingRef.current = true branch. `stop` deliberately never fires
// `onstop`, so no transcription request is made.
class FakeMediaRecorder {
  static isTypeSupported = () => true;
  mimeType = "audio/webm";
  ondataavailable: ((event: unknown) => void) | null = null;
  onstop: (() => void) | null = null;
  start = vi.fn();
  stop = () => {
    recorderStop();
    if (!firesOnStop) return;
    this.ondataavailable?.({
      data: new Blob([RECORDED], { type: "audio/webm" }),
    });
    this.onstop?.();
  };
}

/** Unique, so a re-sent recording is provably the recorded one. */
const RECORDED = "the-one-recording";

async function renderRecording() {
  getUserMedia.mockResolvedValue({ getTracks: () => [] });
  const { result } = renderVoice();

  await act(async () => {
    result.current.handleKeyDown(spaceKeydown());
  });
  await waitFor(() => expect(result.current.isRecording).toBe(true));

  return result;
}

/** Shared so a test can assert the transcript reached the composer. */
const setValue = vi.fn();

function renderVoice(value = "") {
  return renderHook(() => useVoiceRecording({ setValue, value }));
}

beforeEach(() => {
  firesOnStop = false;
  downloaded.length = 0;
  getUserMedia.mockRejectedValue(new Error("no microphone"));
  vi.stubGlobal("MediaRecorder", FakeMediaRecorder);
  Object.defineProperty(navigator, "mediaDevices", {
    configurable: true,
    value: { getUserMedia },
  });
});

afterEach(() => {
  vi.clearAllMocks();
  vi.unstubAllGlobals();
});

describe("useVoiceRecording Space shortcut", () => {
  it("ignores Space while an IME is composing", () => {
    const { result } = renderVoice();
    const event = spaceKeydown(true);

    act(() => result.current.handleKeyDown(event));

    expect(event.preventDefault).not.toHaveBeenCalled();
    expect(getUserMedia).not.toHaveBeenCalled();
  });

  it("starts recording on a plain Space in an empty composer", async () => {
    const { result } = renderVoice();
    const event = spaceKeydown();

    act(() => result.current.handleKeyDown(event));

    expect(event.preventDefault).toHaveBeenCalled();
    await waitFor(() => expect(getUserMedia).toHaveBeenCalledTimes(1));
  });

  it("leaves Space alone once the composer has text", () => {
    const { result } = renderVoice("draft");
    const event = spaceKeydown();

    act(() => result.current.handleKeyDown(event));

    expect(event.preventDefault).not.toHaveBeenCalled();
    expect(getUserMedia).not.toHaveBeenCalled();
  });

  it("stops an in-progress recording on a plain Space", async () => {
    const result = await renderRecording();
    const event = spaceKeydown();

    act(() => result.current.handleKeyDown(event));

    expect(event.preventDefault).toHaveBeenCalled();
    expect(recorderStop).toHaveBeenCalledTimes(1);
    expect(result.current.isRecording).toBe(false);
  });

  it("does not stop a recording on a composing Space", async () => {
    const result = await renderRecording();
    const event = spaceKeydown(true);

    act(() => result.current.handleKeyDown(event));

    // Still swallowed by the while-recording block below, but the recording
    // itself survives — the Space belonged to the IME, not to the shortcut.
    expect(event.preventDefault).toHaveBeenCalled();
    expect(recorderStop).not.toHaveBeenCalled();
    expect(result.current.isRecording).toBe(true);
  });

  it("swallows every other key while recording", async () => {
    const result = await renderRecording();
    const event = keydown("Enter");

    act(() => result.current.handleKeyDown(event));

    expect(event.preventDefault).toHaveBeenCalled();
    expect(recorderStop).not.toHaveBeenCalled();
    expect(result.current.isRecording).toBe(true);
  });
});

describe("useVoiceRecording transcription failures", () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    firesOnStop = true;
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
    vi.spyOn(console, "error").mockImplementation(() => undefined);
  });

  it("keeps the recording and offers it back when transcription fails", async () => {
    failTranscription("Transcription service unavailable");
    const result = await recordAndStop();

    await waitFor(() =>
      expect(result.current.transcriptionError).toBe(
        "Transcription service unavailable",
      ),
    );
    // The whole point: the audio survives the failure.
    expect(result.current.hasFailedRecording).toBe(true);

    act(() => result.current.downloadFailedRecording());
    expect(downloaded).toHaveLength(1);
    expect(await downloaded[0].text()).toBe(RECORDED);
  });

  it("re-sends the same recording on retry, without recording again", async () => {
    failTranscription("Transcription failed");
    const result = await recordAndStop();
    await waitFor(() => expect(result.current.hasFailedRecording).toBe(true));

    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({ text: "the words I said" }),
    });
    await act(async () => result.current.retryTranscription());

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(await sentAudio(0)).toBe(RECORDED);
    expect(await sentAudio(1)).toBe(RECORDED);
    // One microphone session: the retry reused the audio, it did not re-record.
    expect(getUserMedia).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(result.current.transcriptionError).toBeNull());
    expect(result.current.hasFailedRecording).toBe(false);
  });

  it("still has the recording after a retry fails as well", async () => {
    failTranscription("Transcription failed");
    const result = await recordAndStop();
    await waitFor(() => expect(result.current.hasFailedRecording).toBe(true));

    await act(async () => result.current.retryTranscription());

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(result.current.transcriptionError).toBe("Transcription failed");
    expect(result.current.hasFailedRecording).toBe(true);
  });

  it("does not lose the recording to a denied microphone prompt", async () => {
    failTranscription("Transcription failed");
    const result = await recordAndStop();
    await waitFor(() => expect(result.current.hasFailedRecording).toBe(true));

    getUserMedia.mockRejectedValue(
      new DOMException("denied", "NotAllowedError"),
    );
    await act(async () => result.current.startRecording());

    expect(result.current.hasFailedRecording).toBe(true);
    expect(result.current.transcriptionError).toBe("Transcription failed");
  });

  it("does not lose the recording when the recorder itself refuses", async () => {
    // getUserMedia can resolve and the MediaRecorder still throw — an
    // unsupported mime type, a track that died. The catch has no way to give
    // the previous recording back, so it must not have been cleared yet.
    failTranscription("Transcription failed");
    const result = await recordAndStop();
    await waitFor(() => expect(result.current.hasFailedRecording).toBe(true));

    getUserMedia.mockResolvedValue({ getTracks: () => [] });
    vi.stubGlobal(
      "MediaRecorder",
      class {
        static isTypeSupported = () => false;
        constructor() {
          throw new DOMException("mime not supported", "NotSupportedError");
        }
      },
    );
    await act(async () => result.current.startRecording());

    expect(result.current.isRecording).toBe(false);
    expect(result.current.hasFailedRecording).toBe(true);
    expect(result.current.transcriptionError).toBe("Transcription failed");
  });

  it("supersedes the failed recording once a new one is under way", async () => {
    failTranscription("Transcription failed");
    const result = await recordAndStop();
    await waitFor(() => expect(result.current.hasFailedRecording).toBe(true));

    getUserMedia.mockResolvedValue({ getTracks: () => [] });
    await act(async () => result.current.startRecording());

    expect(result.current.hasFailedRecording).toBe(false);
    expect(result.current.transcriptionError).toBeNull();
  });

  it("falls back to a readable message when the failure is not an Error", async () => {
    // fetch itself rejecting — offline, DNS — does not always throw an Error,
    // and "undefined" is not a message anyone can act on.
    fetchMock.mockRejectedValue("network is down");
    const result = await recordAndStop();

    await waitFor(() =>
      expect(result.current.transcriptionError).toBe("Transcription failed"),
    );
    expect(result.current.hasFailedRecording).toBe(true);
  });

  it("does nothing when there is no recording to retry", async () => {
    const { result } = renderVoice();

    act(() => result.current.retryTranscription());

    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("drops the failed recording once the user dismisses it", async () => {
    failTranscription("Transcription failed");
    const result = await recordAndStop();
    await waitFor(() => expect(result.current.hasFailedRecording).toBe(true));

    act(() => result.current.dismissTranscriptionError());

    expect(result.current.transcriptionError).toBeNull();
    expect(result.current.hasFailedRecording).toBe(false);
    act(() => result.current.downloadFailedRecording());
    expect(downloaded).toHaveLength(0);
  });

  it("stays dismissed when the retry it was dismissed during then fails", async () => {
    failTranscription("Transcription failed");
    const result = await recordAndStop();
    await waitFor(() => expect(result.current.hasFailedRecording).toBe(true));

    let failRetry!: (error: Error) => void;
    fetchMock.mockReturnValue(new Promise((_, reject) => (failRetry = reject)));
    await act(async () => result.current.retryTranscription());
    act(() => result.current.dismissTranscriptionError());
    expect(result.current.transcriptionError).toBeNull();

    await act(async () => {
      failRetry(new Error("Transcription failed"));
    });

    // The row the user waved away does not come back a second later.
    expect(result.current.transcriptionError).toBeNull();
    expect(result.current.hasFailedRecording).toBe(false);
    expect(result.current.isTranscribing).toBe(false);
  });

  it("still delivers a transcript that lands after a dismissal", async () => {
    // Dismiss means "stop showing me this", not "throw away what I said".
    failTranscription("Transcription failed");
    const result = await recordAndStop();
    await waitFor(() => expect(result.current.hasFailedRecording).toBe(true));

    let finishRetry!: (value: unknown) => void;
    fetchMock.mockReturnValue(
      new Promise((resolve) => (finishRetry = resolve)),
    );
    await act(async () => result.current.retryTranscription());
    act(() => result.current.dismissTranscriptionError());

    await act(async () => {
      finishRetry({ ok: true, json: async () => ({ text: "what I said" }) });
    });

    expect(setValue).toHaveBeenCalled();
    expect(result.current.transcriptionError).toBeNull();
  });

  function failTranscription(error: string) {
    fetchMock.mockResolvedValue({ ok: false, json: async () => ({ error }) });
  }

  async function recordAndStop() {
    const result = await renderRecording();
    await act(async () => result.current.stopRecording());
    return result;
  }

  async function sentAudio(call: number) {
    const body = fetchMock.mock.calls[call][1].body as FormData;
    return (body.get("audio") as Blob).text();
  }
});
