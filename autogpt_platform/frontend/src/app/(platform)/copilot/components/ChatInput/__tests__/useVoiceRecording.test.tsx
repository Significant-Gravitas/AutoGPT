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

const getUserMedia = vi.fn();
const recorderStop = vi.fn();

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
  stop = recorderStop;
}

async function renderRecording() {
  getUserMedia.mockResolvedValue({ getTracks: () => [] });
  const { result } = renderVoice();

  await act(async () => {
    result.current.handleKeyDown(spaceKeydown());
  });
  await waitFor(() => expect(result.current.isRecording).toBe(true));

  return result;
}

function renderVoice(value = "") {
  return renderHook(() => useVoiceRecording({ setValue: vi.fn(), value }));
}

beforeEach(() => {
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
