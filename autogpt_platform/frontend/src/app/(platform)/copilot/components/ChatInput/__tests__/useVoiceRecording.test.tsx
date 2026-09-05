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

function spaceKeydown(isComposing = false) {
  return {
    key: " ",
    nativeEvent: { key: " ", isComposing },
    preventDefault: vi.fn(),
  } as unknown as React.KeyboardEvent<HTMLTextAreaElement>;
}

function renderVoice(value = "") {
  return renderHook(() => useVoiceRecording({ setValue: vi.fn(), value }));
}

beforeEach(() => {
  getUserMedia.mockRejectedValue(new Error("no microphone"));
  Object.defineProperty(navigator, "mediaDevices", {
    configurable: true,
    value: { getUserMedia },
  });
});

afterEach(() => {
  vi.clearAllMocks();
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
});
